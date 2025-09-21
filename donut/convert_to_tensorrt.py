import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple
import torch

from model import DonutModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def export_to_onnx(model: DonutModel, onnx_path: str, image_size: Tuple[int, int] = (1280, 960)):
    """Экспортирует модель Donut в формат ONNX."""
    logger.info("Экспорт модели в ONNX...")
    dummy_image = torch.randn(1, 3, image_size[0], image_size[1])
    dummy_decoder_input_ids = torch.tensor([[model.model.config.decoder_start_token_id]], dtype=torch.long)

    encoder_onnx_path = onnx_path.replace('.onnx', '_encoder.onnx')
    logger.info(f"Экспорт encoder в {encoder_onnx_path}")

    torch.onnx.export(
        model.model.encoder,
        dummy_image,
        encoder_onnx_path,
        input_names=['pixel_values'],
        output_names=['encoder_outputs'],
        dynamic_axes={'pixel_values': {0: 'batch_size'}},
        opset_version=14,
        verbose=False
    )

    with torch.no_grad():
        encoder_outputs = model.model.encoder(dummy_image)

    decoder_onnx_path = onnx_path.replace('.onnx', '_decoder.onnx')
    logger.info(f"Экспорт decoder в {decoder_onnx_path}")

    torch.onnx.export(
        model.model.decoder,
        {
            "input_ids": dummy_decoder_input_ids,
            "encoder_hidden_states": encoder_outputs.last_hidden_state
        },
        decoder_onnx_path,
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'encoder_hidden_states': {0: 'batch_size'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=14,
        verbose=False
    )

    logger.info("Экспорт в ONNX завершен")


def convert_onnx_to_tensorrt(onnx_path: str, engine_path: str, precision: str = 'fp16', max_batch_size: int = 1, image_size: Tuple[int, int] = (1280, 960), hidden_size: int = 1024, downsample_factor: int = 32, max_length: int = 768):
    """Конвертирует ONNX модель в TensorRT engine."""
    logger.info(f"Конвертация {onnx_path} в TensorRT engine...")

    try:
        import tensorrt as trt
        from tensorrt import Logger
    except ImportError:
        logger.error("TensorRT не установлен. Установите tensorrt: pip install tensorrt")
        return False

    TRT_LOGGER = Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return False


    config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    
    if 'encoder' in onnx_path:
        # Для encoder: pixel_values (batch, 3, H, W)
        profile.set_shape(
            'pixel_values',
            (1, 3, image_size[0], image_size[1]),
            (1, 3, image_size[0], image_size[1]),
            (max_batch_size, 3, image_size[0], image_size[1])
        )
    else:
        # Для decoder: input_ids (batch, seq), encoder_hidden_states (batch, seq_enc, hidden)
        seq_enc = (image_size[0] // downsample_factor) * (image_size[1] // downsample_factor)
        profile.set_shape(
            'input_ids',
            (1, 1),
            (1, 1),
            (max_batch_size, max_length)
        )
        profile.set_shape(
            'encoder_hidden_states',
            (1, seq_enc, hidden_size),
            (1, seq_enc, hidden_size),
            (max_batch_size, seq_enc, hidden_size)
        )
    
    config.add_optimization_profile(profile)

    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Используется FP16 точность")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("Используется INT8 точность")
    else:
        logger.info("Используется FP32 точность")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Не удалось создать TensorRT engine")
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine сохранен в {engine_path}")
    return True


def convert_model_to_tensorrt(
    model_path: str,
    max_batch_size: int = 1
):
    """Полная конвертация модели в TensorRT."""
    output_dir = Path(model_path) / f"eninge"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(model_path) / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    image_size = tuple(config['encoder']['image_size'])
    hidden_size = config['encoder']['hidden_size']
    patch_size = config['encoder']['patch_size']
    depths = config['encoder']['depths']
    downsample_factor = patch_size * (2 ** (len(depths) - 1))
    
    donut_config_path = Path(model_path) / "donut_config.json"
    if donut_config_path.exists():
        with open(donut_config_path, 'r', encoding='utf-8') as f:
            donut_config = json.load(f)
        max_length = donut_config.get('max_length', 768)
        model_precision = donut_config.get('precision', 'fp32')
    else:
        max_length = 768
        model_precision = 'fp32'
    
    if model_precision == 'bf16':
        precision = 'fp16'
    else:
        precision = 'fp32'

    logger.info(f"Загрузка модели из {model_path}")
    model = DonutModel.from_pretrained(
        model_path,
        device='cpu',
        precision='fp32',
        max_length=max_length,
        image_size=image_size
    )

    base_name = 'donut'
    onnx_path = output_dir / f"{base_name}.onnx"
    
    export_to_onnx(model, str(onnx_path), image_size)

    encoder_onnx_path = onnx_path.with_name(f"{base_name}_encoder.onnx")
    decoder_onnx_path = onnx_path.with_name(f"{base_name}_decoder.onnx")
    
    encoder_engine_path = output_dir / f"{base_name}_encoder.engine"
    decoder_engine_path = output_dir / f"{base_name}_decoder.engine"
    
    success_encoder = convert_onnx_to_tensorrt(str(encoder_onnx_path), str(encoder_engine_path), precision, max_batch_size, image_size, hidden_size, downsample_factor, max_length)
    success_decoder = convert_onnx_to_tensorrt(str(decoder_onnx_path), str(decoder_engine_path), precision, max_batch_size, image_size, hidden_size, downsample_factor, max_length)
    
    success = success_encoder and success_decoder

    if success:
        config = {
            'base_name': base_name,
            'precision': precision,
            'image_size': image_size,
            'max_batch_size': max_batch_size,
            'hidden_size': hidden_size,
            'downsample_factor': downsample_factor,
            'max_length': max_length,
            'task_start_token': model.task_start_token,
            'prompt_end_token': model.prompt_end_token,
            'decoder_start_token_id': model.model.config.decoder_start_token_id
        }

        config_path = output_dir / f"tensorrt_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Конфигурация сохранена в {config_path}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Конвертация модели Donut в TensorRT")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Путь к модели Donut")
    parser.add_argument("--max_batch_size", type=int, default=1,
                       help="Максимальный размер пакета")

    args = parser.parse_args()

    success = convert_model_to_tensorrt(
        args.model_path,
        args.max_batch_size
    )

    if success:
        logger.info("Конвертация завершена успешно")
        return 0
    else:
        logger.error("Ошибка конвертации")
        return 1


if __name__ == "__main__":
    sys.exit(main())