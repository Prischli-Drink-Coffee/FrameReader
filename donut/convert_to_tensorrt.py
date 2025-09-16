import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from transformers import DonutProcessor

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

    # Создаем фиктивные входные данные
    dummy_image = torch.randn(1, 3, image_size[0], image_size[1])
    dummy_decoder_input_ids = torch.tensor([[model.model.config.decoder_start_token_id]], dtype=torch.long)

    # Для VisionEncoderDecoderModel нужно экспортировать отдельно encoder и decoder
    # Сначала encoder
    encoder_onnx_path = onnx_path.replace('.onnx', '_encoder.onnx')
    logger.info(f"Экспорт encoder в {encoder_onnx_path}")

    torch.onnx.export(
        model.model.encoder,
        dummy_image,
        encoder_onnx_path,
        input_names=['pixel_values'],
        output_names=['encoder_outputs'],
        dynamic_axes={'pixel_values': {0: 'batch_size'}},
        opset_version=13,
        verbose=False
    )

    # Получаем выход encoder для decoder
    with torch.no_grad():
        encoder_outputs = model.model.encoder(dummy_image)

    # Decoder
    decoder_onnx_path = onnx_path.replace('.onnx', '_decoder.onnx')
    logger.info(f"Экспорт decoder в {decoder_onnx_path}")

    torch.onnx.export(
        model.model.decoder,
        (encoder_outputs.last_hidden_state, dummy_decoder_input_ids),
        decoder_onnx_path,
        input_names=['encoder_outputs', 'decoder_input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'encoder_outputs': {0: 'batch_size'},
            'decoder_input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=13,
        verbose=False
    )

    logger.info("Экспорт в ONNX завершен")


def convert_onnx_to_tensorrt(onnx_path: str, engine_path: str, precision: str = 'fp16', max_batch_size: int = 1):
    """Конвертирует ONNX модель в TensorRT engine."""
    logger.info(f"Конвертация {onnx_path} в TensorRT engine...")

    try:
        import tensorrt as trt
        from tensorrt import Logger, Runtime
    except ImportError:
        logger.error("TensorRT не установлен. Установите tensorrt: pip install tensorrt")
        return False

    TRT_LOGGER = Logger(trt.Logger.WARNING)

    # Создаем builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Парсим ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            return False

    # Создаем конфигурацию
    config = builder.create_builder_config()

    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Используется FP16 точность")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("Используется INT8 точность")
    else:
        logger.info("Используется FP32 точность")

    # Оптимизация
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Строим engine
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Не удалось создать TensorRT engine")
        return False

    # Сохраняем engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine сохранен в {engine_path}")
    return True


def convert_model_to_tensorrt(
    model_path: str,
    output_dir: str,
    precision: str = 'fp16',
    image_size: Tuple[int, int] = (1280, 960),
    max_batch_size: int = 1
):
    """Полная конвертация модели в TensorRT."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем модель
    logger.info(f"Загрузка модели из {model_path}")
    model = DonutModel.from_pretrained(
        model_path,
        device='cpu',  # Для экспорта используем CPU
        precision='fp32',
        max_length=768,
        image_size=image_size
    )

    # Пути для файлов
    base_name = Path(model_path).name
    onnx_path = output_dir / f"{base_name}.onnx"
    
    # Экспорт в ONNX
    export_to_onnx(model, str(onnx_path), image_size)

    # Конвертация encoder и decoder отдельно
    encoder_onnx_path = onnx_path.with_name(f"{base_name}_encoder.onnx")
    decoder_onnx_path = onnx_path.with_name(f"{base_name}_decoder.onnx")
    
    encoder_engine_path = output_dir / f"{base_name}_encoder.engine"
    decoder_engine_path = output_dir / f"{base_name}_decoder.engine"
    
    success_encoder = convert_onnx_to_tensorrt(str(encoder_onnx_path), str(encoder_engine_path), precision, max_batch_size)
    success_decoder = convert_onnx_to_tensorrt(str(decoder_onnx_path), str(decoder_engine_path), precision, max_batch_size)
    
    success = success_encoder and success_decoder

    if success:
        # Сохраняем конфигурацию
        config = {
            'model_path': model_path,
            'precision': precision,
            'image_size': image_size,
            'max_batch_size': max_batch_size,
            'onnx_path': str(onnx_path),
            'encoder_onnx_path': str(encoder_onnx_path),
            'decoder_onnx_path': str(decoder_onnx_path),
            'engine_path': str(encoder_engine_path),  # Для совместимости
            'encoder_engine_path': str(encoder_engine_path),
            'decoder_engine_path': str(decoder_engine_path),
            'task_start_token': model.task_start_token,
            'prompt_end_token': model.prompt_end_token,
            'max_length': model.max_length,
            'decoder_start_token_id': model.model.config.decoder_start_token_id
        }

        config_path = output_dir / f"{base_name}_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Конфигурация сохранена в {config_path}")

    return success


def main():
    parser = argparse.ArgumentParser(description="Конвертация модели Donut в TensorRT")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Путь к модели Donut")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Директория для сохранения результатов")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                       help="Точность TensorRT")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384],
                       help="Размер изображения [высота, ширина]")
    parser.add_argument("--max_batch_size", type=int, default=1,
                       help="Максимальный размер пакета")

    args = parser.parse_args()

    success = convert_model_to_tensorrt(
        args.model_path,
        args.output_dir,
        args.precision,
        tuple(args.image_size),
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