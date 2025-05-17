import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import numpy as np
from tqdm.auto import tqdm

from model import DonutModel
from dataset import DonutDataModule

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX и/или ONNX Runtime не установлены. Конвертация в ONNX будет недоступна.")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT не установлен. Конвертация в TensorRT будет недоступна.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, calibration_data: Dict[str, np.ndarray], batch_size, input_names: List[str]):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_names = input_names
        self.batch_size = batch_size
        self.cache_file = Path(f"./calibration_{'_'.join(input_names)}_bs{batch_size}.cache")

        self.data = {}
        for name in input_names:
            if name in calibration_data and len(calibration_data[name]) > 0:
                data_shape = calibration_data[name].shape

                if data_shape[0] != batch_size and data_shape[0] > 0:
                     repeat_factor = (batch_size + data_shape[0] - 1) // data_shape[0]
                     self.data[name] = np.repeat(calibration_data[name], repeat_factor, axis=0).astype(np.float32 if name != "decoder_input_ids" else np.int32)
                     logger.info(f"Repeated calibration data for {name}. Original shape {data_shape}, new shape {self.data[name].shape}")
                else:
                    self.data[name] = calibration_data[name].astype(np.float32 if name != "decoder_input_ids" else np.int32)


                logger.info(f"Успешно загружены калибровочные данные для {name}: {self.data[name].shape}")
            else:
                logger.warning(f"Не найдены данные для {name} в калибровочных данных или данные пусты")
                self.data[name] = np.array([])

        sample_counts = [len(self.data[name]) for name in input_names if len(self.data[name]) > 0]
        if not sample_counts:
             logger.error("Нет данных для калибровки для любого из входов.")
             self.data = {}
             self.num_total_samples = 0
        else:
            min_samples = min(sample_counts)
            if not all(count == min_samples for count in sample_counts):
                 logger.warning("Разное количество образцов в калибровочных данных для разных входов. Обрезка до наименьшего количества.")
                 for name in input_names:
                     if len(self.data[name]) > min_samples:
                         self.data[name] = self.data[name][:min_samples]
                         logger.warning(f"Калибровочные данные для {name} обрезаны до {min_samples} образцов.")
            self.num_total_samples = min_samples

        self.current_index = 0
        self.device_inputs = {}

        if self.num_total_samples > 0:
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                for name in input_names:
                    if name in self.data and len(self.data[name]) > 0:
                        self.device_inputs[name] = cuda.mem_alloc(self.data[name][0:self.batch_size].nbytes)
                        logger.info(f"Выделена память для калибровочных данных {name}: {self.data[name][0:self.batch_size].nbytes} байт")
                    else:
                        self.device_inputs[name] = None
                        logger.warning(f"Не удалось выделить память для {name}, данные отсутствуют")

            except Exception as e:
                logger.error(f"Ошибка при выделении памяти для калибратора: {e}", exc_info=True)
        else:
             logger.warning("Нет доступных калибровочных данных после загрузки и проверки.")


    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.num_total_samples == 0:
             logger.info("Нет доступных калибровочных данных. Калибровка пропущена.")
             return None

        if self.current_index + self.batch_size > self.num_total_samples:
            logger.info(f"Калибровка завершена, обработано {self.current_index} образцов из {self.num_total_samples}")
            return None

        try:
            batch_start = self.current_index
            batch_end = self.current_index + self.batch_size

            device_bindings = []
            name_to_data = {name: self.data.get(name) for name in self.input_names}
            name_to_device_input = {name: self.device_inputs.get(name) for name in self.input_names}


            for name in names:
                if name in name_to_data and name_to_device_input.get(name) is not None:
                    current_batch_data = name_to_data[name][batch_start:batch_end]
                    import pycuda.driver as cuda
                    cuda.memcpy_htod(name_to_device_input[name], current_batch_data.ravel())
                    device_bindings.append(int(name_to_device_input[name]))
                else:
                    logger.error(f"Не удалось найти данные или device_input для ожидаемого входа TensorRT: {name}")
                    return None

            self.current_index += self.batch_size
            return device_bindings

        except Exception as e:
            logger.error(f"Ошибка при подготовке батча для калибровки: {e}", exc_info=True)
            return None

    def read_calibration_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                logger.info(f"Загрузка данных калибровки из {self.cache_file}")
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "wb") as f:
                logger.info(f"Сохранение данных калибровки в {self.cache_file}")
                f.write(cache)
        except Exception as e:
            logger.error(f"Ошибка при записи калибровочного кеша в {self.cache_file}: {e}", exc_info=True)


class DonutOptimizer:

    def __init__(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 64,
        batch_size: int = 1,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Инициализация оптимизатора для модели {model_path}")
        logger.info(f"Устройство: {self.device}, максимальная длина: {max_length}, размер батча для оптимизации: {batch_size}")

        self.model = DonutModel.from_pretrained(
            model_path,
            device=self.device,
            max_length=max_length
        )
        self.model.eval()
        logger.info(f"Модель загружена успешно")

        self.task_start_token = self.model.task_start_token
        self.prompt_end_token = getattr(self.model, 'prompt_end_token', self.task_start_token)
        self.processor = self.model.processor

        try:
            with torch.no_grad():
                dummy_input = torch.randn(self.batch_size, 3, 384, 384, device=self.device)
                encoder_output = self.model.model.encoder(dummy_input).last_hidden_state
                self.encoder_hidden_size = encoder_output.shape[-1]
                self.encoder_seq_length = encoder_output.shape[1]
                logger.info(f"Обнаружена размерность скрытого состояния энкодера: {self.encoder_hidden_size}")
                logger.info(f"Обнаружена длина последовательности выхода энкодера: {self.encoder_seq_length} (для батча {self.batch_size})")
        except Exception as e:
            logger.warning(f"Не удалось определить выходные размеры энкодера с батчем {self.batch_size}. Использование батча 1 для определения: {e}", exc_info=True)
            try:
                 with torch.no_grad():
                    dummy_input_single = torch.randn(1, 3, 384, 384, device=self.device)
                    encoder_output_single = self.model.model.encoder(dummy_input_single).last_hidden_state
                    self.encoder_hidden_size = encoder_output_single.shape[-1]
                    self.encoder_seq_length = encoder_output_single.shape[1]
                    logger.info(f"Определены размеры скрытого состояния ({self.encoder_hidden_size}) и длины последовательности ({self.encoder_seq_length}) с использованием батча 1.")
            except Exception as e_single:
                 logger.error(f"Критическая ошибка: Не удалось определить выходные размеры энкодера даже с батчем 1: {e_single}", exc_info=True)
                 self.encoder_hidden_size = 1024
                 self.encoder_seq_length = 144
                 logger.warning(f"Использование значений по умолчанию для размеров энкодера: hidden_size={self.encoder_hidden_size}, seq_length={self.encoder_seq_length}")

        try:
            if hasattr(self.model.model.decoder, 'embed_tokens'):
                self.vocab_size = self.model.model.decoder.embed_tokens.weight.shape[0]
            elif hasattr(self.model.model.decoder, 'model') and hasattr(self.model.model.decoder.model, 'embed_tokens'):
                self.vocab_size = self.model.model.decoder.model.embed_tokens.weight.shape[0]
            elif hasattr(self.model.model.decoder, 'model') and hasattr(self.model.model.decoder.model, 'shared'):
                self.vocab_size = self.model.model.decoder.model.shared.weight.shape[0]
            elif hasattr(self.model.model.decoder, 'model') and hasattr(self.model.model.decoder.model.decoder, 'embed_tokens'):
                self.vocab_size = self.model.model.decoder.model.decoder.embed_tokens.weight.shape[0]
            else:
                self.vocab_size = self.model.model.decoder.config.vocab_size
                if not self.vocab_size:
                    self.vocab_size = 50265
                    logger.warning(f"Не удалось определить размер словаря модели, используем значение по умолчанию: {self.vocab_size}")
        except Exception as e:
            self.vocab_size = 50265
            logger.warning(f"Ошибка при определении размера словаря: {e}. Используем значение по умолчанию: {self.vocab_size}")

        logger.info(f"Размер словаря модели: {self.vocab_size}")

    def prepare_sample_input(self, image_size=(384, 384)):
        pixel_values = torch.randn(
            self.batch_size, 3, image_size[0], image_size[1],
            device=self.device
        )
        input_prompt = self.task_start_token
        decoder_input_ids = self.processor.tokenizer(
            input_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].to(self.device)

        if decoder_input_ids.shape[0] != self.batch_size:
             decoder_input_ids = decoder_input_ids.repeat(self.batch_size, 1)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids
        }


    def convert_to_onnx(self, dynamic_axes=True, opset_version=14):
        if not ONNX_AVAILABLE:
            logger.error("ONNX не установлен. Установите onnx и onnxruntime.")
            return False

        logger.info("Подготовка к экспорту в ONNX...")
        encoder_path = self.output_dir / "encoder.onnx"
        decoder_path = self.output_dir / "decoder.onnx"

        try:
            sample_inputs = self.prepare_sample_input()
            logger.info("Экспорт энкодера в ONNX...")

            dynamic_axes_encoder = None
            if dynamic_axes:
                dynamic_axes_encoder = {
                    'pixel_values': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'encoder_outputs': {0: 'batch_size', 1: 'sequence'}
                }

            class EncoderWrapper(torch.nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder

                def forward(self, pixel_values):
                    return self.encoder(pixel_values).last_hidden_state

            encoder_wrapper = EncoderWrapper(self.model.model.encoder)
            encoder_wrapper.eval()

            with torch.no_grad():
                torch.onnx.export(
                    encoder_wrapper,
                    (sample_inputs["pixel_values"],),
                    encoder_path,
                    export_params=True,
                    opset_version=opset_version,
                    input_names=['pixel_values'],
                    output_names=['encoder_outputs'],
                    dynamic_axes=dynamic_axes_encoder,
                    verbose=False,
                    do_constant_folding=True,
                )

            logger.info("Экспорт декодера в ONNX...")

            with torch.no_grad():
                encoder_outputs = self.model.model.encoder(
                    sample_inputs["pixel_values"]
                ).last_hidden_state


            dynamic_axes_decoder = None
            if dynamic_axes:
                dynamic_axes_decoder = {
                    'decoder_input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'encoder_hidden_states': {0: 'batch_size', 1: 'encoder_sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                }

            class DecoderWrapper(torch.nn.Module):
                def __init__(self, decoder):
                    super().__init__()
                    self.decoder = decoder

                def forward(self, decoder_input_ids, encoder_hidden_states):
                    decoder_outputs = self.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=True
                    )
                    return decoder_outputs.logits

            decoder_wrapper = DecoderWrapper(self.model.model.decoder)
            decoder_wrapper.eval()

            with torch.no_grad():
                torch.onnx.export(
                    decoder_wrapper,
                    (sample_inputs["decoder_input_ids"], encoder_outputs),
                    decoder_path,
                    export_params=True,
                    opset_version=opset_version,
                    input_names=['decoder_input_ids', 'encoder_hidden_states'],
                    output_names=['logits'],
                    dynamic_axes=dynamic_axes_decoder,
                    verbose=False,
                    do_constant_folding=True,
                )

            logger.info(f"Модель успешно экспортирована в ONNX: энкодер в {encoder_path}, декодер в {decoder_path}")

            model_info = {
                "encoder_hidden_size": self.encoder_hidden_size,
                "encoder_seq_length": self.encoder_seq_length,
                "vocab_size": self.vocab_size,
                "optimization_batch_size": self.batch_size,
                "max_length": self.max_length,
                "model_path": str(self.model_path),
                "onnx_encoder_path": str(encoder_path),
                "onnx_decoder_path": str(decoder_path)
            }

            import json
            with open(self.output_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=4)

            logger.info(f"Информация о модели сохранена в {self.output_dir / 'model_info.json'}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при экспорте в ONNX: {e}", exc_info=True)
            return False

    def convert_to_tensorrt_direct(self, calibration_data=None, precision="fp32", fp16_first=True):
        if not TRT_AVAILABLE:
            logger.error("TensorRT не установлен. Установите tensorrt.")
            return False

        if precision == "int8" and fp16_first:
            logger.info("Сначала создаем fp16 модель для проверки совместимости перед INT8...")
            success = self.convert_to_tensorrt_direct(calibration_data=None, precision="fp16", fp16_first=False)
            if not success:
                logger.error("Не удалось создать fp16 модель. Отменяем создание int8 модели.")
                return False
            else:
                 logger.info("Создание fp16 модели успешно.")


        logger.info(f"Начало прямой конвертации в TensorRT с точностью {precision}...")

        onnx_encoder_path = self.output_dir / "encoder.onnx"
        onnx_decoder_path = self.output_dir / "decoder.onnx"
        tensorrt_encoder_path = self.output_dir / f"encoder_{precision}.engine"
        tensorrt_decoder_path = self.output_dir / f"decoder_{precision}.engine"

        if not onnx_encoder_path.exists() or not onnx_decoder_path.exists():
             logger.warning("ONNX файлы не найдены. Попытка экспортировать...")
             if not self.convert_to_onnx():
                  logger.error("Не удалось экспортировать ONNX файлы. Невозможно продолжить конвертацию в TensorRT.")
                  return False
             logger.info("ONNX файлы успешно экспортированы.")


        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt_version = trt.__version__
        logger.info(f"Используем TensorRT версии {trt_version}")

        logger.info("Конвертация энкодера в TensorRT...")
        try:
            creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(creation_flag) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:

                with open(onnx_encoder_path, 'rb') as model:
                    if not parser.parse(model.read()):
                        logger.error("Ошибка при парсинге ONNX модели энкодера")
                        for error in range(parser.num_errors):
                            logger.error(parser.get_error(error))
                        return False

                config = builder.create_builder_config()

                try:
                    memory_size = 8 << 30  # 8GB
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_size)
                    logger.info(f"Установлен лимит рабочей памяти для TensorRT 10.x: {memory_size/(1<<30):.2f} GB")
                except (AttributeError, TypeError):
                    config.max_workspace_size = 8 << 30  # 8GB
                    logger.info(f"Установлен лимит рабочей памяти для старых версий TensorRT: {(8<<30)/(1<<30):.2f} GB")

                if precision == "int8":
                    if not hasattr(builder, 'platform_has_fast_int8') or not builder.platform_has_fast_int8:
                        logger.warning("Платформа не поддерживает быстрые вычисления int8")

                    config.set_flag(trt.BuilderFlag.INT8)

                    if calibration_data is not None and "pixel_values" in calibration_data and len(calibration_data["pixel_values"]) > 0:
                        logger.info("Использование калибровочных данных для int8 квантизации энкодера...")
                        try:
                            calibrator = Int8Calibrator({"pixel_values": calibration_data["pixel_values"]}, self.batch_size, ["pixel_values"])
                            config.int8_calibrator = calibrator
                            logger.info("Калибратор int8 для энкодера установлен успешно")
                        except Exception as e:
                            logger.error(f"Ошибка при создании калибратора для энкодера: {e}", exc_info=True)
                            logger.warning("Продолжаем без калибратора для энкодера, что может снизить качество int8 квантизации")
                    else:
                         if precision == "int8":
                            logger.warning("Калибровочные данные для энкодера не предоставлены или не содержат 'pixel_values'. INT8 конвертация энкодера может быть неоптимальной.")


                elif precision == "fp16":
                    if not hasattr(builder, 'platform_has_fast_fp16') or not builder.platform_has_fast_fp16:
                        logger.warning("Платформа не поддерживает быстрые вычисления fp16")
                    config.set_flag(trt.BuilderFlag.FP16)

                profile = builder.create_optimization_profile()

                min_img_size = (224, 224)
                opt_img_size = (384, 384)
                max_img_size = (512, 512)

                min_batch = 1
                opt_batch = self.batch_size
                max_batch = self.batch_size


                min_shape = (min_batch, 3, min_img_size[0], min_img_size[1])
                opt_shape = (opt_batch, 3, opt_img_size[0], opt_img_size[1])
                max_shape = (max_batch, 3, max_img_size[0], max_img_size[1])

                input_name = network.get_input(0).name

                logger.info(f"Профиль оптимизации энкодера: min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)

                logger.info("Сборка TensorRT движка для энкодера (это может занять некоторое время)...")
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    logger.error("Не удалось создать сериализованный движок TensorRT для энкодера")
                    return False

                with open(tensorrt_encoder_path, 'wb') as f:
                    f.write(serialized_engine)

                logger.info(f"Эnкодер сохранен в {tensorrt_encoder_path}")

        except Exception as e:
             logger.error(f"Ошибка при конвертации энкодера в TensorRT: {e}", exc_info=True)
             return False


        logger.info("Конвертация декодера в TensorRT...")
        try:
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(creation_flag) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:

                with open(onnx_decoder_path, 'rb') as model:
                    if not parser.parse(model.read()):
                        logger.error("Ошибка при парсинге ONNX модели декодера")
                        for error in range(parser.num_errors):
                            logger.error(parser.get_error(error))
                        return False

                config = builder.create_builder_config()

                try:
                    memory_size = 8 << 30  # 8GB
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_size)
                except (AttributeError, TypeError):
                    config.max_workspace_size = 8 << 30  # 8GB

                if precision == "int8":
                    config.set_flag(trt.BuilderFlag.INT8)

                    if calibration_data is not None and "decoder_input_ids" in calibration_data and "encoder_hidden_states" in calibration_data \
                        and len(calibration_data["decoder_input_ids"]) > 0 and len(calibration_data["encoder_hidden_states"]) > 0:
                        logger.info("Использование калибровочных данных для int8 квантизации декодера...")
                        try:
                            decoder_input_names = []
                            for i in range(network.num_inputs):
                                name = network.get_input(i).name
                                if "input_ids" in name or "decoder_input_ids" in name:
                                    decoder_input_names.append(name)
                                elif "hidden_states" in name or "encoder_hidden_states" in name:
                                    decoder_input_names.append(name)

                            if len(decoder_input_names) != 2:
                                 logger.error(f"Не удалось определить два входных имени для декодера из ONNX. Найдено: {decoder_input_names}")
                                 decoder_input_names = ["decoder_input_ids", "encoder_hidden_states"]
                                 logger.warning(f"Использование имен по умолчанию для входов декодера: {decoder_input_names}. Убедитесь, что порядок соответствует ONNX.")

                            calibrator = Int8Calibrator(
                                {
                                    decoder_input_names[0]: calibration_data["decoder_input_ids"],
                                    decoder_input_names[1]: calibration_data["encoder_hidden_states"]
                                },
                                self.batch_size,
                                decoder_input_names
                            )
                            config.int8_calibrator = calibrator
                            logger.info("Калибратор int8 для декодера установлен успешно")
                        except Exception as e:
                            logger.error(f"Ошибка при создании калибратора для декодера: {e}", exc_info=True)
                            logger.warning("Продолжаем без калибратора для декодера, что может снизить качество int8 квантизации")
                    else:
                         if precision == "int8":
                             logger.warning("Калибровочные данные для декодера не предоставлены или не содержат необходимые ключи/данные. INT8 конвертация декодера может быть неоптимальной.")

                elif precision == "fp16":
                    config.set_flag(trt.BuilderFlag.FP16)

                input_ids_name = None
                hidden_states_name = None
                for i in range(network.num_inputs):
                    name = network.get_input(i).name
                    if "input_ids" in name or "decoder_input_ids" in name:
                        input_ids_name = name
                    elif "hidden_states" in name or "encoder_hidden_states" in name:
                        hidden_states_name = name

                if input_ids_name is None or hidden_states_name is None:
                    logger.error(f"Не удалось определить имена входов декодера из ONNX для профиля: {input_ids_name}, {hidden_states_name}")
                    input_ids_name = "decoder_input_ids"
                    hidden_states_name = "encoder_hidden_states"
                    logger.warning(f"Используем имена по умолчанию для профиля декодера: input_ids={input_ids_name}, hidden_states={hidden_states_name}")

                logger.info(f"Имена входов декодера для профиля оптимизации: input_ids={input_ids_name}, hidden_states={hidden_states_name}")


                profile = builder.create_optimization_profile()

                min_batch = 1
                opt_batch = self.batch_size
                max_batch = self.batch_size

                min_seq_len = 1
                opt_seq_len = self.max_length // 2
                max_seq_len = self.max_length


                profile.set_shape(input_ids_name,
                                 (min_batch, min_seq_len),
                                 (opt_batch, opt_seq_len),
                                 (max_batch, max_seq_len))

                hidden_size = self.encoder_hidden_size

                enc_seq_len = self.encoder_seq_length
                min_enc_seq_len = enc_seq_len
                opt_enc_seq_len = enc_seq_len
                max_enc_seq_len = enc_seq_len

                logger.info(f"Размеры последовательности энкодера для профиля декодера: min={min_enc_seq_len}, opt={opt_enc_seq_len}, max={max_enc_seq_len}")

                profile.set_shape(hidden_states_name,
                                 (min_batch, min_enc_seq_len, hidden_size),
                                 (opt_batch, opt_enc_seq_len, hidden_size),
                                 (max_batch, max_enc_seq_len, hidden_size))

                config.add_optimization_profile(profile)

                logger.info("Профиль оптимизации декодера настроен, начинаем сборку...")

                logger.info("Сборка TensorRT движка для декодера (это может занять некоторое время)...")
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    logger.error("Не удалось создать сериализованный движок TensorRT для декодера")
                    if precision == "int8" and calibration_data is not None:
                         logger.error("Сборка INT8 движка декодера не удалась. Это может быть связано с проблемами калибровки.")
                    return False

                with open(tensorrt_decoder_path, 'wb') as f:
                    f.write(serialized_engine)

                logger.info(f"Декодер сохранен в {tensorrt_decoder_path}")

            logger.info(f"Конвертация в TensorRT успешно завершена")

            if tensorrt_encoder_path.exists() and tensorrt_decoder_path.exists():
                try:
                    onnx_encoder_size_mb = onnx_encoder_path.stat().st_size / (1024 * 1024)
                    engine_encoder_size_mb = tensorrt_encoder_path.stat().st_size / (1024 * 1024)

                    onnx_decoder_size_mb = onnx_decoder_path.stat().st_size / (1024 * 1024)
                    engine_decoder_size_mb = tensorrt_decoder_path.stat().st_size / (1024 * 1024)

                    logger.info(f"Размер энкодера: ONNX {onnx_encoder_size_mb:.2f} MB -> TensorRT {engine_encoder_size_mb:.2f} MB")
                    logger.info(f"Размер декодера: ONNX {onnx_decoder_size_mb:.2f} MB -> TensorRT {engine_decoder_size_mb:.2f} MB")
                except Exception as size_e:
                     logger.warning(f"Не удалось получить размеры файлов после конвертации: {size_e}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при конвертации декодера в TensorRT: {e}", exc_info=True)
            return False


    def create_calibration_data(self, data_dir, split="val", num_samples=100):
        logger.info(f"Подготовка калибровочных данных из набора {split}...")

        try:
            data_module = DonutDataModule(
                processor=self.processor,
                data_dir=data_dir,
                batch_size=1,
                num_workers=4,
                max_length=self.max_length,
                task_start_token=self.task_start_token,
                prompt_end_token=self.prompt_end_token,
                image_size=(384, 384),
                apply_augmentation=False,
                cache_images=False
            )

            if split == "val":
                dataloader = data_module.val_dataloader()
            elif split == "train":
                dataloader = data_module.train_dataloader()
            elif split == "test":
                dataloader = data_module.test_dataloader()
            else:
                logger.error(f"Неизвестное разделение данных для калибровки: {split}. Доступно: train, val, test.")
                return None


            if dataloader is None:
                logger.error(f"Загрузчик данных для разделения {split} не найден. Проверьте 'data_dir'.")
                return None

            calibration_data = {
                "pixel_values": [],
                "decoder_input_ids": [],
                "encoder_hidden_states": []
            }

            actual_samples = 0

            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, desc=f"Сбор калибровочных данных ({split})")):
                    if actual_samples >= num_samples:
                        break

                    if isinstance(batch, (list, tuple)) and len(batch) >= 1: # Expect at least pixel_values
                        pixel_values = batch[0].to(self.device)
                        input_prompt = self.task_start_token
                        decoder_input_ids = self.processor.tokenizer(
                            input_prompt,
                            add_special_tokens=False,
                            return_tensors="pt"
                        )["input_ids"].to(self.device)

                        if pixel_values.ndim == 3:
                             pixel_values = pixel_values.unsqueeze(0)
                        if decoder_input_ids.ndim == 1:
                             decoder_input_ids = decoder_input_ids.unsqueeze(0)

                    else:
                        logger.warning(f"Неожиданный формат batch (пропуск): {type(batch)}. Ожидается list или tuple с pixel_values.")
                        continue

                    encoder_outputs = self.model.model.encoder(pixel_values).last_hidden_state
                    calibration_data["pixel_values"].append(pixel_values.cpu().numpy())
                    calibration_data["decoder_input_ids"].append(decoder_input_ids.cpu().numpy())
                    calibration_data["encoder_hidden_states"].append(encoder_outputs.cpu().numpy())

                    actual_samples += pixel_values.shape[0] # Increment by the actual batch size (which is 1)

                    if actual_samples >= num_samples:
                        break

            for key in calibration_data:
                if calibration_data[key]:
                    calibration_data[key] = np.concatenate(calibration_data[key], axis=0)
                else:
                    logger.warning(f"Нет собранных данных для ключа '{key}'")
                    calibration_data[key] = np.array([])

            logger.info(f"Собрано {actual_samples} калибровочных образцов")

            calibration_path = self.output_dir / "calibration_data.npz"
            try:
                np.savez(calibration_path, **calibration_data)
                logger.info(f"Калибровочные данные сохранены в {calibration_path}")
            except Exception as save_e:
                 logger.error(f"Ошибка при сохранении калибровочных данных в {calibration_path}: {save_e}", exc_info=True)

            logger.info("Собранные калибровочные данные (формы):")
            for key, value in calibration_data.items():
                 logger.info(f"  {key}: {value.shape if value is not None else 'None'}")

            return calibration_data

        except Exception as e:
            logger.error(f"Ошибка при создании калибровочных данных: {e}", exc_info=True)
            return None


    def benchmark_trt_vs_original(self, precision="int8", num_runs=50):
        logger.info(f"Начало бенчмарка TensorRT ({precision}) vs PyTorch...")

        trt_encoder_path = self.output_dir / f"encoder_{precision}.engine"
        trt_decoder_path = self.output_dir / f"decoder_{precision}.engine"

        if not trt_encoder_path.exists() or not trt_decoder_path.exists():
            logger.error(f"Файлы TensorRT не найдены для точности {precision}. Сначала выполните конвертацию. Энкодер: {trt_encoder_path.exists()}, Декодер: {trt_decoder_path.exists()}")
            return

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            logger.error("Не удалось импортировать pycuda. Установите pycuda для выполнения бенчмарка.")
            return

        try:
            def load_engine(engine_path):
                with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                    return runtime.deserialize_cuda_engine(f.read())

            logger.info(f"Загрузка TensorRT энкодера из {trt_encoder_path}")
            trt_encoder = load_engine(trt_encoder_path)

            logger.info(f"Загрузка TensorRT декодера из {trt_decoder_path}")
            trt_decoder = load_engine(trt_decoder_path)

            trt_encoder_context = trt_encoder.create_execution_context()
            trt_decoder_context = trt_decoder.create_execution_context()

            sample_inputs = self.prepare_sample_input()

            logger.info("Benchmarking PyTorch model (full generation process)...")

            with torch.no_grad():
                for _ in range(5):
                    _ = self.model.generate(
                        sample_inputs["pixel_values"],
                        decoder_input_ids=sample_inputs["decoder_input_ids"],
                        max_length=self.max_length,
                        do_sample=False,
                        num_beams=1
                    )

            torch_times = []
            with torch.no_grad():
                for _ in tqdm(range(num_runs), desc="PyTorch (Generation)"):
                    start_time = time.time()

                    outputs = self.model.generate(
                        sample_inputs["pixel_values"],
                        decoder_input_ids=sample_inputs["decoder_input_ids"],
                        max_length=self.max_length,
                        do_sample=False,
                        num_beams=1,
                    )

                    end_time = time.time()
                    torch_times.append(end_time - start_time)

            avg_torch_time = sum(torch_times) / len(torch_times)
            logger.info(f"Среднее время выполнения PyTorch (генерация): {avg_torch_time*1000:.2f} мс")

            logger.info("Benchmarking TensorRT model (single encoder + single decoder pass)...")

            def run_trt_single_pass(encoder_context, decoder_context, pixel_values_np, decoder_input_ids_np):
                d_pixel_values = cuda.mem_alloc(pixel_values_np.nbytes)
                cuda.memcpy_htod(d_pixel_values, pixel_values_np.ravel())

                encoder_output_binding_idx = trt_encoder.get_binding_index("encoder_outputs")
                encoder_output_shape = trt_encoder_context.get_binding_shape(encoder_output_binding_idx)
                h_encoder_output = np.zeros(encoder_output_shape, dtype=np.float32)
                d_encoder_output = cuda.mem_alloc(h_encoder_output.nbytes)

                encoder_input_binding_idx = trt_encoder.get_binding_index("pixel_values")
                encoder_context.set_binding_shape(encoder_input_binding_idx, pixel_values_np.shape)

                encoder_bindings = [None] * trt_encoder.num_bindings
                for i in range(trt_encoder.num_bindings):
                     name = trt_encoder.get_binding_name(i)
                     if name == "pixel_values":
                          encoder_bindings[i] = int(d_pixel_values)
                     elif name == "encoder_outputs":
                          encoder_bindings[i] = int(d_encoder_output)
                     else:
                         logger.warning(f"Неизвестный биндинг энкодера: {name}")

                encoder_context.execute_v2(encoder_bindings)

                cuda.memcpy_dtoh(h_encoder_output, d_encoder_output)

                d_decoder_input_ids = cuda.mem_alloc(decoder_input_ids_np.nbytes)
                cuda.memcpy_htod(d_decoder_input_ids, decoder_input_ids_np.ravel())

                d_encoder_hidden_states = cuda.mem_alloc(h_encoder_output.nbytes)
                cuda.memcpy_htod(d_encoder_hidden_states, h_encoder_output.ravel())

                decoder_output_binding_idx = trt_decoder.get_binding_index("logits")
                decoder_output_shape = trt_decoder_context.get_binding_shape(decoder_output_binding_idx)
                h_decoder_output = np.zeros(decoder_output_shape, dtype=np.float32)
                d_decoder_output = cuda.mem_alloc(h_decoder_output.nbytes)

                decoder_input_ids_binding_idx = trt_decoder.get_binding_index("decoder_input_ids")
                encoder_hidden_states_binding_idx = trt_decoder.get_binding_index("encoder_hidden_states")

                decoder_context.set_binding_shape(decoder_input_ids_binding_idx, decoder_input_ids_np.shape)
                decoder_context.set_binding_shape(encoder_hidden_states_binding_idx, h_encoder_output.shape)

                decoder_bindings = [None] * trt_decoder.num_bindings
                for i in range(trt_decoder.num_bindings):
                    name = trt_decoder.get_binding_name(i)
                    if name == "decoder_input_ids":
                         decoder_bindings[i] = int(d_decoder_input_ids)
                    elif name == "encoder_hidden_states":
                         decoder_bindings[i] = int(d_encoder_hidden_states)
                    elif name == "logits":
                         decoder_bindings[i] = int(d_decoder_output)
                    else:
                        logger.warning(f"Неизвестный биндинг декодера: {name}")

                decoder_context.execute_v2(decoder_bindings)

                cuda.memcpy_dtoh(h_decoder_output, d_decoder_output)

                d_pixel_values.free()
                d_encoder_output.free()
                d_decoder_input_ids.free()
                d_encoder_hidden_states.free()
                d_decoder_output.free()

                return h_decoder_output

            pixel_values_np = sample_inputs["pixel_values"].cpu().numpy()
            decoder_input_ids_np = self.processor.tokenizer(
                self.task_start_token,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].repeat(self.batch_size, 1).cpu().numpy() # Match optimization batch size

            for _ in range(5):
                _ = run_trt_single_pass(
                    trt_encoder_context,
                    trt_decoder_context,
                    pixel_values_np,
                    decoder_input_ids_np
                )

            trt_times = []
            for _ in tqdm(range(num_runs), desc="TensorRT (Single Pass)"):
                start_time = time.time()

                logits = run_trt_single_pass(
                    trt_encoder_context,
                    trt_decoder_context,
                    pixel_values_np,
                    decoder_input_ids_np
                )

                end_time = time.time()
                trt_times.append(end_time - start_time)

            avg_trt_time = sum(trt_times) / len(trt_times)
            logger.info(f"Среднее время выполнения TensorRT (один проход): {avg_trt_time*1000:.2f} мс")

            speedup_estimate = avg_torch_time / avg_trt_time
            logger.info(f"Оценочное ускорение TensorRT (один проход) по сравнению с PyTorch (генерация): {speedup_estimate:.2f}x")


            benchmark_results = {
                "pytorch_avg_generation_time_ms": avg_torch_time * 1000,
                "tensorrt_avg_single_pass_time_ms": avg_trt_time * 1000,
                "estimated_speedup_single_pass": speedup_estimate,
                "num_runs": num_runs,
                "pytorch_model": str(self.model_path),
                "tensorrt_model": {
                    "encoder": str(trt_encoder_path),
                    "decoder": str(trt_decoder_path)
                },
                "precision": precision,
                "optimization_batch_size": self.batch_size,
                "encoder_hidden_size": self.encoder_hidden_size,
                "encoder_seq_length": self.encoder_seq_length,
                "vocab_size": self.vocab_size
            }

            import json
            with open(self.output_dir / f"benchmark_results_{precision}.json", 'w') as f:
                json.dump(benchmark_results, f, indent=4)

            logger.info(f"Результаты бенчмарка сохранены в {self.output_dir / f'benchmark_results_{precision}.json'}")

        except Exception as e:
            logger.error(f"Ошибка при выполнении бенчмарка: {e}", exc_info=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Оптимизация модели Donut с помощью ONNX, TensorRT и квантизации")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Путь к обученной модели Donut")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Директория для сохранения оптимизированной модели")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Директория с данными для калибровки квантизации int8")

    parser.add_argument("--max_length", type=int, default=64,
                        help="Максимальная длина последовательности токенов для генерации и калибровки")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Размер пакета для инференса и калибровки (TensorRT Optimization Profile batch size)")

    parser.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")

    parser.add_argument("--num_calibration_samples", type=int, default=10000,
                        help="Количество образцов для калибровки int8 квантизации")

    parser.add_argument("--export_onnx", action="store_true",
                        help="Экспортировать модель в формат ONNX")

    parser.add_argument("--convert_tensorrt", action="store_true",
                        help="Конвертировать модель в TensorRT")

    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Точность для TensorRT (fp32, fp16, int8)")

    parser.add_argument("--benchmark", action="store_true",
                        help="Выполнить бенчмарк после оптимизации")

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def main():
    args = parse_args()

    logger.info("Старт оптимизации модели")

    optimizer = DonutOptimizer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    calibration_data = None
    if args.data_dir and args.precision == "int8":
        logger.info("Создание калибровочных данных...")

        calibration_data = optimizer.create_calibration_data(
            data_dir=args.data_dir,
            num_samples=args.num_calibration_samples
        )
        if calibration_data is None:
            logger.error("Не удалось создать калибровочные данные. Отменяем INT8 конвертацию.")
            return

    if args.export_onnx:
        logger.info("Запуск экспорта в ONNX...")
        success = optimizer.convert_to_onnx()
        if not success:
            logger.error("Ошибка при экспорте в ONNX. Останавливаем процесс оптимизации.")
            return

    if args.convert_tensorrt:
        logger.info(f"Запуск прямой конвертации в TensorRT с точностью {args.precision}...")

        fp16_first = (args.precision == "int8")

        success = optimizer.convert_to_tensorrt_direct(
            calibration_data=calibration_data,
            precision=args.precision,
            fp16_first=fp16_first
        )
        if not success:
            logger.error("Ошибка при конвертации в TensorRT. Останавливаем процесс оптимизации.")
            return

    if args.benchmark:
        logger.info("Запуск бенчмарка...")
        optimizer.benchmark_trt_vs_original(precision=args.precision)

    logger.info("Оптимизация завершена.")


if __name__ == "__main__":
    main()