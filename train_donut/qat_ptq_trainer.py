import os
import sys
import logging
import time
from pathlib import Path
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_tensorrt
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import export_torch_mode
# from modelopt.torch.quantization.nn import QuantLinear

from model import DonutModel
from dataset import DonutDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DonutTRTConverter:

    class _VEDTracingWrapper(nn.Module):
        def __init__(self, model_ved):
            super().__init__()
            self.model_ved = model_ved
        def forward(self, pixel_values, decoder_input_ids):
            outputs = self.model_ved(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
            return outputs.logits

    def __init__(
        self,
        model: DonutModel,
        data_module: DonutDataModule,
        output_dir: Union[str, Path],
        batch_size: int = 1,
        precision: str = "fp32",
        calibration_batches: int = 32,
        device: str = "cuda",
        calibration_cache_file: Optional[str] = None,
        calibration_algo: str = "minmax",
    ):

        if precision in ["int8", "fp8"] and not MODELOPT_AVAILABLE:
            pass

        self.model = model
        self.data_module = data_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.precision = precision
        self.calibration_batches = calibration_batches
        self.device = device
        self.calibration_cache_file = calibration_cache_file
        self.calibration_algo = calibration_algo

        if not torch.cuda.is_available():
            logger.warning("CUDA недоступна. TensorRT требует GPU.")
            self.device = "cpu"

        logger.info(f"Инициализирован конвертер TensorRT с целевой точностью {precision}")

    def _prepare_calibration_dataloader(self) -> DataLoader:
        if hasattr(self.data_module, 'train_dataloader'):
            dataloader = self.data_module.train_dataloader()
            if dataloader is None and hasattr(self.data_module, 'val_dataloader'):
                logger.info("Train dataloader не найден, используется val dataloader для калибровки.")
                dataloader = self.data_module.val_dataloader()
        elif hasattr(self.data_module, 'val_dataloader'):
            logger.info("Train dataloader не найден, используется val dataloader для калибровки.")
            dataloader = self.data_module.val_dataloader()
        else:
            raise ValueError("data_module должен иметь метод train_dataloader или val_dataloader")

        if dataloader is None:
            raise ValueError("Не удалось получить dataloader для калибровки.")
        return dataloader

    def _prepare_dummy_input(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dummy_pixel_values = None
        try:
            dataloader = self._prepare_calibration_dataloader()
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
                dummy_pixel_values = batch[0][:self.batch_size].to(self.device)
            else:
                logger.warning("Не удалось получить образец pixel_values из dataloader")
        except Exception as e:
            logger.warning(f"Ошибка при получении образца pixel_values из dataloader: {e}")

        if dummy_pixel_values is None:
            image_h, image_w = self.model.model.config.encoder.image_size if hasattr(self.model.model.config, 'encoder') and hasattr(self.model.model.config.encoder, 'image_size') else (384,384)
            dummy_pixel_values = torch.randn(
                self.batch_size, 3, image_h, image_w,
                device=self.device
            )
            logger.info(f"Создан демонстрационный pixel_values тензор размером {dummy_pixel_values.shape}")

        decoder_start_token_id = self.model.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.model.processor.tokenizer.bos_token_id if self.model.processor.tokenizer.bos_token_id is not None else 0

        max_sequence_length = self.model.max_length if hasattr(self.model, 'max_length') else self.max_length
        logger.info(f"Максимальная длина последовательности для компиляции модели: {max_sequence_length}")
    
        dummy_decoder_input_ids = torch.full(
            (dummy_pixel_values.size(0), max_sequence_length),
            self.model.processor.tokenizer.pad_token_id if hasattr(self.model.processor.tokenizer, 'pad_token_id') else 0,
            dtype=torch.long,
            device=self.device
        )
        
        dummy_decoder_input_ids[:, 0] = decoder_start_token_id
        
        logger.info(f"Создан демонстрационный decoder_input_ids тензор размером {dummy_decoder_input_ids.shape}")

        return dummy_pixel_values, dummy_decoder_input_ids

    def calibrate_loop(self, model_nn_module: nn.Module):
        dataloader = self._prepare_calibration_dataloader()
        logger.info(f"Начало калибровки на {self.calibration_batches} батчах для {type(model_nn_module).__name__}")

        model_nn_module.eval()
        count = 0

        current_decoder_start_token_id = getattr(model_nn_module.config, 'decoder_start_token_id', None)
        if current_decoder_start_token_id is None:
             current_decoder_start_token_id = self.model.processor.tokenizer.bos_token_id if self.model.processor.tokenizer.bos_token_id is not None else 0

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                if i >= self.calibration_batches:
                    break

                pixel_values = batch_data[0].to(self.device)
                current_batch_size = pixel_values.size(0)
                dummy_decoder_input_ids = torch.full(
                    (current_batch_size, 1),
                    current_decoder_start_token_id,
                    dtype=torch.long,
                    device=self.device
                )

                try:
                    _ = model_nn_module(pixel_values=pixel_values, decoder_input_ids=dummy_decoder_input_ids)
                    count += 1
                except Exception as e:
                    logger.warning(f"Ошибка при калибровке пакета {i} ({type(model_nn_module).__name__}): {e}")

        logger.info(f"Калибровка ({type(model_nn_module).__name__}) завершена на {count} пакетах")

    def quantize_model(self) -> nn.Module:
        logger.info(f"TRTConverter: Квантизация полной модели VisionEncoderDecoderModel с целевой точностью {self.precision}")

        if self.precision not in ["int8", "fp8"]:
            raise ValueError(f"Точность {self.precision} не поддерживается для modelopt квантизации в этом методе.")
        if not MODELOPT_AVAILABLE:
             raise ImportError("modelopt не установлен.")

        quant_cfg_map = { "int8": mtq.INT8_DEFAULT_CFG, "fp8": mtq.FP8_DEFAULT_CFG }
        quant_cfg = quant_cfg_map[self.precision]
        model_to_quantize = self.model.model.to(self.device)

        if any(isinstance(m, mtq.QuantLinear) for m in model_to_quantize.modules()):
            logger.warning("Модель, похоже, уже содержит узлы квантизации ModelOpt. Повторная квантизация может быть не нужна или вызвать ошибку.")

        mtq.quantize(model_to_quantize, quant_cfg, forward_loop=lambda m: self.calibrate_loop(m))

        logger.info(f"Полная модель успешно квантизирована (TRTConverter) с точностью {self.precision}")
        return model_to_quantize.eval()

    def export_to_torchscript(self, model_to_export_nn_module: Optional[nn.Module] = None, output_path: Optional[str] = None) -> str:
        if output_path is None:
            suffix = "_exported"
            current_model_precision = self.precision
            if model_to_export_nn_module is not None and hasattr(model_to_export_nn_module, 'config'):
                 pass
            elif self.model.precision in ["int8", "fp8"]:
                 current_model_precision = self.model.precision

            if current_model_precision in ["int8", "fp8"]:
                 suffix = f"_quantized_exported_{current_model_precision}.pt"
            else:
                 suffix = f"_exported_{current_model_precision}.pt"
            output_path = str(self.output_dir / f"donut_model{suffix}")

        if model_to_export_nn_module is None:
            if self.model.precision in ["int8", "fp8"]:
                logger.info(f"Модель, предоставленная TRTConverter, уже помечена как {self.model.precision}. Используется напрямую для экспорта TorchScript.")
                model_to_export_nn_module = self.model.model
            elif self.precision in ["int8", "fp8"] and MODELOPT_AVAILABLE:
                logger.info(f"Целевая точность TRT {self.precision}. Квантизация модели для экспорта TorchScript.")
                model_to_export_nn_module = self.quantize_model()
            else:
                model_to_export_nn_module = self.model.model

        model_to_export_nn_module = model_to_export_nn_module.to(self.device).eval()
        dummy_pixel_values, dummy_decoder_input_ids = self._prepare_dummy_input()

        traceable_entity = model_to_export_nn_module
        if hasattr(model_to_export_nn_module, 'config') and \
           hasattr(model_to_export_nn_module.config, 'encoder') and \
           hasattr(model_to_export_nn_module.config, 'decoder'):
            logger.info("VisionEncoderDecoderModel обнаружен. Обертывание для надежной трассировки JIT в export_to_torchscript.")
            traceable_entity = DonutTRTConverter._VEDTracingWrapper(model_to_export_nn_module)
            traceable_entity.eval()

        try:
            with torch.no_grad():

                with export_torch_mode():
                    traced_model = torch.jit.trace(traceable_entity, (dummy_pixel_values, dummy_decoder_input_ids), strict=False)

                torch.jit.save(traced_model, output_path)
                logger.info(f"Модель успешно экспортирована в TorchScript: {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Ошибка при экспорте модели в TorchScript: {e}")
            logger.exception("Детали ошибки экспорта JIT в TRTConverter:")
            raise

    def compile_with_tensorrt(
        self,
        torchscript_model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        workspace_size: int = 1 << 30
    ) -> str:
        if output_path is None:
            output_path = str(self.output_dir / f"donut_model_{self.precision}_trt.pt")

        logger.info(f"Компиляция модели в TensorRT с точностью {self.precision}: {output_path}")

        model_for_trt_nn_module: nn.Module
        if torchscript_model_path:
            logger.info(f"Загрузка TorchScript модели из: {torchscript_model_path}")
            loaded_jit_model = torch.jit.load(torchscript_model_path).to(self.device).eval()
            if isinstance(loaded_jit_model, DonutTRTConverter._VEDTracingWrapper):
                logger.info("Загружена модель-обертка TorchScript, используется внутренняя модель для TRT.")
                model_for_trt_nn_module = loaded_jit_model.model_ved
            else:
                model_for_trt_nn_module = loaded_jit_model

        elif self.model.precision in ["int8", "fp8"]:
            logger.info(f"Модель, предоставленная TRTConverter, уже {self.model.precision}. Используется напрямую.")
            model_for_trt_nn_module = self.model.model
        elif self.precision in ["int8", "fp8"]:
            if not MODELOPT_AVAILABLE:
                raise ImportError(f"modelopt не установлен, не удается выполнить {self.precision} квантизацию для TRT.")
            logger.info(f"Квантизация модели до {self.precision} перед компиляцией TRT.")
            model_for_trt_nn_module = self.quantize_model()
        else:
            model_for_trt_nn_module = self.model.model.to(self.device).eval()

        dummy_pixel_values, dummy_decoder_input_ids = self._prepare_dummy_input()

        try:
            with torch.no_grad():

                with export_torch_mode():
                    exp_program = torch.export.export(model_for_trt_nn_module, (dummy_pixel_values, dummy_decoder_input_ids), strict=False)

                precision_map = {
                    "fp32": {torch.float32}, "fp16": {torch.float16},
                    "int8": {torch.int8}, "fp8": {torch.float8_e4m3fn}
                }
                enabled_precisions = precision_map.get(self.precision)
                if enabled_precisions is None:
                    raise ValueError(f"Неподдерживаемая точность для TRT: {self.precision}")

                max_sequence_length = self.model.max_length if hasattr(self.model, 'max_length') else self.max_length
                
                input_shapes = {
                    "pixel_values": {
                        "min": [dummy_pixel_values.shape[0], dummy_pixel_values.shape[1], 
                                dummy_pixel_values.shape[2], dummy_pixel_values.shape[3]],
                        "opt": [dummy_pixel_values.shape[0], dummy_pixel_values.shape[1], 
                                dummy_pixel_values.shape[2], dummy_pixel_values.shape[3]],
                        "max": [dummy_pixel_values.shape[0], dummy_pixel_values.shape[1], 
                                dummy_pixel_values.shape[2], dummy_pixel_values.shape[3]]
                    },
                    "decoder_input_ids": {
                        "min": [dummy_decoder_input_ids.shape[0], 1],  # Минимум один токен
                        "opt": [dummy_decoder_input_ids.shape[0], max_sequence_length // 2],  # Оптимальный размер
                        "max": [dummy_decoder_input_ids.shape[0], max_sequence_length]  # Максимальная длина
                    }
                }
                
                logger.info(f"Настройка динамических форм для TensorRT: {input_shapes}")

                trt_model_compiled = torch_tensorrt.dynamo.compile(
                    exp_program,
                    inputs=[dummy_pixel_values, dummy_decoder_input_ids],
                    enabled_precisions=enabled_precisions,
                    min_block_size=1,
                    workspace_size=workspace_size,
                    debug=False,
                    input_shapes=input_shapes
                )

                torch.save(trt_model_compiled, output_path)
                logger.info(f"Модель TensorRT успешно скомпилирована и сохранена в {output_path}")
                return output_path
        except Exception as e:
            logger.error(f"Ошибка при компиляции модели TensorRT: {e}")
            logger.exception("Детали ошибки компиляции TRT:")
            raise

    def benchmark_model(
        self, model_path: str, num_warmup: int = 10, num_iter: int = 100
    ) -> Dict[str, Any]:
        logger.info(f"Измерение производительности модели: {model_path}")
        try:
            # Assuming the model was saved using torch.save
            model_trt = torch.load(model_path, weights_only=False).to(self.device).eval()
            dummy_pixel_values, dummy_decoder_input_ids = self._prepare_dummy_input()

            benchmark_batch_size = dummy_pixel_values.shape[0]
            if benchmark_batch_size != self.batch_size:
                 logger.warning(f"Размер пакета для бенчмарка ({benchmark_batch_size}) отличается от self.batch_size ({self.batch_size}). Используется {benchmark_batch_size}.")

            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model_trt(dummy_pixel_values, dummy_decoder_input_ids)

            if "cuda" in self.device: torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iter):
                    _ = model_trt(dummy_pixel_values, dummy_decoder_input_ids)
                    if "cuda" in self.device: torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / num_iter

            metrics = {
                "total_time_ms": total_time * 1000, "avg_time_ms": avg_time * 1000,
                "avg_time_per_sample_ms": avg_time * 1000 / benchmark_batch_size if benchmark_batch_size > 0 else 0,
                "samples_per_sec": benchmark_batch_size / avg_time if avg_time > 0 else 0,
                "batches_per_sec": 1 / avg_time if avg_time > 0 else 0,
                "num_iter": num_iter, "batch_size": benchmark_batch_size
            }
            logger.info(f"Средняя длительность на пакет: {metrics['avg_time_ms']:.2f} мс. Образцов в секунду: {metrics['samples_per_sec']:.2f}")
            return metrics
        except Exception as e:
            logger.error(f"Ошибка при измерении производительности: {e}")
            raise

class DonutQuantizer:

    class _QuantTracingWrapper(nn.Module):
        def __init__(self, model_ved):
            super().__init__()
            self.model_ved = model_ved
        def forward(self, pixel_values, decoder_input_ids):
            outputs = self.model_ved(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
            return outputs.logits

    def __init__(
        self, model: DonutModel, data_module: DonutDataModule, output_dir: Union[str, Path],
        quantization_type: str = "ptq", learning_rate: float = 1e-5, weight_decay: float = 0.0,
        num_epochs: int = 2, calibration_batches: int = 32, device: str = "cuda", precision: str = "int8",
    ):
        if not MODELOPT_AVAILABLE:
            raise ImportError("modelopt не установлен. Установите его для квантизации.")

        self.model = model
        self.data_module = data_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.quantization_type = quantization_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.calibration_batches = calibration_batches
        self.device = device
        self.precision = precision

        if not torch.cuda.is_available():
            logger.warning("CUDA недоступна. Квантизация может быть медленной на CPU.")
            self.device = "cpu"
        logger.info(f"Инициализирован квантизатор Donut с типом {quantization_type} и точностью {precision}")

    def _prepare_dummy_input_for_quantizer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        image_h, image_w = self.model.model.config.encoder.image_size if hasattr(self.model.model.config, 'encoder') and hasattr(self.model.model.config.encoder, 'image_size') else (384,384)
        current_batch_size = 1
        dummy_pixel_values = torch.randn(current_batch_size, 3, image_h, image_w, device=self.device)

        decoder_start_token_id = self.model.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
             decoder_start_token_id = self.model.processor.tokenizer.bos_token_id if self.model.processor.tokenizer.bos_token_id is not None else 0
        dummy_decoder_input_ids = torch.full((current_batch_size, 1), decoder_start_token_id, dtype=torch.long, device=self.device)
        return dummy_pixel_values, dummy_decoder_input_ids

    def calibrate_loop(self, model_nn_module: nn.Module):
        dataloader = self.data_module.train_dataloader()
        if dataloader is None: dataloader = self.data_module.val_dataloader()
        if dataloader is None: raise ValueError("Не удалось получить dataloader для калибровки в DonutQuantizer.")

        logger.info(f"Начало калибровки (DonutQuantizer) на {self.calibration_batches} батчах для {type(model_nn_module).__name__}")
        model_nn_module.eval()
        count = 0

        current_decoder_start_token_id = getattr(model_nn_module.config, 'decoder_start_token_id', None)
        if current_decoder_start_token_id is None:
             current_decoder_start_token_id = self.model.processor.tokenizer.bos_token_id if self.model.processor.tokenizer.bos_token_id is not None else 0

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                if i >= self.calibration_batches: break
                pixel_values = batch_data[0].to(self.device)
                current_batch_size = pixel_values.size(0)
                dummy_decoder_input_ids = torch.full((current_batch_size, 1), current_decoder_start_token_id, dtype=torch.long, device=self.device)
                try:
                    _ = model_nn_module(pixel_values=pixel_values, decoder_input_ids=dummy_decoder_input_ids)
                    count += 1
                except Exception as e:
                    logger.warning(f"Ошибка при калибровке пакета {i} (DonutQuantizer - {type(model_nn_module).__name__}): {e}")
        logger.info(f"Калибровка (DonutQuantizer - {type(model_nn_module).__name__}) завершена на {count} пакетах")

    def ptq_quantize(self) -> nn.Module:
        logger.info("Начало процесса PTQ квантизации для полной модели")
        quant_cfg_map = {"int8": mtq.INT8_DEFAULT_CFG, "fp8": mtq.FP8_DEFAULT_CFG}
        if self.precision not in quant_cfg_map: raise ValueError(f"Неподдерживаемая точность для PTQ: {self.precision}")
        quant_cfg = quant_cfg_map[self.precision]

        model_to_quantize = self.model.model.to(self.device)
        mtq.quantize(model_to_quantize, quant_cfg, forward_loop=lambda m: self.calibrate_loop(m))
        model_to_quantize.eval()

        q_model_path_dir = self.output_dir / f"donut_ptq_model_{self.precision}"
        q_model_path_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_quantize.state_dict(), q_model_path_dir / "model_state_dict.pt")

        dummy_pixel_values, dummy_decoder_input_ids = self._prepare_dummy_input_for_quantizer()
        traceable_model_wrapper = DonutQuantizer._QuantTracingWrapper(model_to_quantize)
        traceable_model_wrapper.eval()

        with torch.no_grad(), export_torch_mode():
            try:
                traced_model = torch.jit.trace(traceable_model_wrapper, (dummy_pixel_values, dummy_decoder_input_ids), strict=False)
                torch.jit.save(traced_model, q_model_path_dir / "model_quantized.jit.pt")
                logger.info(f"Квантизированная PTQ модель (TorchScript) сохранена в {q_model_path_dir}")
            except Exception as e:
                logger.warning(f"Не удалось экспортировать PTQ модель в TorchScript: {e}")
                logger.exception("Детали ошибки экспорта JIT в PTQ:")
        return model_to_quantize

    def qat_quantize(self) -> nn.Module:
        logger.info("Начало процесса QAT квантизации для полной модели")
        quant_cfg_map = {"int8": mtq.INT8_DEFAULT_CFG, "fp8": mtq.FP8_DEFAULT_CFG}
        if self.precision not in quant_cfg_map: raise ValueError(f"Неподдерживаемая точность для QAT: {self.precision}")
        quant_cfg = quant_cfg_map[self.precision]

        model_to_quantize = self.model.model.to(self.device)
        mtq.quantize(model_to_quantize, quant_cfg, forward_loop=lambda m: self.calibrate_loop(m))

        optimizer = torch.optim.AdamW(model_to_quantize.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        train_dataloader = self.data_module.train_dataloader()
        if train_dataloader is None: raise ValueError("Не удалось получить train_dataloader для QAT.")

        for epoch in range(self.num_epochs):
            model_to_quantize.train()
            logger.info(f"QAT Эпоха {epoch+1}/{self.num_epochs}")
            running_loss = 0.0
            for i, batch in enumerate(tqdm(train_dataloader, desc=f"QAT Эпоха {epoch+1}")):
                pixel_values, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                outputs = model_to_quantize(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Обнаружен NaN/Inf loss в QAT на шаге {i}, эпоха {epoch+1}. Пропуск шага.")
                    optimizer.zero_grad()
                    continue
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 10 == 0:
                    logger.info(f"[{epoch+1}, {i+1}] QAT потери: {running_loss/10:.3f}")
                    running_loss = 0.0

        model_to_quantize.eval()
        q_model_path_dir = self.output_dir / f"donut_qat_model_{self.precision}"
        q_model_path_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_to_quantize.state_dict(), q_model_path_dir / "model_state_dict.pt")

        dummy_pixel_values, dummy_decoder_input_ids = self._prepare_dummy_input_for_quantizer()
        traceable_model_wrapper = DonutQuantizer._QuantTracingWrapper(model_to_quantize)
        traceable_model_wrapper.eval()
        with torch.no_grad(), export_torch_mode():
            try:
                traced_model = torch.jit.trace(traceable_model_wrapper, (dummy_pixel_values, dummy_decoder_input_ids), strict=False)
                torch.jit.save(traced_model, q_model_path_dir / "model_quantized.jit.pt")
                logger.info(f"Квантизированная QAT модель (TorchScript) сохранена в {q_model_path_dir}")
            except Exception as e:
                logger.warning(f"Не удалось экспортировать QAT модель в TorchScript: {e}")
                logger.exception("Детали ошибки экспорта JIT в QAT:")
        return model_to_quantize

    def quantize(self) -> DonutModel:
        quantized_nn_module: nn.Module
        if self.quantization_type.lower() == "ptq":
            quantized_nn_module = self.ptq_quantize()
        elif self.quantization_type.lower() == "qat":
            quantized_nn_module = self.qat_quantize()
        else:
            raise ValueError(f"Неизвестный тип квантизации: {self.quantization_type}.")

        quantized_donut_model = DonutModel(
            model=quantized_nn_module, processor=self.model.processor, device=self.device,
            precision=self.precision, max_length=self.model.max_length,
            task_start_token=self.model.task_start_token, prompt_end_token=self.model.prompt_end_token
        )
        quantized_donut_model.model.to(self.device)
        return quantized_donut_model

def optimize_donut_model(
    model_path: str, data_dir: str, output_dir: str, optimization_type: str = "ptq",
    batch_size: int = 1, num_epochs: int = 2, num_workers: int = 8, max_length: int = 64,
    learning_rate: float = 1e-5, calibration_batches: int = 32, device: str = "cuda",
    image_size: Tuple[int, int] = (384, 384), task_start_token: str = "<s_500k>",
    prompt_end_token: Optional[str] = "<s_prompt>", sort_json_key: bool = False, apply_augmentation: bool = True,
    distributed: bool = False, pin_memory: bool = False, cache_images: bool = False,
    train_limit_samples: Optional[int] = None, val_limit_samples: Optional[int] = None,
    augmentation_prob: float = 0.3, max_rotation: float = 8.0, brightness_min: float = 0.8,
    brightness_max: float = 1.2, contrast_min: float = 0.8, contrast_max: float = 1.2,
    blur_min: int = 0, blur_max: int = 2, noise_level: float = 0.05,
    sharpness_min: float = 0.8, sharpness_max: float = 1.5
) -> Dict[str, Any]:

    results = {
        "model_path": model_path, "optimization_type": optimization_type, "output_dir": output_dir,
        "optimized_model_path": None, "benchmark_results": None
    }
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Загрузка модели из {model_path}")
    original_donut_model = DonutModel.from_pretrained(
        model_path, device=device, image_size=image_size, max_length=max_length,
        task_start_token=task_start_token, prompt_end_token=prompt_end_token
    )
    original_donut_model.model.to(device)

    logger.info(f"Загрузка данных из {data_dir}")
    data_module = DonutDataModule(
        processor=original_donut_model.processor, data_dir=data_dir, batch_size=batch_size,
        num_workers=num_workers, max_length=max_length, task_start_token=task_start_token,
        prompt_end_token=prompt_end_token, sort_json_key=sort_json_key, image_size=image_size,
        apply_augmentation=apply_augmentation, distributed=distributed, pin_memory=pin_memory,
        cache_images=cache_images, train_limit_samples=train_limit_samples, val_limit_samples=val_limit_samples,
        augmentation_prob=augmentation_prob, max_rotation=max_rotation,
        brightness_range=(brightness_min, brightness_max), contrast_range=(contrast_min, contrast_max),
        blur_range=(blur_min, blur_max), noise_level=noise_level, sharpness_range=(sharpness_min, sharpness_max)
    )

    if optimization_type.startswith("trt_"):
        trt_precision = optimization_type.split("_")[1]
        converter = DonutTRTConverter(
            model=original_donut_model, data_module=data_module, output_dir=output_dir_path,
            batch_size=batch_size, precision=trt_precision, calibration_batches=calibration_batches,
            device=device, calibration_cache_file=str(output_dir_path / f"calibration_{trt_precision}.cache")
        )
        trt_path = converter.compile_with_tensorrt()
        benchmark_results = converter.benchmark_model(trt_path)
        results["optimized_model_path"] = trt_path
        results["benchmark_results"] = benchmark_results

    elif optimization_type in ["ptq", "qat"]:
        quantizer_precision = "fp8" if "fp8" in optimization_type else "int8"
        quantizer = DonutQuantizer(
            model=original_donut_model, data_module=data_module, output_dir=output_dir_path,
            quantization_type=optimization_type, learning_rate=learning_rate, num_epochs=num_epochs,
            calibration_batches=calibration_batches, device=device, precision=quantizer_precision
        )
        quantized_donut_model = quantizer.quantize()
        quantized_model_save_dir = output_dir_path / f"donut_{optimization_type}_{quantizer_precision}_modelopt"
        quantized_donut_model.save_pretrained(quantized_model_save_dir)
        logger.info(f"{optimization_type.upper()} модель (ModelOpt) сохранена в: {quantized_model_save_dir}")
        results["optimized_model_path"] = str(quantized_model_save_dir)

        logger.info(f"Конвертация квантизированной ({optimization_type.upper()}) модели в TensorRT")
        trt_converter_precision = quantized_donut_model.precision
        converter = DonutTRTConverter(
            model=quantized_donut_model, data_module=data_module, output_dir=output_dir_path,
            batch_size=batch_size, precision=trt_converter_precision,
            calibration_batches=calibration_batches, device=device,
            calibration_cache_file=str(output_dir_path / f"calibration_quantized_{trt_converter_precision}.cache")
        )
        # export_to_torchscript is called before compile_with_tensorrt
        torchscript_path = converter.export_to_torchscript(model_to_export_nn_module=quantized_donut_model.model)
        trt_path = converter.compile_with_tensorrt(torchscript_model_path=torchscript_path)
        benchmark_results = converter.benchmark_model(trt_path)
        results["optimized_model_path"] = trt_path
        results["benchmark_results"] = benchmark_results
    else:
        raise ValueError(f"Неизвестный тип оптимизации: {optimization_type}.")

    with open(output_dir_path / "optimization_results.json", "w", encoding="utf-8") as f:
        json_results = {k: str(v) if isinstance(v, Path) else v for k, v in results.items()}
        json.dump(json_results, f, indent=2)
    logger.info(f"Оптимизация завершена. Результаты сохранены в {output_dir_path / 'optimization_results.json'}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Оптимизация модели Donut с использованием TensorRT и квантизации")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Путь к предварительно обученной модели Donut (директория)")
    parser.add_argument("--data_dir", type=str, required=True, help="Директория с данными")
    parser.add_argument("--output_dir", type=str, required=True, help="Директория для сохранения результатов")
    parser.add_argument("--optimization_type", type=str, default="ptq",
                        choices=["trt_fp32", "trt_fp16", "trt_int8", "trt_fp8", "ptq", "qat"], help="Тип оптимизации")
    parser.add_argument("--batch_size", type=int, default=1, help="Размер пакета")
    parser.add_argument("--device", type=str, default="cuda", help="Устройство (cuda или cpu)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 384], help="Размер изображения [высота, ширина]")
    parser.add_argument("--max_length", type=int, default=64, help="Максимальная длина последовательности")
    parser.add_argument("--task_start_token", type=str, default="<s_500k>", help="Токен начала задачи")
    parser.add_argument("--prompt_end_token", type=str, default="<s_prompt>", help="Токен конца промпта")
    parser.add_argument("--num_workers", type=int, default=8, help="Количество рабочих процессов DataLoader")
    parser.add_argument("--calibration_batches", type=int, default=32, help="Количество пакетов для калибровки")
    parser.add_argument("--qat_num_epochs", type=int, default=2, help="Количество эпох для QAT")
    parser.add_argument("--qat_learning_rate", type=float, default=1e-5, help="Скорость обучения для QAT")
    parser.add_argument("--train_limit_samples", type=int, default=None, help="Ограничение обучающих выборок")
    parser.add_argument("--val_limit_samples", type=int, default=None, help="Ограничение валидационных выборок")
    args = parser.parse_args()

    optimize_donut_model(
        model_path=args.pretrained_model_path, data_dir=args.data_dir, output_dir=args.output_dir,
        optimization_type=args.optimization_type, batch_size=args.batch_size, num_epochs=args.qat_num_epochs,
        num_workers=args.num_workers, max_length=args.max_length, learning_rate=args.qat_learning_rate,
        calibration_batches=args.calibration_batches, device=args.device, image_size=tuple(args.image_size),
        task_start_token=args.task_start_token, prompt_end_token=args.prompt_end_token,
        train_limit_samples=args.train_limit_samples, val_limit_samples=args.val_limit_samples
    )