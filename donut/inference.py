import os
import sys
import json
import logging
import argparse
from pathlib import Path
import time
import re
from typing import Dict, List, Optional, Tuple, Union, Any
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from tqdm.auto import tqdm

from model import DonutModel
from dataset import DonutDataModule
from utils import MetricsCalculator, MemoryOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DonutInference:
    def __init__(
        self,
        model: DonutModel,
        output_dir: Union[str, Path],
        num_beams: int = 5,
        max_length: int = 768,
        device: Optional[Union[str, torch.device]] = None,
        save_visualizations: bool = True,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (1280, 960)
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_beams = num_beams
        self.max_length = max_length
        self.device = device or model.device
        self.save_visualizations = save_visualizations
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.metrics_calculator = MetricsCalculator()
        
        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        
        logger.info(f"Инициализирован модуль вывода Donut, устройство: {self.device}")
    
    def predict_batch(self, pixel_values: torch.Tensor) -> List[str]:
        """Генерирует предсказания для пакета изображений."""
        with torch.no_grad():
            pred_tokens = self.model.generate(
                pixel_values=pixel_values.to(self.device),
                num_beams=self.num_beams,
                max_length=self.max_length
            )
            
        return pred_tokens
    
    def process_sample(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> Dict[str, Any]:
        """Обрабатывает один образец и возвращает результаты."""
        try:
            image = Image.open(image_path).convert("RGB")
            
            pixel_values = self.model.processor(
                image, 
                return_tensors="pt"
            ).pixel_values
 
            start_time = time.time()
            
            with torch.no_grad():
                if prompt:
                    predictions = self.model.generate(
                        pixel_values=pixel_values.to(self.device),
                        prompt=prompt,
                        num_beams=self.num_beams,
                        max_length=self.max_length,
                        return_json=True
                    )
                else:
                    predictions = self.model.generate(
                        pixel_values=pixel_values.to(self.device),
                        num_beams=self.num_beams,
                        max_length=self.max_length,
                        return_json=True
                    )
            
            inference_time = time.time() - start_time
            
            result = {
                "image_path": str(image_path),
                "predictions": predictions[0] if isinstance(predictions, list) else predictions,
                "inference_time": inference_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {image_path}: {e}")
            return {"image_path": str(image_path), "error": str(e)}
    
    def evaluate_dataset(
        self, 
        data_module: DonutDataModule,
        split: str = "test",
        num_samples: Optional[int] = None,
        save_individual_results: bool = True
    ) -> Dict[str, Any]:
        """Оценивает модель на датасете и возвращает метрики."""
        
        if split == "test":
            dataloader = data_module.test_dataloader()
        elif split == "val" or split == "valid":
            dataloader = data_module.val_dataloader()
        else:
            dataloader = data_module.train_dataloader()
        
        if dataloader is None:
            raise ValueError(f"Загрузчик данных для разделения '{split}' не найден")

        self.model.eval()
        
        all_metrics = {
            "cer": [],
            "wer": [],
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "inference_times": []
        }
        
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        batch_count = 0
        sample_count = 0
        
        max_batches = None if num_samples is None else (num_samples // self.batch_size) + 1

        for i, batch in enumerate(tqdm(dataloader, desc=f"Оценка ({split})", total=max_batches)):
            if max_batches is not None and i >= max_batches:
                break
            
            try:
                if len(batch) == 4:
                    pixel_values, labels, prompt_end_indices, target_sequences = batch
                else:
                    pixel_values, labels = batch
                    target_sequences = [""] * len(pixel_values)
                
                batch_size = pixel_values.shape[0]
                
                batch_start_time = time.time()
                
                pred_tokens = self.predict_batch(pixel_values)

                print(f"pred_tokens: {pred_tokens}")
                print(f"target_sequences: {target_sequences}")
                
                batch_inference_time = time.time() - batch_start_time
                all_metrics["inference_times"].extend([batch_inference_time / batch_size] * batch_size)
 
                for j, (pred, target) in enumerate(zip(pred_tokens, target_sequences)):
                    if sample_count >= num_samples and num_samples is not None:
                        break
                    
                    pred_clean = re.sub(r"<[^>]*>", "", pred).strip()
                    target_clean = re.sub(r"<[^>]*>", "", target).strip()
                    
                    all_predictions.append(pred_clean)
                    all_targets.append(target_clean)
                    
                    sample_metrics = {
                        "cer": self.metrics_calculator.calculate_cer(pred_clean, target_clean),
                        "wer": self.metrics_calculator.calculate_wer(pred_clean, target_clean)
                    }
                    
                    sample_rouge = self.metrics_calculator.calculate_rouge(pred_clean, target_clean)
                    sample_metrics.update(sample_rouge)
                    
                    for metric_name, value in sample_metrics.items():
                        all_metrics[metric_name].append(value)
                    
                    if save_individual_results and j < 5:
                        self._save_sample_visualization(
                            pixel_values[j], 
                            pred_clean, 
                            target_clean, 
                            sample_metrics,
                            sample_count
                        )
                    
                    sample_count += 1
                
                batch_count += 1
                
            except Exception as e:
                logger.error(f"Ошибка при обработке пакета {i}: {e}")
                continue

        avg_metrics = {
            metric: np.mean(values) if values else 0.0
            for metric, values in all_metrics.items()
        }
        
        avg_metrics["num_samples"] = sample_count
        avg_metrics["total_time"] = time.time() - start_time
        avg_metrics["avg_time_per_sample"] = avg_metrics["total_time"] / max(1, sample_count)
        
        metrics_file = self.output_dir / f"{split}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(avg_metrics, f, indent=2, ensure_ascii=False)
        
        self._save_metrics_visualization(all_metrics, split)
        
        self._save_predictions_samples(all_predictions, all_targets, split)
        
        return avg_metrics
    
    def _save_sample_visualization(
        self, 
        pixel_values: torch.Tensor, 
        prediction: str, 
        target: str, 
        metrics: Dict[str, float],
        sample_idx: int
    ) -> None:
        """Сохраняет визуализацию для одного образца."""
        if not self.save_visualizations:
            return

        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True, parents=True)

        pixel_values = pixel_values.detach().cpu().numpy()
        pixel_values = np.transpose(pixel_values, (1, 2, 0))

        if pixel_values.max() > 1.0 or pixel_values.min() < 0.0:
            pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        ax1.imshow(pixel_values)
        ax1.axis('off')
        ax1.set_title("Входное изображение", fontsize=14)
   
        table_data = [
            ["Предсказание", prediction[:100] + ("..." if len(prediction) > 100 else "")],
            ["Ожидаемый результат", target[:100] + ("..." if len(target) > 100 else "")],
            ["CER", f"{metrics['cer']:.4f}"],
            ["WER", f"{metrics['wer']:.4f}"],
            ["ROUGE-1", f"{metrics['rouge-1']:.4f}"],
            ["ROUGE-2", f"{metrics['rouge-2']:.4f}"],
            ["ROUGE-L", f"{metrics['rouge-l']:.4f}"]
        ]
        
        ax2.axis('off')
        ax2.set_title("Результаты", fontsize=14)
        
        table = ax2.table(
            cellText=table_data,
            colWidths=[0.2, 0.8],
            loc='center',
            cellLoc='left'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontsize=12, ha='left')
            elif row % 2 == 0:
                cell.set_facecolor('#D9E1F2')
            else:
                cell.set_facecolor('#E9EDF5')
        
        plt.tight_layout()
 
        sample_file = samples_dir / f"sample_{sample_idx:04d}.png"
        plt.savefig(sample_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    def _save_metrics_visualization(self, metrics: Dict[str, List[float]], split: str) -> None:
        """Сохраняет визуализацию метрик."""
        if not self.save_visualizations:
            return
            
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

        sns.set(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax1 = axes[0, 0]
        if metrics["cer"]:
            sns.histplot(metrics["cer"], kde=True, ax=ax1, color="blue", alpha=0.7, label="CER")
        if metrics["wer"]:
            sns.histplot(metrics["wer"], kde=True, ax=ax1, color="red", alpha=0.5, label="WER")
        
        ax1.set_title("Распределение метрик ошибок CER/WER", fontsize=14)
        ax1.set_xlabel("Значение метрики", fontsize=12)
        ax1.set_ylabel("Частота", fontsize=12)
        ax1.legend()
        
        ax2 = axes[0, 1]
        if metrics["rouge-1"]:
            sns.histplot(metrics["rouge-1"], kde=True, ax=ax2, color="green", alpha=0.7, label="ROUGE-1")
        if metrics["rouge-2"]:
            sns.histplot(metrics["rouge-2"], kde=True, ax=ax2, color="orange", alpha=0.5, label="ROUGE-2")
        if metrics["rouge-l"]:
            sns.histplot(metrics["rouge-l"], kde=True, ax=ax2, color="purple", alpha=0.5, label="ROUGE-L")
        
        ax2.set_title("Распределение метрик ROUGE", fontsize=14)
        ax2.set_xlabel("Значение метрики", fontsize=12)
        ax2.set_ylabel("Частота", fontsize=12)
        ax2.legend()

        ax3 = axes[1, 0]
        if metrics["inference_times"]:
            sns.histplot(metrics["inference_times"], kde=True, ax=ax3, color="teal", alpha=0.7)
        
        ax3.set_title("Распределение времени инференса", fontsize=14)
        ax3.set_xlabel("Время (сек)", fontsize=12)
        ax3.set_ylabel("Частота", fontsize=12)

        plt.suptitle(f"Метрики оценки ({split})", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        metrics_file = plots_dir / f"{split}_metrics_visualization.png"
        plt.savefig(metrics_file, dpi=100, bbox_inches='tight')
        plt.close(fig)

        plt.figure(figsize=(10, 8))

        try:
            import pandas as pd
            metrics_df = pd.DataFrame({
                'CER': metrics['cer'],
                'WER': metrics['wer'],
                'ROUGE-1': metrics['rouge-1'],
                'ROUGE-2': metrics['rouge-2'],
                'ROUGE-L': metrics['rouge-l'],
                'Time': metrics['inference_times']
            })
            
            corr_matrix = metrics_df.corr()
            
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5,
                square=True
            )
            
            plt.title(f"Корреляция метрик ({split})", fontsize=16)
            plt.tight_layout()
            
            correlation_file = plots_dir / f"{split}_metrics_correlation.png"
            plt.savefig(correlation_file, dpi=100, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            logger.warning("Pandas не установлен, корреляционная матрица не создана")
    
    def _save_predictions_samples(self, predictions: List[str], targets: List[str], split: str) -> None:
        """Сохраняет примеры предсказаний для анализа."""
        if not self.save_visualizations:
            return

        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True, parents=True)

        max_examples = min(100, len(predictions))
        indices = list(range(len(predictions)))
        
        if len(predictions) > 100:
            random.seed(42)
            indices = random.sample(indices, max_examples)
        
        samples = []
        for i in indices:
            samples.append({
                "prediction": predictions[i],
                "target": targets[i],
                "cer": self.metrics_calculator.calculate_cer(predictions[i], targets[i]),
                "wer": self.metrics_calculator.calculate_wer(predictions[i], targets[i]),
                "rouge": self.metrics_calculator.calculate_rouge(predictions[i], targets[i])
            })
        
        predictions_file = predictions_dir / f"{split}_predictions_samples.json"
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        self._save_predictions_comparison(samples, split)
    
    def _save_predictions_comparison(self, samples: List[Dict[str, Any]], split: str) -> None:
        """Сохраняет сравнительный анализ лучших, худших и средних предсказаний."""
        sorted_by_cer = sorted(samples, key=lambda x: x["cer"])
        
        best_samples = sorted_by_cer[:5]
        worst_samples = sorted_by_cer[-5:]

        middle_idx = len(sorted_by_cer) // 2
        middle_samples = sorted_by_cer[middle_idx-2:middle_idx+3]

        def format_sample(sample, idx):
            return (
                f"Пример #{idx}:\n"
                f"Предсказание: {sample['prediction']}\n"
                f"Ожидаемое:    {sample['target']}\n"
                f"CER: {sample['cer']:.4f}, WER: {sample['wer']:.4f}, ROUGE-2: {sample['rouge']['rouge-2']:.4f}\n"
                f"{'-' * 80}\n"
            )
        
        report = (
            f"Анализ предсказаний ({split})\n"
            f"{'=' * 80}\n\n"
            f"ЛУЧШИЕ ПРЕДСКАЗАНИЯ (по CER)\n"
            f"{'-' * 80}\n"
        )
        
        for i, sample in enumerate(best_samples):
            report += format_sample(sample, i + 1)
        
        report += (
            f"\nСРЕДНИЕ ПРЕДСКАЗАНИЯ\n"
            f"{'-' * 80}\n"
        )
        
        for i, sample in enumerate(middle_samples):
            report += format_sample(sample, i + 1)
        
        report += (
            f"\nХУДШИЕ ПРЕДСКАЗАНИЯ (по CER)\n"
            f"{'-' * 80}\n"
        )
        
        for i, sample in enumerate(worst_samples):
            report += format_sample(sample, i + 1)
        
        report_file = self.output_dir / f"{split}_predictions_analysis.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)


class DonutInferenceTRT:
    """Класс для инференса с TensorRT моделью."""
    
    def __init__(
        self,
        tensorrt_dir: str,
        device: str = "cuda",
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 50
    ):
        self.tensorrt_dir = Path(tensorrt_dir)
        self.model_path = self.tensorrt_dir.parent
        self.device = device
        self.batch_size = batch_size
        self.config_path = self.tensorrt_dir / f"tensorrt_config.json"
        self.temperature = temperature
        self.top_k = top_k
        
        self._resources_initialized = False
        self._cuda_context_valid = True
        self._tensorrt_objects_valid = True
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.base_name = self.config.get('base_name', 'donut')
        self.task_start_token = self.config['task_start_token']
        self.decoder_start_token_id = self.config.get('decoder_start_token_id')
        self.prompt_end_token = self.config.get('prompt_end_token', self.task_start_token)
        self.max_length = self.config['max_length']
        self.image_size = tuple(self.config['image_size'])

        self.encoder_engine_path = self.tensorrt_dir / f"{self.base_name}_encoder.engine"
        self.decoder_engine_path = self.tensorrt_dir / f"{self.base_name}_decoder.engine"
        
        if not self.encoder_engine_path.exists():
            raise FileNotFoundError(f"Encoder engine not found: {self.encoder_engine_path}")
        if not self.decoder_engine_path.exists():
            raise FileNotFoundError(f"Decoder engine not found: {self.decoder_engine_path}")
        
        from transformers import DonutProcessor
        self.processor = DonutProcessor.from_pretrained(self.model_path)
        
        self.encoder_engine = None
        self.decoder_engine = None
        self.encoder_context = None
        self.decoder_context = None

        self.metrics_calculator = MetricsCalculator()
        
        self._init_tensorrt()
        logger.info(f"Инициализирован TensorRT инференс для {tensorrt_dir}")
    
    def _init_tensorrt(self):
        """Инициализация TensorRT для версии 10.9.0.34."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            self.cuda = cuda
            self.cuda_available = True
            self.trt = trt
        except ImportError as e:
            logger.error(f"Не удалось импортировать TensorRT или PyCUDA: {e}")
            raise ImportError("TensorRT или PyCUDA не установлены")
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(self.encoder_engine_path, 'rb') as f:
            encoder_engine_data = f.read()
        self.encoder_engine = runtime.deserialize_cuda_engine(encoder_engine_data)
        self.encoder_context = self.encoder_engine.create_execution_context()
        
        with open(self.decoder_engine_path, 'rb') as f:
            decoder_engine_data = f.read()
        self.decoder_engine = runtime.deserialize_cuda_engine(decoder_engine_data)
        self.decoder_context = self.decoder_engine.create_execution_context()
        
        logger.info(f"TensorRT engines loaded")
        
        self.cuda_stream = cuda.Stream()
        
        self.encoder_inputs = {}
        self.encoder_outputs = {}
        self.encoder_buffers = {}
        self.encoder_gpu_buffers = {}
        
        for i in range(self.encoder_engine.num_io_tensors):
            name = self.encoder_engine.get_tensor_name(i)
            mode = self.encoder_engine.get_tensor_mode(name)
            shape = tuple(self.encoder_engine.get_tensor_shape(name))
            
            logger.info(f"Encoder tensor {i}: {name}, mode={mode}, shape={shape}")
            
            tensor_info = {
                'mode': mode,
                'shape': shape,
                'dtype': np.float32
            }
            
            if mode == trt.TensorIOMode.INPUT:
                self.encoder_inputs[name] = tensor_info
                if 'pixel_values' in name.lower() or i == 0:
                    self.encoder_input_name = name
                    self.encoder_input_shape = shape
                size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
                gpu_buffer = cuda.mem_alloc(size)
                self.encoder_gpu_buffers[name] = gpu_buffer
                self.encoder_buffers[name] = np.empty(shape, dtype=np.float32)
            else:
                self.encoder_outputs[name] = tensor_info
                size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
                gpu_buffer = cuda.mem_alloc(size)
                self.encoder_gpu_buffers[name] = gpu_buffer
                self.encoder_buffers[name] = np.empty(shape, dtype=np.float32)
                
                if 'encoder_outputs' in name.lower() or 'hidden' in name.lower():
                    self.encoder_output_name = name
                    self.encoder_output_shape = shape
        
        self.decoder_inputs = {}
        self.decoder_outputs = {}
        self.decoder_gpu_buffers = {}
        
        for i in range(self.decoder_engine.num_io_tensors):
            name = self.decoder_engine.get_tensor_name(i)
            mode = self.decoder_engine.get_tensor_mode(name)
            shape = tuple(self.decoder_engine.get_tensor_shape(name))
            
            logger.info(f"Decoder tensor {i}: {name}, mode={mode}, shape={shape}")
            
            if mode == trt.TensorIOMode.INPUT:
                tensor_info = {
                    'mode': mode,
                    'shape': shape,
                    'dtype': np.int32 if 'input_ids' in name.lower() else np.float32
                }
                self.decoder_inputs[name] = tensor_info
                
                if 'input_ids' in name.lower():
                    self.decoder_input_ids_name = name
                    self.decoder_input_ids_shape = shape
                elif 'encoder_hidden' in name.lower() or 'hidden' in name.lower():
                    self.decoder_encoder_hidden_name = name
                    self.decoder_encoder_hidden_shape = shape
            else:
                tensor_info = {
                    'mode': mode,
                    'shape': shape,
                    'dtype': np.float32
                }
                self.decoder_outputs[name] = tensor_info
                
                if 'logits' in name.lower() or i == 0:
                    self.decoder_output_name = name
                    self.decoder_output_shape = shape
        
        self._resources_initialized = True
    
    def cleanup(self):
        """Явное освобождение CUDA ресурсов."""
        if not self._resources_initialized:
            return
            
        try:
            if self._cuda_context_valid:
                try:
                    self.cuda.Context.get_current()
                    if hasattr(self, 'encoder_gpu_buffers'):
                        for buffer in self.encoder_gpu_buffers.values():
                            if buffer:
                                buffer.free()
                        self.encoder_gpu_buffers.clear()
                    if hasattr(self, 'decoder_gpu_buffers'):
                        for buffer in self.decoder_gpu_buffers.values():
                            if buffer:
                                buffer.free()
                        self.decoder_gpu_buffers.clear()
                    logger.info("CUDA буферы успешно освобождены")
                except Exception as e:
                    logger.warning(f"Ошибка при освобождении CUDA буферов: {e}")
                    self._cuda_context_valid = False
            if self._tensorrt_objects_valid:
                try:
                    if hasattr(self, 'encoder_context') and self.encoder_context is not None:
                        del self.encoder_context
                        self.encoder_context = None
                    if hasattr(self, 'decoder_context') and self.decoder_context is not None:
                        del self.decoder_context
                        self.decoder_context = None
                    if hasattr(self, 'encoder_engine') and self.encoder_engine is not None:
                        del self.encoder_engine
                        self.encoder_engine = None
                    if hasattr(self, 'decoder_engine') and self.decoder_engine is not None:
                        del self.decoder_engine
                        self.decoder_engine = None
                    logger.info("TensorRT объекты успешно освобождены")
                except Exception as e:
                    logger.warning(f"Ошибка при освобождении TensorRT объектов: {e}")
                self._tensorrt_objects_valid = False
            
            if self._cuda_context_valid:
                try:
                    if hasattr(self, 'cuda_stream'):
                        del self.cuda_stream
                    logger.info("CUDA stream успешно освобожден")
                except Exception as e:
                    logger.warning(f"Ошибка при освобождении CUDA stream: {e}")
            
            self._cuda_context_valid = False
            self._resources_initialized = False
            logger.info("Все ресурсы успешно освобождены")
            
        except Exception as e:
            logger.warning(f"Общая ошибка при освобождении ресурсов: {e}")
            self._cuda_context_valid = False
            self._tensorrt_objects_valid = False
            self._resources_initialized = False

    def __del__(self):
        """Очистка ресурсов CUDA."""
        try:
            if self._resources_initialized:
                self.cleanup()
        except Exception as e:
            pass
    
    def __enter__(self):
        """Поддержка context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Поддержка context manager."""
        self.cleanup()

    def _set_binding_shapes(self, context, binding_shapes):
        """Устанавливает формы binding для контекста."""
        for binding_name, shape in binding_shapes.items():
            context.set_binding_shape(binding_name, shape)
    
    def predict_batch(self, pixel_values: torch.Tensor) -> List[str]:
        """Генерирует предсказания для пакета изображений с TensorRT."""
        pixel_values = pixel_values.to('cpu').numpy()
        batch_size = pixel_values.shape[0]
        
        predictions = []
        
        for i in range(batch_size):
            try:
                encoder_outputs = self._run_encoder(pixel_values[i:i+1])
                pred = self._generate_single(encoder_outputs)
                logger.info(f"Предсказание для изображения {i}: {pred}")
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Ошибка при генерации для изображения {i}: {e}")
                predictions.append("")
        
        return predictions
    
    def _generate_single(self, encoder_outputs: np.ndarray) -> str:
        """Генерирует предсказание для одного изображения."""
        
        if self.decoder_start_token_id is None:
            try:
                self.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)
                if self.decoder_start_token_id == self.processor.tokenizer.unk_token_id:
                    self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
            except:
                self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        
        # logger.info(f"Используем decoder_start_token_id: {decoder_start_token_id}")
        
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int32)
        
        generated_tokens = []
        max_length = self.max_length
        
        eos_token_id = self.processor.tokenizer.eos_token_id
        pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
        
        for step in range(max_length):
            try:
                logits = self._run_decoder(encoder_outputs, decoder_input_ids)
                next_token_logits = logits[0, -1, :]
                next_token_logits = next_token_logits / self.temperature
                top_k_indices = np.argpartition(next_token_logits, -self.top_k)[-self.top_k:]
                top_k_logits = next_token_logits[top_k_indices]
                exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                probs = exp_logits / np.sum(exp_logits)
                next_token_idx = np.argmax(probs)            
                next_token_id = int(top_k_indices[next_token_idx])
    
                if next_token_id == eos_token_id:
                    # logger.info(f"Встретили EOS токен на шаге {step}")
                    break
                
                if pad_token_id is not None and next_token_id == pad_token_id:
                    # logger.info(f"Встретили PAD токен на шаге {step}")
                    break
                
                if len(generated_tokens) >= 3 and all(t == next_token_id for t in generated_tokens[-3:]):
                    # logger.warning(f"Обнаружено зацикливание на токене {next_token_id}, останавливаем генерацию")
                    break
                
                generated_tokens.append(next_token_id)
                
                new_input = np.array([[next_token_id]], dtype=np.int32)
                decoder_input_ids = np.concatenate([decoder_input_ids, new_input], axis=1)
                    
            except Exception as e:
                logger.error(f"Ошибка на шаге {step}: {e}")
                break
        
        if generated_tokens:
            try:
                pred_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                pred_text = pred_text.strip()
            except Exception as e:
                logger.error(f"Ошибка при декодировании токенов: {e}")
                pred_text = ""
        else:
            logger.warning("Не сгенерировано ни одного токена!")
            pred_text = ""

        # logger.info(f"Сгенерированные токены: {generated_tokens[:10]}{'...' if len(generated_tokens) > 10 else ''}")
        # logger.info(f"Сгенерированный текст: '{pred_text}'")
        
        return pred_text
    
    def _run_encoder(self, pixel_values: np.ndarray) -> np.ndarray:
        """Запуск encoder через TensorRT."""
        if pixel_values.shape != self.encoder_input_shape:
            pixel_values = pixel_values.reshape(self.encoder_input_shape)
        pixel_values = pixel_values.astype(np.float32)
        
        try:
            self.cuda.memcpy_htod_async(
                self.encoder_gpu_buffers[self.encoder_input_name], 
                pixel_values, 
                self.cuda_stream
            )
            
            for name, gpu_buffer in self.encoder_gpu_buffers.items():
                self.encoder_context.set_tensor_address(name, int(gpu_buffer))
            
            success = self.encoder_context.execute_async_v3(self.cuda_stream.handle)
            if not success:
                raise RuntimeError("Encoder inference failed")
            
            self.cuda_stream.synchronize()
            
            output_buffer = self.encoder_buffers[self.encoder_output_name]
            self.cuda.memcpy_dtoh_async(
                output_buffer, 
                self.encoder_gpu_buffers[self.encoder_output_name], 
                self.cuda_stream
            )
            self.cuda_stream.synchronize()
            
            return output_buffer.copy()
            
        except Exception as e:
            logger.error(f"Ошибка в encoder: {e}")
            raise
    
    def _run_decoder(self, encoder_outputs: np.ndarray, decoder_input_ids: np.ndarray) -> np.ndarray:
        """Запуск decoder через TensorRT."""
        encoder_outputs = encoder_outputs.astype(np.float32)
        decoder_input_ids = decoder_input_ids.astype(np.int32)
        self.decoder_context.set_input_shape(self.decoder_input_ids_name, decoder_input_ids.shape)
        
        decoder_gpu_buffers = {}
        decoder_cpu_buffers = {}
        
        for name, tensor_info in self.decoder_inputs.items():
            if name == self.decoder_input_ids_name:
                shape = decoder_input_ids.shape
                data = decoder_input_ids
            else:
                shape = encoder_outputs.shape
                data = encoder_outputs
            
            size = int(np.prod(shape) * np.dtype(tensor_info['dtype']).itemsize)
            gpu_buffer = self.cuda.mem_alloc(size)
            decoder_gpu_buffers[name] = gpu_buffer
            
            self.cuda.memcpy_htod_async(gpu_buffer, data, self.cuda_stream)
        
        for name, tensor_info in self.decoder_outputs.items():
            actual_shape = self.decoder_context.get_tensor_shape(name)
            size = int(np.prod(actual_shape) * np.dtype(tensor_info['dtype']).itemsize)
            gpu_buffer = self.cuda.mem_alloc(size)
            decoder_gpu_buffers[name] = gpu_buffer
            decoder_cpu_buffers[name] = np.empty(actual_shape, dtype=tensor_info['dtype'])
        
        try:
            for name, gpu_buffer in decoder_gpu_buffers.items():
                self.decoder_context.set_tensor_address(name, int(gpu_buffer))
            
            success = self.decoder_context.execute_async_v3(self.cuda_stream.handle)
            if not success:
                raise RuntimeError("Decoder inference failed")
            
            self.cuda_stream.synchronize()
            
            output_buffer = decoder_cpu_buffers[self.decoder_output_name]
            self.cuda.memcpy_dtoh_async(
                output_buffer, 
                decoder_gpu_buffers[self.decoder_output_name], 
                self.cuda_stream
            )
            self.cuda_stream.synchronize()
            
            return output_buffer.copy()
            
        finally:
            for buffer in decoder_gpu_buffers.values():
                buffer.free()
    
    def evaluate_dataset(
        self, 
        data_module: DonutDataModule,
        split: str = "test",
        num_samples: Optional[int] = None,
        save_individual_results: bool = True
    ) -> Dict[str, Any]:
        """Оценивает модель на датасете и возвращает метрики."""
        
        if split == "test":
            dataloader = data_module.test_dataloader()
        elif split == "val" or split == "valid":
            dataloader = data_module.val_dataloader()
        else:
            dataloader = data_module.train_dataloader()
        
        if dataloader is None:
            raise ValueError(f"Загрузчик данных для разделения '{split}' не найден")
        
        all_metrics = {
            "cer": [],
            "wer": [],
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "inference_times": []
        }
        
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        batch_count = 0
        sample_count = 0
        
        max_batches = None if num_samples is None else (num_samples // self.batch_size) + 1

        for i, batch in enumerate(tqdm(dataloader, desc=f"Оценка ({split})", total=max_batches)):
            if max_batches is not None and i >= max_batches:
                break
            
            try:
                if len(batch) == 4:
                    pixel_values, labels, prompt_end_indices, target_sequences = batch
                else:
                    pixel_values, labels = batch
                    target_sequences = [""] * len(pixel_values)
                
                batch_size = pixel_values.shape[0]
                
                batch_start_time = time.time()
                
                pred_tokens = self.predict_batch(pixel_values)
                
                batch_inference_time = time.time() - batch_start_time
                all_metrics["inference_times"].extend([batch_inference_time / batch_size] * batch_size)
 
                for j, (pred, target) in enumerate(zip(pred_tokens, target_sequences)):
                    if sample_count >= num_samples and num_samples is not None:
                        break
                    
                    pred_clean = re.sub(r"<[^>]*>", "", pred).strip()
                    target_clean = re.sub(r"<[^>]*>", "", target).strip()
                    
                    all_predictions.append(pred_clean)
                    all_targets.append(target_clean)
                    
                    sample_metrics = {
                        "cer": self.metrics_calculator.calculate_cer(pred_clean, target_clean),
                        "wer": self.metrics_calculator.calculate_wer(pred_clean, target_clean)
                    }
                    
                    sample_rouge = self.metrics_calculator.calculate_rouge(pred_clean, target_clean)
                    sample_metrics.update(sample_rouge)
                    
                    for metric_name, value in sample_metrics.items():
                        all_metrics[metric_name].append(value)
                    
                    sample_count += 1
                
                batch_count += 1
                
            except Exception as e:
                logger.error(f"Ошибка при обработке пакета {i}: {e}")
                continue

        avg_metrics = {
            metric: np.mean(values) if values else 0.0
            for metric, values in all_metrics.items()
        }
        
        avg_metrics["num_samples"] = sample_count
        avg_metrics["total_time"] = time.time() - start_time
        avg_metrics["avg_time_per_sample"] = avg_metrics["total_time"] / max(1, sample_count)
        
        return avg_metrics


def create_comparison_table(results_1: Dict[str, Any], results_2: Dict[str, Any], output_dir: Path):
    """Создает таблицу сравнения двух моделей и сохраняет как изображение и JSON."""
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    metrics = ['CER', 'WER', 'ROUGE-2', 'ROUGE-L', 'Avg Time (ms)', 'Total Time (s)']
    
    model_1_name = results_1.get('model_name', 'Model 1')
    model_2_name = results_2.get('model_name', 'Model 2')
    
    data = [
        ['Метрика', model_1_name, model_2_name, 'Разница'],
        ['CER', f"{results_1['cer']:.4f}", f"{results_2['cer']:.4f}", f"{results_2['cer'] - results_1['cer']:.4f}"],
        ['WER', f"{results_1['wer']:.4f}", f"{results_2['wer']:.4f}", f"{results_2['wer'] - results_1['wer']:.4f}"],
        # ['ROUGE-1', f"{results_1['rouge-1']:.4f}", f"{results_2['rouge-1']:.4f}", f"{results_2['rouge-1'] - results_1['rouge-1']:.4f}"],
        ['ROUGE-2', f"{results_1['rouge-2']:.4f}", f"{results_2['rouge-2']:.4f}", f"{results_2['rouge-2'] - results_1['rouge-2']:.4f}"],
        ['ROUGE-L', f"{results_1['rouge-l']:.4f}", f"{results_2['rouge-l']:.4f}", f"{results_2['rouge-l'] - results_1['rouge-l']:.4f}"],
        ['Avg Time (ms)', f"{results_1['avg_time_per_sample']*1000:.2f}", f"{results_2['avg_time_per_sample']*1000:.2f}", f"{(results_2['avg_time_per_sample'] - results_1['avg_time_per_sample'])*1000:.2f}"],
        ['Total Time (s)', f"{results_1['total_time']:.2f}", f"{results_2['total_time']:.2f}", f"{results_2['total_time'] - results_1['total_time']:.2f}"],
        ['Samples', str(results_1['num_samples']), str(results_2['num_samples']), '-']
    ]
    
    comparison_data = {
        "metadata": {
            "comparison_timestamp": datetime.now().isoformat(),
            "model_1_name": model_1_name,
            "model_2_name": model_2_name,
            "total_samples_model_1": results_1['num_samples'],
            "total_samples_model_2": results_2['num_samples']
        },
        "metrics": {
            "cer": {
                "model_1": results_1['cer'],
                "model_2": results_2['cer'],
                "difference": results_2['cer'] - results_1['cer'],
                "improvement": "model_1" if results_1['cer'] < results_2['cer'] else "model_2",
                "improvement_percentage": abs((results_2['cer'] - results_1['cer']) / results_1['cer'] * 100) if results_1['cer'] != 0 else 0
            },
            "wer": {
                "model_1": results_1['wer'],
                "model_2": results_2['wer'],
                "difference": results_2['wer'] - results_1['wer'],
                "improvement": "model_1" if results_1['wer'] < results_2['wer'] else "model_2",
                "improvement_percentage": abs((results_2['wer'] - results_1['wer']) / results_1['wer'] * 100) if results_1['wer'] != 0 else 0
            },
            "rouge_1": {
                "model_1": results_1['rouge-1'],
                "model_2": results_2['rouge-1'],
                "difference": results_2['rouge-1'] - results_1['rouge-1'],
                "improvement": "model_2" if results_2['rouge-1'] > results_1['rouge-1'] else "model_1",
                "improvement_percentage": abs((results_2['rouge-1'] - results_1['rouge-1']) / results_1['rouge-1'] * 100) if results_1['rouge-1'] != 0 else 0
            },
            "rouge_2": {
                "model_1": results_1['rouge-2'],
                "model_2": results_2['rouge-2'],
                "difference": results_2['rouge-2'] - results_1['rouge-2'],
                "improvement": "model_2" if results_2['rouge-2'] > results_1['rouge-2'] else "model_1",
                "improvement_percentage": abs((results_2['rouge-2'] - results_1['rouge-2']) / results_1['rouge-2'] * 100) if results_1['rouge-2'] != 0 else 0
            },
            "rouge_l": {
                "model_1": results_1['rouge-l'],
                "model_2": results_2['rouge-l'],
                "difference": results_2['rouge-l'] - results_1['rouge-l'],
                "improvement": "model_2" if results_2['rouge-l'] > results_1['rouge-l'] else "model_1",
                "improvement_percentage": abs((results_2['rouge-l'] - results_1['rouge-l']) / results_1['rouge-l'] * 100) if results_1['rouge-l'] != 0 else 0
            },
            "avg_time_per_sample_ms": {
                "model_1": results_1['avg_time_per_sample'] * 1000,
                "model_2": results_2['avg_time_per_sample'] * 1000,
                "difference": (results_2['avg_time_per_sample'] - results_1['avg_time_per_sample']) * 1000,
                "improvement": "model_1" if results_1['avg_time_per_sample'] < results_2['avg_time_per_sample'] else "model_2",
                "speedup_factor": results_2['avg_time_per_sample'] / results_1['avg_time_per_sample'] if results_1['avg_time_per_sample'] != 0 else 0
            },
            "total_time_s": {
                "model_1": results_1['total_time'],
                "model_2": results_2['total_time'],
                "difference": results_2['total_time'] - results_1['total_time'],
                "improvement": "model_1" if results_1['total_time'] < results_2['total_time'] else "model_2",
                "speedup_factor": results_2['total_time'] / results_1['total_time'] if results_1['total_time'] != 0 else 0
            }
        },
        "summary": {
            "better_accuracy_model": None,
            "faster_model": None,
            "overall_winner": None
        }
    }
    
    model_1_accuracy_score = (
        (1 - results_1['cer']) +  # Чем меньше CER, тем лучше
        (1 - results_1['wer']) +  # Чем меньше WER, тем лучше
        results_1['rouge-2'] +     # Чем больше ROUGE, тем лучше
        results_1['rouge-l']
    ) / 4
    
    model_2_accuracy_score = (
        (1 - results_2['cer']) +
        (1 - results_2['wer']) +
        results_2['rouge-2'] +
        results_2['rouge-l']
    ) / 4
    
    comparison_data["summary"]["better_accuracy_model"] = model_1_name if model_1_accuracy_score > model_2_accuracy_score else model_2_name
    comparison_data["summary"]["faster_model"] = model_1_name if results_1['avg_time_per_sample'] < results_2['avg_time_per_sample'] else model_2_name
    
    model_1_overall_score = model_1_accuracy_score * 0.7 + (1 / results_1['avg_time_per_sample']) * 0.3
    model_2_overall_score = model_2_accuracy_score * 0.7 + (1 / results_2['avg_time_per_sample']) * 0.3
    
    comparison_data["summary"]["overall_winner"] = model_1_name if model_1_overall_score > model_2_overall_score else model_2_name
    
    comparison_json_file = output_dir / "model_comparison.json"
    with open(comparison_json_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Данные сравнения сохранены в {comparison_json_file}")
    
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', weight='bold')
        elif col == 3 and row > 0:
            if data[row][3] == '-':
                continue
            
            diff_value = float(data[row][3])
            metric_name = data[row][0]
            
            # Для CER, WER: меньше = лучше (отрицательная разница = улучшение)
            # Для ROUGE: больше = лучше (положительная разница = улучшение)
            if metric_name in ['CER', 'WER', 'Avg Time (ms)', 'Total Time (s)']:
                if diff_value < 0:
                    cell.set_facecolor('#C6EFCE')  # Зеленый для улучшения
                elif diff_value > 0:
                    cell.set_facecolor('#FFC7CE')  # Красный для ухудшения
            elif metric_name.startswith('ROUGE'):
                if diff_value > 0:
                    cell.set_facecolor('#C6EFCE')  # Зеленый для улучшения
                elif diff_value < 0:
                    cell.set_facecolor('#FFC7CE')  # Красный для ухудшения
        elif row % 2 == 0:
            cell.set_facecolor('#F2F2F2')
    
    comparison_file = output_dir / "model_comparison.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Таблица сравнения сохранена в {comparison_file}")
    
    return comparison_file, comparison_json_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Оценка и визуализация модели Donut")
    
    # Параметры модели
    model_group = parser.add_argument_group("Параметры модели")
    model_group.add_argument("--model_path", type=str, required=True,
                        help="Путь к обученной модели")
    model_group.add_argument("--image_size", type=int, nargs=2, default=[384, 384],
                        help="Размер изображения для модели [высота, ширина]")
    model_group.add_argument("--max_length", type=int, default=64,
                        help="Максимальная длина генерации")
    model_group.add_argument("--num_beams", type=int, default=5,
                        help="Количество лучей для генерации")
    model_group.add_argument("--task_start_token", type=str, default='<s_500k>',
                        help="Токен начала задачи (если None, берется из модели)")
    model_group.add_argument("--prompt_end_token", type=str, default=None,
                        help="Токен конца промпта (если None, берется из модели)")
    
    # Параметры вывода
    output_group = parser.add_argument_group("Параметры вывода")
    output_group.add_argument("--output_dir", type=str, default="./output/inference_results",
                        help="Директория для сохранения результатов")
    output_group.add_argument("--save_visualizations", action="store_true", default=True,
                        help="Сохранять визуализации результатов")
    output_group.add_argument("--batch_size", type=int, default=1,
                        help="Размер пакета для инференса")
    
    # Режимы работы
    mode_group = parser.add_argument_group("Режимы работы")
    mode_group.add_argument("--mode", type=str, choices=["single_image", "dataset", "both", "compare"], default="dataset",
                        help="Режим работы: одиночное изображение, датасет, оба или сравнение моделей")
    mode_group.add_argument("--image_path", type=str, default=None,
                        help="Путь к изображению для одиночной оценки")
    mode_group.add_argument("--data_dir", type=str, default='./dataset/ocr/donut/real',
                        help="Директория с данными для оценки датасета")
    mode_group.add_argument("--split", type=str, choices=["train", "valid", "test"], default="valid",
                        help="Раздел данных для оценки")
    mode_group.add_argument("--num_samples", type=int, default=10,
                        help="Количество образцов для оценки (если None, все доступные)")
    
    # Параметры сравнения моделей
    compare_group = parser.add_argument_group("Параметры сравнения моделей")
    compare_group.add_argument("--model_path_2", type=str, default='./output/tensorrt',
                        help="Путь ко второй модели для сравнения")
    compare_group.add_argument("--model_type_1", type=str, choices=["pytorch", "tensorrt"], default="pytorch",
                        help="Тип первой модели")
    compare_group.add_argument("--model_type_2", type=str, choices=["pytorch", "tensorrt"], default="tensorrt",
                        help="Тип второй модели")
    
    # Параметры вычислений
    compute_group = parser.add_argument_group("Параметры вычислений")
    compute_group.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")
    compute_group.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Точность вычислений")
    compute_group.add_argument("--num_workers", type=int, default=4,
                        help="Количество рабочих процессов для загрузки данных")
    
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if args.mode == "single_image" and args.image_path is None:
        parser.error("Для режима 'single_image' требуется указать --image_path")
        
    if args.mode in ["dataset", "both"] and args.data_dir is None:
        parser.error("Для режима 'dataset' или 'both' требуется указать --data_dir")
    
    return args


def setup_model(args):
    logger.info(f"Загрузка модели из {args.model_path}")
    
    return DonutModel.from_pretrained(
        args.model_path,
        device=args.device,
        precision=args.precision,
        max_length=args.max_length,
        image_size=args.image_size,
        task_start_token=args.task_start_token,
        prompt_end_token=args.prompt_end_token
    )


def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "inference_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    MemoryOptimizer.optimize_memory_usage()

    model = setup_model(args)

    inference = DonutInference(
        model=model,
        output_dir=output_dir,
        num_beams=args.num_beams,
        max_length=args.max_length,
        device=args.device,
        save_visualizations=args.save_visualizations,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    if args.mode in ["single_image", "both"] and args.image_path is not None:
        try:
            result = inference.process_sample(args.image_path)
            
            result_file = output_dir / "single_image_result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Результат для {args.image_path}:")
            logger.info(f"Время инференса: {result['inference_time']:.4f} сек")
            
            if "error" in result:
                logger.error(f"Ошибка: {result['error']}")
            else:
                logger.info(f"Предсказание: {json.dumps(result['predictions'], ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {args.image_path}: {e}")
    
    if args.mode in ["dataset", "both"] and args.data_dir is not None:
        try:
            data_module = DonutDataModule(
                processor=model.processor,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_length=args.max_length,
                task_start_token=model.task_start_token,
                prompt_end_token=model.prompt_end_token if hasattr(model, "prompt_end_token") else None,
                sort_json_key=True,
                image_size=args.image_size,
                apply_augmentation=False
            )
            
            metrics = inference.evaluate_dataset(
                data_module=data_module,
                split=args.split,
                num_samples=args.num_samples,
                save_individual_results=args.save_visualizations
            )

            logger.info(f"Метрики для {args.split} ({metrics['num_samples']} образцов):")
            logger.info(f"CER: {metrics['cer']:.4f}, WER: {metrics['wer']:.4f}")
            logger.info(f"ROUGE-1: {metrics['rouge-1']:.4f}, ROUGE-2: {metrics['rouge-2']:.4f}, ROUGE-L: {metrics['rouge-l']:.4f}")
            logger.info(f"Среднее время на образец: {metrics['avg_time_per_sample']:.4f} сек")
            
        except Exception as e:
            logger.error(f"Ошибка при оценке датасета: {e}", exc_info=True)
    
    if args.mode == "compare":
        if args.model_path_2 is None:
            logger.error("Для режима сравнения требуется указать --model_path_2")
            return 1
        
        try:
            if args.model_type_1 == "pytorch":
                model_1 = setup_model(args)
                inference_1 = DonutInference(
                    model=model_1,
                    output_dir=output_dir / "model_1",
                    num_beams=args.num_beams,
                    max_length=args.max_length,
                    device=args.device,
                    save_visualizations=False,
                    batch_size=args.batch_size,
                    image_size=args.image_size
                )
            elif args.model_type_1 == "tensorrt":
                inference_1 = DonutInferenceTRT(
                    tensorrt_dir=args.model_path,
                    device=args.device,
                    batch_size=args.batch_size
                )
            
            args_temp = args
            args_temp.model_path = args.model_path_2
            if args.model_type_2 == "pytorch":
                model_2 = setup_model(args_temp)
                inference_2 = DonutInference(
                    model=model_2,
                    output_dir=output_dir / "model_2",
                    num_beams=args.num_beams,
                    max_length=args.max_length,
                    device=args.device,
                    save_visualizations=False,
                    batch_size=args.batch_size,
                    image_size=args.image_size
                )
            elif args.model_type_2 == "tensorrt":
                inference_2 = DonutInferenceTRT(
                    tensorrt_dir=args.model_path_2,
                    device=args.device,
                    batch_size=args.batch_size
                )
            
            data_module = DonutDataModule(
                processor=model_1.processor if args.model_type_1 == "pytorch" else inference_1.processor,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_length=args.max_length,
                task_start_token=model_1.task_start_token if args.model_type_1 == "pytorch" else inference_1.task_start_token,
                prompt_end_token=model_1.prompt_end_token if args.model_type_1 == "pytorch" else inference_1.prompt_end_token,
                sort_json_key=True,
                image_size=args.image_size,
                apply_augmentation=False
            )
            
            logger.info("Оценка первой модели...")
            results_1 = inference_1.evaluate_dataset(
                data_module=data_module,
                split=args.split,
                num_samples=args.num_samples,
                save_individual_results=False
            )
            results_1['model_name'] = f"{args.model_type_1.upper()} Model 1"
            
            logger.info("Оценка второй модели...")
            results_2 = inference_2.evaluate_dataset(
                data_module=data_module,
                split=args.split,
                num_samples=args.num_samples,
                save_individual_results=False
            )
            results_2['model_name'] = f"{args.model_type_2.upper()} Model 2"
            
            comparison_file, comparison_json_file = create_comparison_table(results_1, results_2, output_dir)
            
            logger.info(f"Сравнение моделей завершено. Таблица сохранена в {comparison_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении моделей: {e}", exc_info=True)
            return 1

    logger.info(f"Результаты сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
