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
from transformers import DonutProcessor

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
        
        # Режим оценки
        self.model.eval()
        
        all_metrics = {
            "cer": [],
            "wer": [],
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "accuracy": [],
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
                    
                    sample_metrics["accuracy"] = 1.0 - sample_metrics["cer"]
                    
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
            ["ROUGE-L", f"{metrics['rouge-l']:.4f}"],
            ["Точность", f"{metrics['accuracy']:.4f}"]
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
        if metrics["rouge-l"]:
            sns.histplot(metrics["rouge-l"], kde=True, ax=ax2, color="purple", alpha=0.5, label="ROUGE-L")
        
        ax2.set_title("Распределение метрик ROUGE", fontsize=14)
        ax2.set_xlabel("Значение метрики", fontsize=12)
        ax2.set_ylabel("Частота", fontsize=12)
        ax2.legend()

        ax3 = axes[1, 0]
        if metrics["accuracy"]:
            sns.histplot(metrics["accuracy"], kde=True, ax=ax3, color="orange", alpha=0.7)
        
        ax3.set_title("Распределение точности", fontsize=14)
        ax3.set_xlabel("Точность", fontsize=12)
        ax3.set_ylabel("Частота", fontsize=12)

        ax4 = axes[1, 1]
        if metrics["inference_times"]:
            sns.histplot(metrics["inference_times"], kde=True, ax=ax4, color="teal", alpha=0.7)
        
        ax4.set_title("Распределение времени инференса", fontsize=14)
        ax4.set_xlabel("Время (сек)", fontsize=12)
        ax4.set_ylabel("Частота", fontsize=12)

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
                'ROUGE-L': metrics['rouge-l'],
                'Accuracy': metrics['accuracy'],
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
                f"CER: {sample['cer']:.4f}, WER: {sample['wer']:.4f}, ROUGE-L: {sample['rouge']['rouge-l']:.4f}\n"
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
        engine_path: str,
        config_path: str,
        device: str = "cuda",
        batch_size: int = 1
    ):
        self.engine_path = engine_path
        self.config_path = config_path
        self.device = device
        self.batch_size = batch_size
        
        # Загружаем конфигурацию
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.task_start_token = self.config['task_start_token']
        self.prompt_end_token = self.config.get('prompt_end_token', self.task_start_token)
        self.max_length = self.config['max_length']
        self.image_size = tuple(self.config['image_size'])
        
        # Инициализируем TensorRT
        self._init_tensorrt()
        
        # Создаем процессор
        from transformers import DonutProcessor
        self.processor = DonutProcessor.from_pretrained(self.config['model_path'])
        
        logger.info(f"Инициализирован TensorRT инференс для {engine_path}")
    
    def _init_tensorrt(self):
        """Инициализация TensorRT runtime."""
        try:
            import tensorrt as trt
            from tensorrt import Logger, Runtime
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                self.cuda_available = True
            except ImportError:
                logger.warning("PyCUDA не установлен. TensorRT inference будет работать медленнее.")
                self.cuda_available = False
        except ImportError:
            raise ImportError("TensorRT не установлен")
        
        TRT_LOGGER = Logger(trt.Logger.WARNING)
        self.runtime = Runtime(TRT_LOGGER)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Получаем индексы входов/выходов
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        
        logger.info(f"TensorRT bindings: inputs={self.input_names}, outputs={self.output_names}")
    
    def predict_batch(self, pixel_values: torch.Tensor) -> List[str]:
        """Генерирует предсказания для пакета изображений с TensorRT."""
        import torch
        import numpy as np
        
        pixel_values = pixel_values.to('cpu').numpy()
        batch_size = pixel_values.shape[0]
        
        # Для простоты используем последовательную генерацию
        predictions = []
        
        for i in range(batch_size):
            pred = self._generate_single(pixel_values[i:i+1])
            predictions.append(pred)
        
        return predictions
    
    def _generate_single(self, pixel_values: np.ndarray) -> str:
        """Генерирует предсказание для одного изображения."""
        import torch
        import numpy as np
        
        # Encoder inference
        encoder_outputs = self._run_encoder(pixel_values)
        
        # Decoder generation
        decoder_input_ids = torch.tensor([[self.config['decoder_start_token_id']]], dtype=torch.long)
        
        generated_tokens = []
        max_length = self.max_length
        
        for _ in range(max_length):
            logits = self._run_decoder(encoder_outputs, decoder_input_ids.numpy())
            
            next_token_id = np.argmax(logits[0, -1, :])
            
            if next_token_id == self.processor.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token_id)
            decoder_input_ids = torch.cat([
                decoder_input_ids, 
                torch.tensor([[next_token_id]], dtype=torch.long)
            ], dim=1)
        
        # Декодируем токены
        pred_tokens = self.processor.tokenizer.convert_ids_to_tokens(generated_tokens)
        pred_text = self.processor.tokenizer.convert_tokens_to_string(pred_tokens)
        
        return pred_text
    
    def _run_encoder(self, pixel_values: np.ndarray) -> np.ndarray:
        """Запуск encoder через TensorRT."""
        if not self.cuda_available:
            raise RuntimeError("PyCUDA не установлен, TensorRT inference недоступен")
        
        # Предполагаем, что у нас есть encoder engine
        encoder_engine_path = self.engine_path.replace('.engine', '_encoder.engine')
        
        if not Path(encoder_engine_path).exists():
            raise FileNotFoundError(f"Encoder engine не найден: {encoder_engine_path}")
        
        # Загружаем encoder engine
        with open(encoder_engine_path, 'rb') as f:
            encoder_engine_data = f.read()
        
        encoder_engine = self.runtime.deserialize_cuda_engine(encoder_engine_data)
        encoder_context = encoder_engine.create_execution_context()
        
        # Выделяем память
        d_input = cuda.mem_alloc(pixel_values.nbytes)
        d_output = cuda.mem_alloc(encoder_engine.get_binding_shape(1).numel() * 4)  # float32
        
        # Копируем входные данные
        cuda.memcpy_htod(d_input, pixel_values)
        
        # Запуск
        encoder_context.execute_v2([int(d_input), int(d_output)])
        
        # Копируем результат
        output_shape = encoder_engine.get_binding_shape(1)
        output_size = output_shape.numel() * 4
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        return output
    
    def _run_decoder(self, encoder_outputs: np.ndarray, decoder_input_ids: np.ndarray) -> np.ndarray:
        """Запуск decoder через TensorRT."""
        if not self.cuda_available:
            raise RuntimeError("PyCUDA не установлен, TensorRT inference недоступен")
        
        # Аналогично для decoder
        decoder_engine_path = self.engine_path.replace('.engine', '_decoder.engine')
        
        if not Path(decoder_engine_path).exists():
            raise FileNotFoundError(f"Decoder engine не найден: {decoder_engine_path}")
        
        with open(decoder_engine_path, 'rb') as f:
            decoder_engine_data = f.read()
        
        decoder_engine = self.runtime.deserialize_cuda_engine(decoder_engine_data)
        decoder_context = decoder_engine.create_execution_context()
        
        # Выделяем память
        d_encoder = cuda.mem_alloc(encoder_outputs.nbytes)
        d_decoder_input = cuda.mem_alloc(decoder_input_ids.nbytes)
        d_output = cuda.mem_alloc(decoder_engine.get_binding_shape(2).numel() * 4)
        
        # Копируем данные
        cuda.memcpy_htod(d_encoder, encoder_outputs)
        cuda.memcpy_htod(d_decoder_input, decoder_input_ids)
        
        # Запуск
        decoder_context.execute_v2([int(d_encoder), int(d_decoder_input), int(d_output)])
        
        # Копируем результат
        output_shape = decoder_engine.get_binding_shape(2)
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        
        return output
    
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
            "accuracy": [],
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
                        "cer": MetricsCalculator.calculate_cer(pred_clean, target_clean),
                        "wer": MetricsCalculator.calculate_wer(pred_clean, target_clean)
                    }
                    
                    sample_rouge = MetricsCalculator.calculate_rouge(pred_clean, target_clean)
                    sample_metrics.update(sample_rouge)
                    
                    sample_metrics["accuracy"] = 1.0 - sample_metrics["cer"]
                    
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
    """Создает таблицу сравнения двух моделей и сохраняет как изображение."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Данные для таблицы
    metrics = ['CER', 'WER', 'ROUGE-1', 'ROUGE-L', 'Accuracy', 'Avg Time (ms)', 'Total Time (s)']
    
    model_1_name = results_1.get('model_name', 'Model 1')
    model_2_name = results_2.get('model_name', 'Model 2')
    
    data = [
        ['Метрика', model_1_name, model_2_name, 'Разница'],
        ['CER', f"{results_1['cer']:.4f}", f"{results_2['cer']:.4f}", f"{results_2['cer'] - results_1['cer']:.4f}"],
        ['WER', f"{results_1['wer']:.4f}", f"{results_2['wer']:.4f}", f"{results_2['wer'] - results_1['wer']:.4f}"],
        ['ROUGE-1', f"{results_1['rouge-1']:.4f}", f"{results_2['rouge-1']:.4f}", f"{results_2['rouge-1'] - results_1['rouge-1']:.4f}"],
        ['ROUGE-L', f"{results_1['rouge-l']:.4f}", f"{results_2['rouge-l']:.4f}", f"{results_2['rouge-l'] - results_1['rouge-l']:.4f}"],
        ['Accuracy', f"{results_1['accuracy']:.4f}", f"{results_2['accuracy']:.4f}", f"{results_2['accuracy'] - results_1['accuracy']:.4f}"],
        ['Avg Time (ms)', f"{results_1['avg_time_per_sample']*1000:.2f}", f"{results_2['avg_time_per_sample']*1000:.2f}", f"{(results_2['avg_time_per_sample'] - results_1['avg_time_per_sample'])*1000:.2f}"],
        ['Total Time (s)', f"{results_1['total_time']:.2f}", f"{results_2['total_time']:.2f}", f"{results_2['total_time'] - results_1['total_time']:.2f}"],
        ['Samples', str(results_1['num_samples']), str(results_2['num_samples']), '-']
    ]
    
    table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2.0)
    
    # Стилизация
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', weight='bold')
        elif col == 3 and row > 0:  # Колонка разницы
            diff_value = float(data[row][3]) if data[row][3] != '-' else 0
            if diff_value < 0:
                cell.set_facecolor('#C6EFCE')  # Зеленый для улучшения
            elif diff_value > 0:
                cell.set_facecolor('#FFC7CE')  # Красный для ухудшения
        elif row % 2 == 0:
            cell.set_facecolor('#F2F2F2')
    
    plt.title('Сравнение моделей Donut', fontsize=16, pad=20)
    
    comparison_file = output_dir / "model_comparison.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Таблица сравнения сохранена в {comparison_file}")
    
    return comparison_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="Оценка и визуализация модели Donut")
    
    # Параметры модели
    model_group = parser.add_argument_group("Параметры модели")
    model_group.add_argument("--model_path", type=str, required=True,
                        help="Путь к обученной модели")
    model_group.add_argument("--image_size", type=int, nargs=2, default=[1280, 960],
                        help="Размер изображения для модели [высота, ширина]")
    model_group.add_argument("--max_length", type=int, default=768,
                        help="Максимальная длина генерации")
    model_group.add_argument("--num_beams", type=int, default=5,
                        help="Количество лучей для генерации")
    model_group.add_argument("--task_start_token", type=str, default=None,
                        help="Токен начала задачи (если None, берется из модели)")
    model_group.add_argument("--prompt_end_token", type=str, default=None,
                        help="Токен конца промпта (если None, берется из модели)")
    
    # Параметры вывода
    output_group = parser.add_argument_group("Параметры вывода")
    output_group.add_argument("--output_dir", type=str, default="./inference_results",
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
    mode_group.add_argument("--data_dir", type=str, default=None,
                        help="Директория с данными для оценки датасета")
    mode_group.add_argument("--split", type=str, choices=["train", "valid", "test"], default="test",
                        help="Раздел данных для оценки")
    mode_group.add_argument("--num_samples", type=int, default=None,
                        help="Количество образцов для оценки (если None, все доступные)")
    
    # Параметры сравнения моделей
    compare_group = parser.add_argument_group("Параметры сравнения моделей")
    compare_group.add_argument("--model_path_2", type=str, default=None,
                        help="Путь ко второй модели для сравнения")
    compare_group.add_argument("--model_type_1", type=str, choices=["pytorch", "tensorrt"], default="pytorch",
                        help="Тип первой модели")
    compare_group.add_argument("--model_type_2", type=str, choices=["pytorch", "tensorrt"], default="pytorch",
                        help="Тип второй модели")
    compare_group.add_argument("--tensorrt_config_1", type=str, default=None,
                        help="Путь к конфигу TensorRT для первой модели")
    compare_group.add_argument("--tensorrt_config_2", type=str, default=None,
                        help="Путь к конфигу TensorRT для второй модели")
    
    # Параметры вычислений
    compute_group = parser.add_argument_group("Параметры вычислений")
    compute_group.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")
    compute_group.add_argument("--precision", type=str, default="fp32",
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
            logger.info(f"ROUGE-1: {metrics['rouge-1']:.4f}, ROUGE-L: {metrics['rouge-l']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Среднее время на образец: {metrics['avg_time_per_sample']:.4f} сек")
            
        except Exception as e:
            logger.error(f"Ошибка при оценке датасета: {e}", exc_info=True)
    
    if args.mode == "compare":
        if args.model_path_2 is None:
            logger.error("Для режима сравнения требуется указать --model_path_2")
            return 1
        
        try:
            # Загружаем первую модель
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
                if args.tensorrt_config_1 is None:
                    logger.error("Для TensorRT модели требуется указать --tensorrt_config_1")
                    return 1
                inference_1 = DonutInferenceTRT(
                    engine_path=args.model_path,
                    config_path=args.tensorrt_config_1,
                    device=args.device,
                    batch_size=args.batch_size
                )
            
            # Загружаем вторую модель
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
                if args.tensorrt_config_2 is None:
                    logger.error("Для TensorRT модели требуется указать --tensorrt_config_2")
                    return 1
                inference_2 = DonutInferenceTRT(
                    engine_path=args.model_path_2,
                    config_path=args.tensorrt_config_2,
                    device=args.device,
                    batch_size=args.batch_size
                )
            
            # Оцениваем обе модели
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
            
            # Создаем таблицу сравнения
            comparison_file = create_comparison_table(results_1, results_2, output_dir)
            
            logger.info(f"Сравнение моделей завершено. Таблица сохранена в {comparison_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении моделей: {e}", exc_info=True)
            return 1

    logger.info(f"Результаты сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())