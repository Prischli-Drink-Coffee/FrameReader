import gc
import sys
import logging
import re
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from nltk.metrics import distance as nltk_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingSpeedup:
    
    @staticmethod
    def setup_distributed() -> Tuple[int, int, int]:
        if not torch.distributed.is_available():
            return
        
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
            
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = local_rank
        
        torch.cuda.set_device(device)
        return local_rank, world_size, device
    
    @staticmethod
    def wrap_model_for_distributed(model: Any, local_rank: int) -> Any:
        if torch.cuda.is_available():
            model = model.to(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True
            )
        
        return model
    
    @staticmethod
    def get_mixed_precision_scaler(device_type: str, precision: str, enabled: bool = True) -> Optional[Any]:
        if precision not in ["fp16", "bf16", "fp32"]:
            raise ValueError(f"Неподдерживаемая точность: {precision}. Используйте fp16, bf16 или fp32")
        
        if precision == "fp32":
            return None
        
        if precision == "bf16":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            return None
        
        if precision == "fp16" and enabled:
            try:
                from torch.cuda.amp import GradScaler
                return GradScaler()
            except ImportError:
                logger.warning("torch.cuda.amp.GradScaler не найден. Используется fp32")
                
        return None


class MemoryOptimizer:
    
    @staticmethod
    def optimize_memory_usage() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Выполнена оптимизация памяти")
    
    @staticmethod
    def print_memory_usage() -> Dict[str, float]:
        memory_info = {"cpu_allocated_gb": 0.0}
        
        if torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
            memory_info["gpu_max_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        memory_info["cpu_allocated_gb"] = 0.0
        
        logger.info(f"Использование памяти: {memory_info}")
        return memory_info


class MetricsCalculator:
    @staticmethod
    def calculate_cer(pred_text: str, target_text: str) -> float:
        """
        Рассчитывает Character Error Rate (CER).
        CER = min(1.0, (S + D + I) / N), где S - замены, D - удаления, I - вставки, N - длина целевой строки
        """
        if not target_text:
            return 1.0 if pred_text else 0.0
        
        distances = nltk_distance.edit_distance(pred_text, target_text)
        return min(1.0, distances / len(target_text))
    
    @staticmethod
    def calculate_wer(pred_text: str, target_text: str) -> float:
        """
        Рассчитывает Word Error Rate (WER).
        """
        pred_words = re.findall(r'\w+', pred_text.lower())
        target_words = re.findall(r'\w+', target_text.lower())
        
        if not target_words:
            return 1.0 if pred_words else 0.0
        
        distances = nltk_distance.edit_distance(pred_words, target_words)
        return min(1.0, distances / len(target_words))
    
    @staticmethod
    def calculate_rouge(pred_text: str, target_text: str) -> Dict[str, float]:
        """
        Упрощенная имплементация ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
        """
        if not pred_text.strip():
            # logger.warning(f"Пустое предсказание при вычислении Rouge. Цель: '{target_text[:30]}...'")
            return {
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0
            }
            
        if not target_text.strip():
            # logger.warning(f"Пустая цель при вычислении Rouge. Предсказание: '{pred_text[:30]}...'")
            return {
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0
            }
        
        try:
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(pred_text, target_text)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except ImportError:
            logger.warning("Rouge не установлен. Возвращаются нулевые метрики.")
            return {
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0
            }
        except Exception as e:
            logger.warning(f"Ошибка при вычислении Rouge: {e}. Предсказание: '{pred_text[:30]}...', Цель: '{target_text[:30]}...'")
            return {
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0
            }
    
    @staticmethod
    def evaluate_predictions(preds: List[str], targets: List[str]) -> Dict[str, float]:
        """
        Оценивает предсказания по нескольким метрикам.
        """
        metrics = {
            "cer": 0.0,
            "wer": 0.0,
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0,
        }
        
        if not preds or not targets or len(preds) != len(targets):
            logger.warning(f"Невозможно оценить предсказания: длина предсказаний={len(preds) if preds else 0}, длина целей={len(targets) if targets else 0}")
            return metrics
        
        total_samples = len(preds)
        
        for pred, target in zip(preds, targets):
            metrics["cer"] += MetricsCalculator.calculate_cer(pred, target)
            metrics["wer"] += MetricsCalculator.calculate_wer(pred, target)
            
            rouge_scores = MetricsCalculator.calculate_rouge(pred, target)
            for key, value in rouge_scores.items():
                metrics[key] += value
        
        for key in metrics:
            metrics[key] /= max(total_samples, 1)
            
        return metrics


class MetricsVisualizer:
    """Класс для визуализации метрик обучения."""
    
    def __init__(
        self, 
        output_dir: Union[str, Path],
        dpi: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        self.metrics_history = {
            'train_steps': [],
            'train_epochs': [],
            'train_loss': [],
            'train_cer': [],
            'train_wer': [],
            'learning_rates': [],
            'rouge_scores': [],

            'val_steps': [],
            'val_epochs': [],
            'val_loss': [],
            'val_cer': [],
            'val_wer': [],
            'val_rouge_scores': [],
        }
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plotting_available = True
            
            plt.style.use('ggplot')
            sns.set_style("whitegrid")
            
            logger.info("Библиотеки визуализации успешно импортированы")
        except ImportError:
            self.plotting_available = False
            logger.warning("Matplotlib или seaborn не установлены, визуализация недоступна")
    
    def update_metrics(
        self, 
        metrics: Dict[str, float], 
        step: int,
        epoch: float,
        is_val: bool = False,
    ) -> None:
        """
        Обновляет историю метрик и визуализирует их, если прошло достаточное количество шагов.
        
        Args:
            metrics: Словарь с метриками
            step: Текущий глобальный шаг
            epoch: Текущая эпоха (может быть дробной)
            is_val: Флаг, указывающий, что метрики получены на валидационном наборе
        """
        if not is_val:
            self.metrics_history['train_steps'].append(step)
            self.metrics_history['train_epochs'].append(epoch)
            
            for metric_name, value in metrics.items():
                if metric_name == 'train_loss':
                    self.metrics_history['train_loss'].append(value)
                elif metric_name == 'train_cer':
                    self.metrics_history['train_cer'].append(value)
                elif metric_name == 'train_wer':
                    self.metrics_history['train_wer'].append(value)
                elif metric_name == 'learning_rate':
                    self.metrics_history['learning_rates'].append(value)
            
            if any(k.startswith('train_rouge') for k in metrics):
                rouge_dict = {k.replace('train_', ''): v for k, v in metrics.items() if k.startswith('train_rouge')}
                self.metrics_history['rouge_scores'].append(rouge_dict)
        else:
            self.metrics_history['val_steps'].append(step)
            self.metrics_history['val_epochs'].append(epoch)
            
            for metric_name, value in metrics.items():
                if metric_name == 'val_loss':
                    self.metrics_history['val_loss'].append(value)
                elif metric_name == 'val_cer':
                    self.metrics_history['val_cer'].append(value)
                elif metric_name == 'val_wer':
                    self.metrics_history['val_wer'].append(value)
            
            if any(k.startswith('val_rouge') for k in metrics):
                rouge_dict = {k.replace('val_', ''): v for k, v in metrics.items() if k.startswith('val_rouge')}
                self.metrics_history['val_rouge_scores'].append(rouge_dict)
        
        self.visualize_metrics(step, epoch)
        self.save_metrics(step)
    
    def save_metrics(self, step: int) -> None:
        """
        Сохраняет текущие метрики в JSON файл.
        
        Args:
            step: Текущий глобальный шаг
        """
        try:
            import json
            
            serializable_metrics = {}
            for key, value in self.metrics_history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], (int, float, str, bool, dict)):
                        serializable_metrics[key] = value
    
            metrics_file = self.output_dir / "metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"Ошибка при сохранении метрик: {e}")

    def visualize_metrics(
        self, 
        step: int, 
        epoch: float
    ) -> None:
        """
        Визуализирует метрики обучения и сохраняет графики.
        
        Args:
            step: Текущий глобальный шаг
            epoch: Текущая эпоха (может быть дробной)
        """
        if not self.plotting_available:
            return
            
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(18, 12))

            has_train_loss = len(self.metrics_history['train_steps']) > 0 and len(self.metrics_history['train_loss']) > 0
            has_val_loss = len(self.metrics_history['val_steps']) > 0 and len(self.metrics_history['val_loss']) > 0
            has_train_cer = len(self.metrics_history['train_cer']) > 0
            has_val_cer = len(self.metrics_history['val_cer']) > 0
            has_train_wer = len(self.metrics_history['train_wer']) > 0
            has_val_wer = len(self.metrics_history['val_wer']) > 0
            has_rouge = len(self.metrics_history['rouge_scores']) > 0
            has_val_rouge = len(self.metrics_history['val_rouge_scores']) > 0
            has_lr = len(self.metrics_history['learning_rates']) > 0

            # График функции потери
            plt.subplot(3, 2, 1)
            if has_train_loss:
                plt.plot(self.metrics_history['train_epochs'], self.metrics_history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
            if has_val_loss:
                plt.plot(self.metrics_history['val_epochs'], self.metrics_history['val_loss'], 'r-', label='Val Loss', alpha=0.9, linewidth=2)
            plt.title('Функция потерь')
            plt.xlabel('Эпоха')
            plt.ylabel('Потеря')
            plt.legend()
            plt.grid(True)
            
            # График CER
            plt.subplot(3, 2, 2)
            if has_train_cer:
                cer_epochs = self.metrics_history['train_epochs'][:len(self.metrics_history['train_cer'])]
                plt.plot(cer_epochs, self.metrics_history['train_cer'], 'b-', label='Train CER', alpha=0.7)
            if has_val_cer:
                plt.plot(self.metrics_history['val_epochs'], self.metrics_history['val_cer'], 'r-', label='Val CER', alpha=0.9, linewidth=2)
                
            plt.title('Character Error Rate (CER)')
            plt.xlabel('Эпоха')
            plt.ylabel('CER')
            plt.legend()
            plt.grid(True)
            
            # График WER
            plt.subplot(3, 2, 3)
            if has_train_wer:
                wer_epochs = self.metrics_history['train_epochs'][:len(self.metrics_history['train_wer'])]
                plt.plot(wer_epochs, self.metrics_history['train_wer'], 'b-', label='Train WER', alpha=0.7)
            if has_val_wer:
                plt.plot(self.metrics_history['val_epochs'], self.metrics_history['val_wer'], 'r-', label='Val WER', alpha=0.9, linewidth=2)
                
            plt.title('Word Error Rate (WER)')
            plt.xlabel('Эпоха')
            plt.ylabel('WER')
            plt.legend()
            plt.grid(True)
            
            # График ROUGE метрик
            plt.subplot(3, 2, 4)
            if has_rouge:
                rouge_scores = self.metrics_history['rouge_scores']
                epochs = self.metrics_history['train_epochs'][:len(rouge_scores)]
                    
                rouge1 = [score.get('rouge-1', 0) for score in rouge_scores]
                rouge2 = [score.get('rouge-2', 0) for score in rouge_scores]
                rougeL = [score.get('rouge-l', 0) for score in rouge_scores]
                
                plt.plot(epochs, rouge1, 'g-', label='Train ROUGE-1', alpha=0.8)
                plt.plot(epochs, rouge2, 'b-', label='Train ROUGE-2', alpha=0.8)
                plt.plot(epochs, rougeL, 'r-', label='Train ROUGE-L', alpha=0.8)
            
            if has_val_rouge:
                val_rouge_scores = self.metrics_history['val_rouge_scores']
                val_epochs = self.metrics_history['val_epochs'][:len(val_rouge_scores)]
                    
                val_rouge1 = [score.get('rouge-1', 0) for score in val_rouge_scores]
                val_rouge2 = [score.get('rouge-2', 0) for score in val_rouge_scores]
                val_rougeL = [score.get('rouge-l', 0) for score in val_rouge_scores]
                
                plt.plot(val_epochs, val_rouge1, 'g--', label='Val ROUGE-1', alpha=0.8)
                plt.plot(val_epochs, val_rouge2, 'b--', label='Val ROUGE-2', alpha=0.8)
                plt.plot(val_epochs, val_rougeL, 'r--', label='Val ROUGE-L', alpha=0.8)
                
            plt.title('ROUGE метрики')
            plt.xlabel('Эпоха')
            plt.ylabel('Значение')
            plt.legend()
            plt.grid(True)
            
            # График Learning Rate
            plt.subplot(3, 2, 5)
            if has_lr:
                lr_epochs = self.metrics_history['train_epochs'][:len(self.metrics_history['learning_rates'])]
                plt.plot(lr_epochs, self.metrics_history['learning_rates'], 'purple', label='Learning Rate', alpha=0.8)
            plt.title('Learning Rate')
            plt.xlabel('Эпоха')
            plt.ylabel('LR')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()

            (self.output_dir / "plot").mkdir(parents=True, exist_ok=True)
            
            epoch_filename = self.output_dir / "plot" / f"metrics_epoch_{int(epoch)}.png"
            plt.savefig(epoch_filename, dpi=self.dpi)
                
            plt.close()
            
        except Exception as e:
            logger.warning(f"Ошибка при визуализации метрик: {e}")
            import traceback
            logger.error(f"Трассировка ошибки: {traceback.format_exc()}")

