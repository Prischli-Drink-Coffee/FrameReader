import gc
import sys
import logging
import os
import time
import json
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

from model import DonutModel
from dataset import DonutDataModule, JSONParseEvaluator
from utils import TrainingSpeedup, MemoryOptimizer, MetricsCalculator, MetricsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DonutTrainer:
    
    def __init__(
        self,
        model: DonutModel,
        data_module: DonutDataModule,
        output_dir: Union[str, Path],
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_ratio: float = 0.05,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_interval: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        enable_distributed: bool = False,
        report_to: str = "none",
        memory_efficient: bool = True,
        evaluate_during_training: bool = True,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.01,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.data_module = data_module
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.evaluate_during_training = evaluate_during_training
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.save_interval = save_interval
        
        self.device = device or self.model.device
        self.precision = getattr(self.model, 'precision', 'fp32')
        
        self.metrics_evaluator = MetricsCalculator()
        self.metrics_visualizer = MetricsVisualizer(
            output_dir
        )
        
        self.global_step = 0
        self.early_stopping_counter = 0
        self.best_metric = float("inf")

        if enable_distributed:
            self.local_rank, self.world_size, device_id = TrainingSpeedup.setup_distributed()
            self.model = TrainingSpeedup.wrap_model_for_distributed(self.model, self.local_rank)
            self.is_main_process = self.local_rank == 0
            self.rank = self.local_rank
            logger.info(f"Распределенное обучение инициализировано: ранг={self.local_rank}, всего={self.world_size}")
        else:
            self.local_rank = 0
            self.world_size = 1
            self.is_main_process = True
            self.rank = 0
            logger.info("Используется однопроцессорное обучение")
        
        if memory_efficient:
            MemoryOptimizer.optimize_memory_usage()
        
        self.optimizer = self._create_optimizer()
        
        self.use_mixed_precision = self.precision in ["bf16", "fp16"]
        
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.scaler = TrainingSpeedup.get_mixed_precision_scaler(
            device_type=str(self.device),
            precision=self.precision,
            enabled=self.use_mixed_precision and self.precision == "fp16"
        )
        
        self.tracking = None
        if report_to != "none" and self.is_main_process:
            self._setup_tracking(report_to)
        
        logger.info(f"Инициализирован тренер Donut (точность={self.precision})")
        logger.info(f"Шаги накопления градиента: {self.gradient_accumulation_steps}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {                    
                "params": [
                    p for n, p in self.model.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        try:
            from torch.optim.adamw import AdamW
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        except ImportError:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        trainable_params = sum(p.numel() for group in optimizer_grouped_parameters for p in group["params"])
        logger.info(f"Оптимизатор инициализирован с {trainable_params:,} обучаемыми параметрами")
        
        return optimizer
    
    def _setup_tracking(self, report_to: str) -> None:
        try:
            if report_to == "tensorboard":
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self.tracking = SummaryWriter(log_dir=self.output_dir / "tensorboard_logs")
                    logger.info("Настроен логгер TensorBoard")
                except ImportError:
                    logger.warning("TensorBoard не установлен, логгирование отключено")
            elif report_to == "wandb":
                try:
                    import wandb
                    wandb.init(project="donut-ocr", dir=str(self.output_dir))
                    self.tracking = wandb
                    logger.info("Настроен логгер Weights & Biases")
                except ImportError:
                    logger.warning("Wandb не установлен, логгирование отключено")
        except ImportError as e:
            logger.warning(f"Не удалось настроить систему логгирования: {e}")
    
    def _create_scheduler(self, num_training_steps: int):
        warmup_steps = int(num_training_steps * self.warmup_ratio)
        logger.info(f"Используется косинусный планировщик с {warmup_steps} шагами разогрева из {num_training_steps} общих шагов")
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    def train(self, start_epoch: int = 0) -> Dict[str, List[float]]:
        train_dataloader = self.data_module.train_dataloader()
        val_dataloader = self.data_module.val_dataloader()
        
        if train_dataloader is None:
            raise ValueError("Тренировочный загрузчик данных не настроен")

        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps

        if not hasattr(self, 'scheduler') or self.scheduler is None:
            self.scheduler = self._create_scheduler(total_steps)

        best_val_loss = float("inf")
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            
            if self.data_module and hasattr(self.data_module, 'resampler_for_epoch'):
                self.data_module.resampler_for_epoch(epoch)
            
            train_loss = self._train_epoch(train_dataloader, self.scheduler, epoch, start_epoch)
            
            val_loss = 0.0
            val_metrics = {}
            
            if val_dataloader and self.evaluate_during_training:
                val_loss, val_metrics = self._evaluate(val_dataloader)
                
                current_epoch = epoch + 1
                for metric_name, metric_value in val_metrics.items():
                    logger.info(f"Эпоха {current_epoch} Валидация {metric_name}: {metric_value:.4f}")
                
                metrics_dict = {"val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
                self.metrics_visualizer.update_metrics(
                    metrics_dict, 
                    step=self.global_step, 
                    epoch=current_epoch,
                    is_val=True
                )
                
                if val_loss < best_val_loss - self.early_stopping_threshold:
                    best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    
                    if self.is_main_process:
                        best_model_dir = self.output_dir / "best_model"
                        self.model.save_pretrained(best_model_dir)
                        logger.info(f"Новая лучшая модель сохранена, val_loss: {val_loss:.4f}")
                elif val_loss > best_val_loss - self.early_stopping_threshold:
                    self.early_stopping_counter += 1
                    logger.info(f"Раннее останов: {self.early_stopping_counter}/{self.early_stopping_patience}, "
                              f"лучшая val_loss: {best_val_loss:.4f}, текущая: {val_loss:.4f}")
                    
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Раннее останов после {epoch + 1} эпох")
                        break
            
            if self.is_main_process and (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch)
            
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Эпоха {epoch + 1}/{self.num_epochs} завершена за {epoch_duration:.2f} сек")
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
                torch.cuda.empty_cache()
                gc.collect()
        
        if self.is_main_process:
            final_model_path = self.output_dir / "final_model"
            self.model.save_pretrained(final_model_path)
            self.metrics_visualizer.visualize_metrics(self.global_step, self.num_epochs)
            self.metrics_visualizer.save_metrics(self.global_step)
            total_duration = time.time() - start_time
            logger.info(f"Обучение завершено за {total_duration:.2f} сек")
            logger.info(f"Средняя скорость: {total_steps * self.gradient_accumulation_steps / total_duration:.2f} шагов/сек")
        
        return self.metrics_visualizer.metrics_history
    
    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        
        trainer_state = {
            'epoch': epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'global_step': self.global_step,
            'early_stopping_counter': self.early_stopping_counter,
            'best_metric': self.best_metric
        }
        torch.save(trainer_state, checkpoint_dir / "trainer_state.pt")
        
        config_dict = {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'warmup_ratio': self.warmup_ratio,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_grad_norm': self.max_grad_norm,
            'precision': self.precision,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold
        }
        
        with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Контрольная точка сохранена для эпохи {epoch+1} в {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Union[str, Path], load_optimizer: bool = True) -> int:
        checkpoint_dir = Path(checkpoint_dir)
        trainer_state_path = checkpoint_dir / "trainer_state.pt"
        training_config_path = checkpoint_dir / "training_config.json"
        
        if not trainer_state_path.exists():
            raise FileNotFoundError(f"Файл состояния тренера не найден: {trainer_state_path}")

        self.model = DonutModel.from_pretrained(
            checkpoint_dir,
            device=self.device
        )
        logger.info(f"Модель загружена из {checkpoint_dir}")

        trainer_state = torch.load(trainer_state_path, map_location=self.device)
        self.global_step = trainer_state['global_step']
        
        if 'early_stopping_counter' in trainer_state:
            self.early_stopping_counter = trainer_state['early_stopping_counter']
        
        if 'best_metric' in trainer_state:
            self.best_metric = trainer_state['best_metric']
        
        if training_config_path.exists():
            with open(training_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

                exclude_params = {'num_epochs', 'learning_rate', 'weight_decay', 'warmup_ratio', 
                                'gradient_accumulation_steps', 'max_grad_norm', 'early_stopping_patience', 
                                'early_stopping_threshold'}
                
                for k, v in config.items():
                    if hasattr(self, k) and k not in exclude_params:
                        setattr(self, k, v)
                        
            logger.info(f"Конфигурация обучения загружена из {training_config_path}")
        
        self.optimizer = self._create_optimizer()
        
        if load_optimizer and 'optimizer_state' in trainer_state:
            try:
                self.optimizer.load_state_dict(trainer_state['optimizer_state'])
                logger.info("Состояние оптимизатора загружено")
            except Exception as e:
                logger.warning(f"Не удалось загрузить состояние оптимизатора: {e}")
        
        if 'scheduler_state' in trainer_state and trainer_state['scheduler_state'] is not None:
            total_steps = len(self.data_module.train_dataloader()) * self.num_epochs // self.gradient_accumulation_steps
            self.scheduler = self._create_scheduler(total_steps)
            self.scheduler.load_state_dict(trainer_state['scheduler_state'])
            logger.info("Состояние планировщика загружено")

        next_epoch = trainer_state['epoch'] + 1
        logger.info(f"Контрольная точка загружена, продолжение с эпохи {next_epoch}")
        
        return next_epoch
    
    def _log_metrics(self, metrics_dict: Dict[str, float], step: int) -> None:
        """Логирование метрик в систему отслеживания."""
        if not self.is_main_process:
            return
            
        if self.tracking is None:
            return
        
        try:
            if hasattr(self.tracking, 'add_scalar'):  # TensorBoard
                for key, value in metrics_dict.items():
                    self.tracking.add_scalar(key, value, step)
            elif hasattr(self.tracking, 'log'):  # WandB
                self.tracking.log(metrics_dict, step=step)
        except Exception as e:
            logger.warning(f"Ошибка при логировании метрик: {e}")
    
    def decode_prediction(self, processor, outputs, labels=None) -> Tuple[str, str]:
        """Декодирует предсказания модели."""
        pred_ids = outputs.logits[0].argmax(dim=-1)
        pred = processor.batch_decode(
            pred_ids.unsqueeze(0), 
            skip_special_tokens=True
        )[0]

        task_token = self.data_module.task_start_token
        prompt_end_token = self.data_module.prompt_end_token

        pred_cleaned = pred.replace(task_token, "").replace(prompt_end_token, "").strip()
        pred_display = re.sub(r"<s_([^>]*)>", "", pred_cleaned)
        pred_display = re.sub(r"</s_[^>]*>", "", pred_display)
        pred_display = pred_display.replace("<sep/>", ", ")
        pred_display = re.sub(r"\s+", " ", pred_display).strip()

        label_display = ""
        if labels is not None:
            label_ids = labels[0].clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            
            raw_label_text = processor.batch_decode(
                label_ids.unsqueeze(0), 
                skip_special_tokens=False
            )[0]
            
            label_text = processor.batch_decode(
                label_ids.unsqueeze(0), 
                skip_special_tokens=True
            )[0]
            
            label_cleaned = label_text.replace(task_token, "").replace(prompt_end_token, "").strip()
            label_display = re.sub(r"<s_([^>]*)>", "", label_cleaned)
            label_display = re.sub(r"</s_[^>]*>", "", label_display)
            label_display = label_display.replace("<sep/>", ", ")
            label_display = re.sub(r"\s+", " ", label_display).strip()
        
        return pred_display, label_display

    def display_debug(
        self,
        outputs,
        processor,
        labels,
        label_display,
        pred_display,
        prompt_end_indices
    ):
        """Отображает отладочную информацию о предсказаниях."""
        logger.info(f"{100*'-'}")

        raw_pred_ids = outputs.logits[0].argmax(dim=-1)
        raw_pred = processor.batch_decode(
            raw_pred_ids.unsqueeze(0),
            skip_special_tokens=False
        )[0]

        original_labels = labels[0].clone()
        
        label_tokens = []
        if labels is not None:
            label_mask = original_labels != -100
            label_ids = original_labels[label_mask]
            
            for idx in label_ids:
                token = processor.tokenizer._convert_id_to_token(idx.item())
                label_tokens.append(token)
            
            tokens_str = ' '.join(label_tokens)
            
            labels_for_decoding = original_labels.clone()
            labels_for_decoding[labels_for_decoding == -100] = processor.tokenizer.pad_token_id
            raw_label_text = processor.batch_decode(
                labels_for_decoding.unsqueeze(0), 
                skip_special_tokens=False
            )[0]

        pred_tokens = []
        for token_id in raw_pred_ids:
            token = processor.tokenizer._convert_id_to_token(token_id.item())
            pred_tokens.append(token)
        
        target_tokens = []
        if labels is not None and prompt_end_indices is not None:
            start_idx = prompt_end_indices[0].item() if isinstance(prompt_end_indices, torch.Tensor) else prompt_end_indices
            target_ids = original_labels[start_idx:]
            for idx in target_ids:
                if idx != -100:
                    token = processor.tokenizer._convert_id_to_token(idx.item())
                    target_tokens.append(token)
        
        try:
            for index, t in enumerate(label_tokens):
                if index == 0:
                    t = t.replace('▁', '')
                else:
                    t = t.replace('▁', ' ')
                label_tokens[index] = t
            label_json = self.model.token2json(''.join(label_tokens))
        except Exception as e:
            label_json = {"error": f"Не удалось преобразовать GT в JSON: {str(e)}"}
            
        try:
            pred_json = self.model.token2json(raw_pred)
        except Exception as e:
            pred_json = {"error": f"Не удалось преобразовать предсказание в JSON: {str(e)}"}
        
        logger.info("Вывод в text:")
        logger.info(f"GT Text: {label_display}")
        logger.info(f"PR Text: {pred_display}")
        
        logger.info("Вывод в tokens:")
        logger.info(f"Pred Tokens: {' '.join(pred_tokens[:20])}...")
        
        logger.info("Вывод в JSON:")
        logger.info(f"GT JSON: {json.dumps(label_json, ensure_ascii=False)[:100]}...")
        logger.info(f"Pred JSON: {json.dumps(pred_json, ensure_ascii=False)[:100]}...")
    
    def _tokens_to_readable_text(self, tokens):
        """Преобразует токены в читаемый текст."""
        if isinstance(tokens, str):
            return tokens
            
        result = []
        for token in tokens:
            if token and token != '_':
                result.append(token)
        
        return ' '.join(result)
    
    def _train_epoch(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        start_epoch: int = 0
    ) -> float:
        """Обучение на одной эпохе."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        progress_bar = tqdm(
            total=len(dataloader),
            disable=not self.is_main_process,
            desc=f"Эпоха {epoch+1}/{self.num_epochs}",
            leave=True
        )
        
        batch_predictions = []
        batch_targets = []
        
        for step, batch in enumerate(dataloader):
            current_epoch = epoch + step/len(dataloader)

            # print(batch)
            
            loss, _ = self._training_step(batch)
            
            total_loss += loss.item()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.global_step += 1
                curr_lr = scheduler.get_last_lr()[0]
                
                basic_train_metrics = {
                    "train_loss": loss.item(),
                    "learning_rate": curr_lr,
                }
                
                pixel_values, labels, _ = batch

                self.model.eval()
                
                with torch.no_grad():
                    outputs = self.model.forward(pixel_values=pixel_values, labels=labels)

                    pred_clean, label_clean = self.decode_prediction(
                        self.model.processor,
                        outputs=outputs,
                        labels=labels
                    )
                    
                batch_predictions.append(pred_clean)
                batch_targets.append(label_clean)
                
                metrics = self.metrics_evaluator.evaluate_predictions(
                    batch_predictions, batch_targets
                )
                
                full_train_metrics = {
                    **basic_train_metrics,
                    "train_cer": metrics["cer"],
                    "train_wer": metrics["wer"],
                    "train_rouge-1": metrics["rouge-1"],
                    "train_rouge-2": metrics["rouge-2"],
                    "train_rouge-l": metrics["rouge-l"],
                }

                if epoch != start_epoch:
                    self.metrics_visualizer.update_metrics(
                        full_train_metrics, 
                        step=self.global_step,
                        epoch=current_epoch,
                        is_val=False
                    )

                batch_predictions = []
                batch_targets = []
                
                self.model.train()
            
            if self.is_main_process:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.7f}"
                })
                progress_bar.update(1)
            
        progress_bar.close()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        
        logger.info(f"Эпоха {epoch+1} средняя потеря: {avg_loss:.4f}, время: {epoch_time:.2f} сек")
        
        return avg_loss
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        """Выполняет один шаг обучения."""
        pixel_values, labels, _ = batch
        
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        if self.use_mixed_precision and self.precision == "fp16" and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model.forward(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.get_model_parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                
        elif self.use_mixed_precision and self.precision == "bf16":
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model.forward(pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.get_model_parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                
        else:
            outputs = self.model.forward(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.get_model_parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
        
        return loss * self.gradient_accumulation_steps, outputs
    
    def _evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, Dict[str, float]]:
        """Оценивает модель на валидационных данных."""
        self.model.eval()
        val_loss = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Валидация", disable=not self.is_main_process)):
                pixel_values, labels, prompt_end_indices, target_sequences = batch
                
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model.forward(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

                pred_clean, label_clean = self.decode_prediction(
                    self.model.processor, 
                    outputs, 
                    labels
                )
                
                all_predictions.append(pred_clean)
                all_targets.append(label_clean)
                
                if self.is_main_process:
                    self.display_debug(
                        outputs,
                        self.model.processor,
                        labels,
                        label_clean,
                        pred_clean,
                        prompt_end_indices
                    )

        avg_val_loss = val_loss / len(dataloader)
        
        metrics = self.metrics_evaluator.evaluate_predictions(
            all_predictions, all_targets
        )
        
        return avg_val_loss, metrics
