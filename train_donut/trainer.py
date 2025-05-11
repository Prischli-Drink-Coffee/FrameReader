import logging
import os
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np

from model import DonutModel
from dataset import DonutDataModule, JSONParseEvaluator
from utils import TrainingSpeedup, MemoryOptimizer


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
        log_interval: int = 10,
        save_interval: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        enable_distributed: bool = False,
        report_to: str = "none",  # 'tensorboard', 'wandb', 'none'
        memory_efficient: bool = True,
        evaluate_during_training: bool = True,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.01,
    ):

        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "time_per_epoch": [],
        }
        self.global_step = 0

        self.model = model
        self.data_module = data_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        self.device = device or self.model.device
        self.precision = getattr(self.model, 'precision', 'fp32')
        
        self.evaluate_during_training = evaluate_during_training
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_counter = 0
        self.best_metric = float("inf")

        self.is_distributed = enable_distributed
        if self.is_distributed:
            self.rank, self.local_rank, self.world_size = TrainingSpeedup.setup_distributed()
            self.model.model = TrainingSpeedup.wrap_model_for_distributed(self.model.model, self.local_rank)
        else:
            self.rank, self.world_size = 0, 1
        
        if memory_efficient:
            MemoryOptimizer.optimize_memory_usage()
        
        self.optimizer = self._create_optimizer()
        
        self.use_mixed_precision = self.precision in ["bf16", "fp16"]

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.scaler = TrainingSpeedup.get_mixed_precision_scaler(
            device_type=self.device,
            precision=self.precision,
            enabled=self.use_mixed_precision and self.precision == "fp16"
        )
        
        self.tracking = None
        if report_to != "none":
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
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                eps=1e-8,
            )
            logger.info("Используется 8-битный AdamW оптимизатор для экономии памяти")
        except ImportError:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                eps=1e-8,
            )
            logger.info("Используется стандартный AdamW оптимизатор")
        
        trainable_params = sum(p.numel() for group in optimizer_grouped_parameters for p in group["params"])
        logger.info(f"Оптимизатор инициализирован с {trainable_params:,} обучаемыми параметрами")
        
        return optimizer
    
    def _setup_tracking(self, report_to: str) -> None:
        try:
            if report_to == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.tracking = SummaryWriter(log_dir=str(self.output_dir / "logs"))
                logger.info(f"Инициализирован TensorBoard для логирования в {self.output_dir / 'logs'}")
            
            elif report_to == "wandb":
                import wandb
                wandb.init(
                    project="Donut-OCR", 
                    name=f"donut-training-{time.strftime('%Y%m%d-%H%M%S')}",
                    config={
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "precision": self.precision,
                        "epochs": self.num_epochs,
                        "grad_accum": self.gradient_accumulation_steps,
                        "model_type": type(self.model.model).__name__,
                    }
                )
                self.tracking = wandb
                logger.info("Инициализирован Weights & Biases для логирования")
        except ImportError as e:
            logger.warning(f"Не удалось инициализировать систему логирования ({report_to}): {e}")
            self.tracking = None
    
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
            raise ValueError("Загрузчик обучающих данных не инициализирован")

        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps

        if not hasattr(self, 'metrics') or self.metrics is None:
            self.metrics = {
                "train_loss": [],
                "val_loss": [],
                "learning_rates": [],
                "time_per_epoch": [],
            }
 
        if not hasattr(self, 'global_step') or self.global_step is None:
            self.global_step = 0

        if not hasattr(self, 'scheduler') or self.scheduler is None:
            self.scheduler = self._create_scheduler(total_steps)

        best_val_loss = float("inf")
        if "val_loss" in self.metrics and self.metrics["val_loss"]:
            best_val_loss = min(self.metrics["val_loss"])

        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            logger.info(f"Начало эпохи {epoch+1}/{self.num_epochs}")

            if hasattr(self.data_module, 'resampler_for_epoch'):
                self.data_module.resampler_for_epoch(epoch)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_loss = self._train_epoch(train_dataloader, self.scheduler, epoch+1)
            self.metrics["train_loss"].append(train_loss)
            self.metrics["learning_rates"].append(self.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start_time
            self.metrics["time_per_epoch"].append(epoch_time)

            if val_dataloader and self.evaluate_during_training:
                val_loss = self._evaluate(val_dataloader)
                self.metrics["val_loss"].append(val_loss)
                
                logger.info(
                    f"Эпоха {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"время={epoch_time/60:.1f}мин, {epoch_time/len(train_dataloader):.3f}с/шаг"
                )
                
                self._log_metrics({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/epoch_time_min": epoch_time / 60,
                }, epoch+1)

                improved = False

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = True
                    logger.info(f"Улучшение валидационной метрики потерь: {val_loss:.4f}")
                
                if improved:
                    self.early_stopping_counter = 0
                    if self.rank == 0:
                        self.model.save_pretrained(self.output_dir / "best_model")
                        logger.info(f"Сохранена новая лучшая модель с val_loss={val_loss:.4f}")
                else:
                    self.early_stopping_counter += 1
                    logger.info(f"Нет улучшения метрик. Счетчик ранней остановки: {self.early_stopping_counter}/{self.early_stopping_patience}")
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        logger.info(f"Раннее прекращение обучения после {epoch+1} эпох")
                        break
            else:
                logger.info(
                    f"Эпоха {epoch+1}: train_loss={train_loss:.4f}, "
                    f"время={epoch_time/60:.1f}мин, {epoch_time/len(train_dataloader):.3f}с/шаг"
                )
                
                self._log_metrics({
                    "train/loss": train_loss,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/epoch_time_min": epoch_time / 60,
                }, epoch+1)
            
            if self.rank == 0 and ((epoch+1) % self.save_interval == 0 or epoch+1 == self.num_epochs):
                self._save_checkpoint(epoch)
            
            if self.is_distributed and torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        if self.rank == 0:
            self.model.save_pretrained(self.output_dir / "final_model")
            logger.info(f"Сохранена финальная модель после {self.num_epochs} эпох")
        
        return self.metrics
    
    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        trainer_state = {
            'epoch': epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'metrics': self.metrics,
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
        
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        self._plot_metrics(self.metrics, epoch+1)
        
        logger.info(f"Контрольная точка сохранена для эпохи {epoch+1} в {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Union[str, Path], load_optimizer: bool = True) -> int:
        checkpoint_dir = Path(checkpoint_dir)
        trainer_state_path = checkpoint_dir / "trainer_state.pt"
        training_config_path = checkpoint_dir / "training_config.json"
        
        if not trainer_state_path.exists():
            logger.warning(f"Файл состояния тренера не найден: {trainer_state_path}")
            return 0

        self.model = DonutModel.from_pretrained(
            checkpoint_dir,
            device=self.device,
            precision=self.precision
        )
        logger.info(f"Модель загружена из {checkpoint_dir}")

        trainer_state = torch.load(trainer_state_path, map_location=self.device)
        self.metrics = trainer_state['metrics']
        self.global_step = trainer_state['global_step']
        
        if 'early_stopping_counter' in trainer_state:
            self.early_stopping_counter = trainer_state['early_stopping_counter']
        
        if 'best_metric' in trainer_state:
            self.best_metric = trainer_state['best_metric']
        
        if training_config_path.exists():
            with open(training_config_path, "r", encoding="utf-8") as f:
                training_config = json.load(f)
            
            # self.learning_rate = training_config.get('learning_rate', self.learning_rate)
            # self.weight_decay = training_config.get('weight_decay', self.weight_decay)
            logger.info(f"Загружена конфигурация обучения из {training_config_path}")
        
        self.optimizer = self._create_optimizer()
        
        if load_optimizer and 'optimizer_state' in trainer_state:
            try:
                self.optimizer.load_state_dict(trainer_state['optimizer_state'])
                logger.info("Состояние оптимизатора успешно загружено")
            except Exception as e:
                logger.warning(f"Не удалось загрузить состояние оптимизатора: {e}")
        
        if 'scheduler_state' in trainer_state and trainer_state['scheduler_state'] is not None:
            train_dataloader = self.data_module.train_dataloader()
            if train_dataloader is not None:
                total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
                self.scheduler = self._create_scheduler(total_steps)
                
                try:
                    self.scheduler.load_state_dict(trainer_state['scheduler_state'])
                    logger.info("Состояние планировщика успешно загружено")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить состояние планировщика: {e}")

        next_epoch = trainer_state['epoch'] + 1
        logger.info(f"Контрольная точка загружена, продолжение с эпохи {next_epoch}")
        
        return next_epoch
    
    def _log_metrics(self, metrics_dict: Dict[str, float], step: int) -> None:
        if self.tracking is None:
            return
        try:
            if hasattr(self.tracking, "add_scalar"):  # TensorBoard
                for name, value in metrics_dict.items():
                    self.tracking.add_scalar(name, value, step)
            elif hasattr(self.tracking, "log"):  # Weights & Biases
                self.tracking.log(metrics_dict, step=step)
        except Exception as e:
            logger.warning(f"Ошибка при логировании метрик: {e}")
    
    def _plot_metrics(self, metrics: Dict[str, List[float]], epoch: int) -> None:
        try:
            fig, axs = plt.subplots(2, 1, figsize=(10, 15))

            axs[0].plot(metrics["train_loss"], label="Обучение")
            if "val_loss" in metrics and metrics["val_loss"]:
                axs[0].plot(metrics["val_loss"], label="Валидация")
            axs[0].set_xlabel("Эпоха")
            axs[0].set_ylabel("Потери")
            axs[0].set_title("Динамика потерь")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(metrics["learning_rates"])
            axs[1].set_xlabel("Эпоха")
            axs[1].set_ylabel("Скорость обучения")
            axs[1].set_title("Динамика скорости обучения")
            axs[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"metrics_epoch_{epoch}.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Ошибка при создании графиков: {e}")
    
    def decode_prediction(self, processor, outputs, labels=None) -> Tuple[str, str]:
        pred_ids = outputs.logits[0].argmax(dim=-1)
        pred = processor.batch_decode(
            pred_ids.unsqueeze(0), 
            skip_special_tokens=True
        )[0]

        task_token = self.data_module.task_start_token
        prompt_end_token = self.data_module.prompt_end_token

        pred_cleaned = pred.replace(task_token, "").strip()

        pred_display = re.sub(r"<s_([^>]*)>", "", pred_cleaned)
        pred_display = re.sub(r"</s_[^>]*>", "", pred_display)
        pred_display = pred_display.replace("<sep/>", ", ")
        pred_display = re.sub(r"\s+", " ", pred_display).strip()
        pred_display = pred_display[:20].ljust(20, '_')

        label_display = ""
        if labels is not None:
            decoded_labels = labels[0].clone()

            pad_token_id = processor.tokenizer.pad_token_id
            decoded_labels[decoded_labels == -100] = pad_token_id
            
            label = processor.batch_decode(
                decoded_labels.unsqueeze(0), 
                skip_special_tokens=True
            )[0]

            label_cleaned = label.replace(task_token, "").replace(prompt_end_token, "")
            label_display = re.sub(r"<s_([^>]*)>", "", label_cleaned)
            label_display = re.sub(r"</s_[^>]*>", "", label_display)
            label_display = label_display.replace("<sep/>", ", ")
            label_display = re.sub(r"\s+", " ", label_display).strip()
            label_display = label_display[:20].ljust(20, '_')
        
        return pred_display, label_display

    def display_debug(
        self,
        outputs,
        processor,
        target_sequences,
        labels,
        label_display,
        pred_display
    ):
        print(f"{100*'-'}")

        # Получаем предсказания из логитов
        raw_pred_ids = outputs.logits[0].argmax(dim=-1)
        raw_pred = processor.batch_decode(
            raw_pred_ids.unsqueeze(0),
            skip_special_tokens=False
        )[0]

        original_labels = labels[0].clone()

        # Обработка меток
        label_tokens = []
        if labels is not None:
            label_token_ids = labels[0].clone()
            masked_indices = label_token_ids == -100
            label_token_ids[masked_indices] = processor.tokenizer.pad_token_id
            for token_id in label_token_ids:
                if token_id != processor.tokenizer.pad_token_id:
                    token = processor.tokenizer.convert_ids_to_tokens(token_id.item())
                    label_tokens.append(token)

        label_json = self.model.token2json(''.join(label_tokens))
        label_text = label_json['text_sequence'] 
        label_text = re.sub(r"<s_([^>]*)>", "", label_text)
        label_text = re.sub(r"</s_[^>]*>", "", label_text)
        label_text = label_text.replace("<sep/>", ", ").replace('</s>', '')
        label_text = re.sub(r"\s+", " ", label_text).strip()
        label_text = self._tokens_to_readable_text(label_text)
        label_json['text_sequence'] = self._tokens_to_readable_text(label_text.split('_'))
        label_str = json.dumps(label_json, ensure_ascii=False, indent=2)
        
        # Обработка предсказаний
        pred_tokens = []
        for token_id in raw_pred_ids:
            token = processor.tokenizer.convert_ids_to_tokens(token_id.item())
            pred_tokens.append(token)

        pred_json = self.model.token2json(raw_pred)
        pred_text = pred_json['text_sequence'] 
        pred_text = re.sub(r"<s_([^>]*)>", "", pred_text)
        pred_text = re.sub(r"</s_[^>]*>", "", pred_text)
        pred_text = pred_text.replace("<sep/>", ", ").replace('</s>', '')
        pred_text = re.sub(r"\s+", " ", pred_text).strip()
        pred_text = self._tokens_to_readable_text(pred_text)
        pred_json['text_sequence'] = pred_text
        pred_str = json.dumps(pred_json, ensure_ascii=False, indent=2)

        print(f"Метка в формате строки:\n {label_display.replace('_', '')}")
        print(f"Метка в формате тензора:\n {original_labels}")
        print(f"Метка в формате листа токенов:\n {label_tokens}")
        print(f"Метка в формате JSON:\n {label_json}")
        print(f"Предсказание в формате строки:\n {pred_display.replace('_', '')}")
        print(f"Предсказание в формате тензора:\n {raw_pred_ids}")
        print(f"Предсказание в формате листа токенов:\n {pred_tokens}")
        print(f"Предсказание в формате JSON:\n {pred_json}")
        
        print(f"{100*'-'}")
    
    def _tokens_to_readable_text(self, tokens):
        if isinstance(tokens, str):
            text = tokens
            if text.startswith('_'):
                text = text[1:]
            text = text.replace('_', ' ')
            return text.strip() 
        text = ""
        for i, token in enumerate(tokens):
            if '_' in token:
                if i == 0 and token.startswith('_'):
                    token = token[1:]
                token = token.replace('_', ' ')
            text += token
        return re.sub(r'\s+', ' ', text).strip()
    
    def _train_epoch(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int
    ) -> float:

        self.model.train()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Обучение эпохи {epoch}", 
            disable=self.rank != 0,
            leave=False, 
            position=0
        )

        processor = self.data_module.processor
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):

            pixel_values, labels, target_sequences = batch

            batch_dict = {
                "pixel_values": pixel_values.to(self.device),
                "labels": labels.to(self.device)
            }
            
            loss, outputs = self._training_step(batch_dict)
            (loss / self.gradient_accumulation_steps).backward()
            
            pred_display, label_display = self.decode_prediction(processor, outputs, labels)

            if step % 1000 == 0:
                self.display_debug(outputs, processor, target_sequences, labels, label_display, pred_display)
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_model_parameters(), 
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item()
            total_steps += 1

            if step % self.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / total_steps:.4f}",
                    "lbl": label_display,
                    "pred": pred_display,
                    "lr": f"{lr:.2e}",
                    "step": f"{self.global_step}"
                })

        if self.is_distributed and torch.distributed.is_initialized():
            dist_loss = torch.tensor([total_loss / total_steps], device=self.device)
            torch.distributed.all_reduce(dist_loss, op=torch.distributed.ReduceOp.SUM)
            train_loss = (dist_loss / self.world_size).item()
        else:
            train_loss = total_loss / total_steps
        
        return train_loss
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Any]:
        if self.use_mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16):
                outputs = self.model.forward(
                    batch["pixel_values"], 
                    batch["labels"]
                )
                loss = outputs.loss
        else:
            outputs = self.model.forward(
                batch["pixel_values"], 
                batch["labels"]
            )
            loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Обнаружено недопустимое значение loss: {loss}")
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        
        return loss, outputs
    
    def _evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        json_evaluator = JSONParseEvaluator()
        total_json_accuracy = 0
        total_predictions = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc="Валидация", 
            disable=self.rank != 0,
            leave=False, 
            position=0
        )
        
        processor = self.data_module.processor
        
        with torch.no_grad():
            for batch in progress_bar:
                pixel_values, labels, prompt_end_index, target_sequences = batch

                batch_dict = {
                    "pixel_values": pixel_values.to(self.device),
                    "labels": labels.to(self.device)
                }
                
                if self.use_mixed_precision:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.precision == "fp16" else torch.bfloat16):
                        outputs = self.model.forward(
                            batch_dict["pixel_values"], 
                            batch_dict["labels"]
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model.forward(
                        batch_dict["pixel_values"], 
                        batch_dict["labels"]
                    )
                    loss = outputs.loss
                
                total_loss += loss.item()
                
                generated_outputs = self.model.generate(
                    batch_dict["pixel_values"],
                    max_length=self.model.max_length,
                    return_json=True
                )

                for i, (prediction, target) in enumerate(zip(generated_outputs, target_sequences)):
                    prediction_text = json.dumps(prediction) if isinstance(prediction, dict) else prediction
                    target_text = target.replace(processor.tokenizer.eos_token, "").strip()

                    try:
                        if isinstance(target, str) and (target.startswith("{") or target.startswith("[")):
                            target_json = json.loads(target)
                        else:
                            target_json = self.model.token2json(target)
                        
                        json_accuracy = json_evaluator.cal_acc(prediction, target_json)
                        total_json_accuracy += json_accuracy
                        total_predictions += 1
                    except Exception as e:
                        logger.warning(f"Ошибка при оценке JSON: {e}")
                
                total_steps += 1

                avg_json_acc = total_json_accuracy / total_predictions if total_predictions > 0 else 0
                pred_display, label_display = self.decode_prediction(processor, outputs, labels)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / total_steps:.4f}",
                    "json_acc": f"{avg_json_acc:.4f}",
                    "lbl": label_display,
                    "pred": pred_display
                })
            
            avg_loss = total_loss / total_steps
            avg_json_accuracy = total_json_accuracy / total_predictions if total_predictions > 0 else 0

            if self.is_distributed and torch.distributed.is_initialized():
                dist_metrics = torch.tensor([avg_loss, avg_json_accuracy], device=self.device)
                torch.distributed.all_reduce(dist_metrics, op=torch.distributed.ReduceOp.SUM)
                avg_loss, avg_json_accuracy = (dist_metrics / self.world_size).tolist()

            self._log_metrics({
                "val/json_accuracy": avg_json_accuracy
            }, self.global_step)
            
            logger.info(f"Оценка валидации: loss={avg_loss:.4f}, json_accuracy={avg_json_accuracy:.4f}")
            
            return avg_loss
