import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

from model import TrOCRModel
from dataset import TrOCRDataModule
from utils import TrainingSpeedup, MemoryOptimizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    
    def __init__(
        self,
        model: TrOCRModel,
        data_module: TrOCRDataModule,
        output_dir: Union[str, Path],
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_ratio: float = 0.05,
        gradient_accumulation_steps: int = 32,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        save_interval: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        enable_distributed: bool = False,
        report_to: str = "none",  # 'tensorboard', 'wandb', 'none'
        memory_efficient: bool = True,
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

        self.is_distributed = enable_distributed
        if self.is_distributed:
            self.rank, self.local_rank, self.world_size = TrainingSpeedup.setup_distributed()
            self.model = TrainingSpeedup.wrap_model_for_distributed(self.model, self.local_rank)
        else:
            self.rank, self.world_size = 0, 1
        
        if memory_efficient:
            MemoryOptimizer.optimize_memory_usage()
        
        self.optimizer = self._create_optimizer()

        self.use_mixed_precision = self.precision in ["bf16", "fp16"]
        self.scaler = TrainingSpeedup.get_mixed_precision_scaler(
            device_type=self.device.type,
            precision=self.precision,
            enabled=self.use_mixed_precision and self.precision == "fp16"
        )
        
        self.tracking = None
        if report_to != "none":
            self._setup_tracking(report_to)
        
        logger.info(f"Инициализирован оптимизированный тренер (точность={self.precision})")
        logger.info(f"Шаги накопления градиента: {self.gradient_accumulation_steps}")

        self._check_torch_compatibility()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        if hasattr(self.model, "model"):
            parameters = self.model.model.named_parameters()
        else:
            parameters = self.model.named_parameters()
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in parameters 
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in parameters 
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
                    project="RusTitW-OCR", 
                    name=f"trocr-training-{time.strftime('%Y%m%d-%H%M%S')}",
                    config={
                        "lr": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "precision": self.precision,
                        "epochs": self.num_epochs,
                        "grad_accum": self.gradient_accumulation_steps,
                    }
                )
                self.tracking = wandb
                logger.info("Инициализирован Weights & Biases для логирования")
        except ImportError as e:
            logger.warning(f"Не удалось инициализировать систему логирования ({report_to}): {e}")
            self.tracking = None
    
    def _get_scaler(self) -> Optional[torch.cuda.amp.GradScaler]:
        if not self.use_mixed_precision or self.precision == "bf16":
            return None
        if hasattr(torch.cuda.amp, 'GradScaler'):
            try:
                return torch.cuda.amp.GradScaler()
            except Exception as e:
                logger.warning(f"Не удалось создать GradScaler: {e}")
                return None
        return None
    
    def _check_torch_compatibility(self) -> None:
        torch_version = torch.__version__
        logger.info(f"Версия PyTorch: {torch_version}")
        self.has_new_autocast_api = False
        try:
            from packaging import version
            if version.parse(torch_version) >= version.parse("1.10.0"):
                self.has_new_autocast_api = True
        except ImportError:
            if int(torch_version.split('.')[0]) >= 1 and int(torch_version.split('.')[1]) >= 10:
                self.has_new_autocast_api = True
        
        logger.info(f"Использование нового API autocast: {self.has_new_autocast_api}")
    
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
        
        if not hasattr(self, 'metrics'):
            self.metrics = {
                "train_loss": [],
                "val_loss": [],
                "learning_rates": [],
                "time_per_epoch": [],
            }
        
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        
        if not hasattr(self, 'scheduler'):
            self.scheduler = self._create_scheduler(total_steps)
        
        best_val_loss = float("inf")
        if "val_loss" in self.metrics and self.metrics["val_loss"]:
            best_val_loss = min(self.metrics["val_loss"])

        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            logger.info(f"Начало эпохи {epoch+1}/{self.num_epochs}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            train_loss = self._train_epoch(train_dataloader, self.scheduler, epoch+1)
            self.metrics["train_loss"].append(train_loss)
            self.metrics["learning_rates"].append(self.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start_time
            self.metrics["time_per_epoch"].append(epoch_time)
            
            if val_dataloader:
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
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.rank == 0:
                        self.model.save_pretrained(self.output_dir / "best_model")
                        logger.info(f"Сохранена новая лучшая модель с val_loss={val_loss:.4f}")
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
                checkpoint_dir = self.output_dir / f"checkpoint-{epoch+1}"
                self.model.save_pretrained(checkpoint_dir)
                
                trainer_state = {
                    'epoch': epoch,
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                    'metrics': self.metrics,
                    'global_step': self.global_step
                }
                torch.save(trainer_state, checkpoint_dir / "trainer_state.pt")
                
                logger.info(f"Контрольная точка сохранена для эпохи {epoch+1} в {checkpoint_dir}")
                
                with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(self.metrics, f, indent=2)
                
                self._plot_metrics(self.metrics, epoch+1)
                
            if self.is_distributed and torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        return self.metrics
    
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
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))

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

    def decode_label(self, model, batch, processor, outputs) -> Tuple[str, str]:
        decoded_labels = batch["labels"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        decoded_labels[decoded_labels == -100] = pad_token_id          
        label = processor.batch_decode(
            decoded_labels[:1], 
            skip_special_tokens=True
        )[0]
        pred_ids = outputs.logits[0].argmax(dim=-1)
        pred = processor.batch_decode(
            pred_ids.unsqueeze(0), 
            skip_special_tokens=True
        )[0]
        label = label[:10].ljust(10, '_')
        pred = pred[:10].ljust(10, '_')
        return label, pred
    
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
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.clone()

            loss, outputs = self._training_step(batch, processor)
            label, pred = self.decode_label(self.model, batch, processor, outputs)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()
                scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_steps += 1

            if step % self.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "lbl": label,
                    "pred": pred,
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "avg_loss": f"{total_loss / total_steps:.4f}",
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
    
    def _training_step(self, batch: Dict[str, torch.Tensor], processor) -> torch.Tensor:

        if "pixel_values" not in batch:
            raise ValueError("Ключ 'pixel_values' отсутствует в батче")
        if "labels" not in batch:
            raise ValueError("Ключ 'labels' отсутствует в батче")
        
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        if pixel_values.dim() > 4:
            logger.warning(f"Неожиданная форма pixel_values: {pixel_values.shape}. Исправление...")
            pixel_values = pixel_values.view(-1, pixel_values.size(-3), pixel_values.size(-2), pixel_values.size(-1))
            batch["pixel_values"] = pixel_values
            logger.info(f"Новая форма pixel_values: {batch['pixel_values'].shape}")
        
        if torch.isnan(pixel_values).any():
            logger.warning("Обнаружены NaN значения в pixel_values")
            batch["pixel_values"] = torch.nan_to_num(pixel_values)
        
        if torch.isnan(labels).any():
            logger.warning("Обнаружены NaN значения в labels")
            batch["labels"] = torch.nan_to_num(labels, nan=-100)
        
        if hasattr(processor.tokenizer, "vocab_size"):
            max_valid_index = processor.tokenizer.vocab_size - 1
            invalid_mask = (labels != -100) & (labels > max_valid_index)
            if invalid_mask.any():
                invalid_indices = torch.nonzero(invalid_mask).tolist()
                logger.warning(f"Обнаружены недопустимые индексы токенов, превышающие vocab_size={max_valid_index}: "
                            f"{labels[invalid_mask].tolist()} на позициях {invalid_indices}")
                labels = labels.clone()
                labels[invalid_mask] = -100
                batch["labels"] = labels
        
        if self.use_mixed_precision:
            if self.precision == "bf16":
                loss, outputs = self._mixed_precision_step_bf16(batch, processor)
            else:
                loss, outputs = self._mixed_precision_step_fp16(batch, processor)
        else:
            loss, outputs = self._full_precision_step(batch, processor)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Обнаружено недопустимое значение loss: {loss}")
            loss = torch.tensor(0.1, device=loss.device, requires_grad=True)
        
        (loss / self.gradient_accumulation_steps).backward()
        
        return loss, outputs
    
    def _mixed_precision_step_bf16(self, batch: Dict[str, torch.Tensor], processor) -> torch.Tensor:

        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]
            if pixel_values.dim() == 5 and pixel_values.size(0) == 1 and pixel_values.size(1) == 1:
                batch["pixel_values"] = pixel_values.squeeze(1)
            elif pixel_values.dim() > 4:
                logger.warning(f"Неожиданная форма pixel_values: {pixel_values.shape}, исправление...")
                batch["pixel_values"] = pixel_values.view(-1, 
                                                        pixel_values.size(-3), 
                                                        pixel_values.size(-2), 
                                                        pixel_values.size(-1))
                logger.info(f"Новая форма: {batch['pixel_values'].shape}")
        
        if self.has_new_autocast_api:
            with torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type=self.device.type):
                outputs = self.model.forward(
                    batch["pixel_values"], 
                    batch["labels"]
                )
                loss = outputs.loss
        else:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.forward(
                    batch["pixel_values"], 
                    batch["labels"]
                )
                loss = outputs.loss
        
        return loss, outputs
    
    def _mixed_precision_step_fp16(self, batch: Dict[str, torch.Tensor], processor) -> torch.Tensor:

        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]
            if pixel_values.dim() == 5 and pixel_values.size(0) == 1 and pixel_values.size(1) == 1:
                batch["pixel_values"] = pixel_values.squeeze(1)
            elif pixel_values.dim() > 4:
                # logger.warning(f"Неожиданная форма pixel_values: {pixel_values.shape}, исправление...")
                batch["pixel_values"] = pixel_values.view(-1, 
                                                        pixel_values.size(-3), 
                                                        pixel_values.size(-2), 
                                                        pixel_values.size(-1))
        if self.has_new_autocast_api:
            with torch.amp.autocast(enabled=True, device_type=self.device.type, dtype=torch.float16):
                outputs = self.model.forward(
                    batch["pixel_values"], 
                    batch["labels"]
                )
                loss = outputs.loss
        else:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.forward(
                    batch["pixel_values"], 
                    batch["labels"]
                )
                loss = outputs.loss
        
        return loss, outputs
    
    def _full_precision_step(self, batch: Dict[str, torch.Tensor], processor) -> torch.Tensor:

        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]
            if pixel_values.dim() == 5 and pixel_values.size(0) == 1 and pixel_values.size(1) == 1:
                batch["pixel_values"] = pixel_values.squeeze(1)
            elif pixel_values.dim() > 4:
                # logger.warning(f"Неожиданная форма pixel_values: {pixel_values.shape}, исправление...")
                batch["pixel_values"] = pixel_values.view(-1, 
                                                        pixel_values.size(-3), 
                                                        pixel_values.size(-2), 
                                                        pixel_values.size(-1))
        outputs = self.model.forward(
            batch["pixel_values"], 
            batch["labels"]
        )
        loss = outputs.loss
        
        return loss, outputs
    
    def _optimizer_step(self) -> None:
        if self.precision == "fp16" and self.scaler:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.get_model_parameters() if p.requires_grad], 
            self.max_grad_norm
        )
        
        if self.precision == "fp16" and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def _evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        progress_bar = tqdm(
            dataloader, 
            desc="Оценка", 
            disable=self.rank != 0,
            leave=False, 
            position=0
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                if self.use_mixed_precision:
                    if self.has_new_autocast_api:
                        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
                        with torch.amp.autocast(enabled=True, device_type=self.device.type, dtype=dtype):
                            outputs = self.model.forward(
                                batch["pixel_values"], 
                                batch["labels"]
                            )
                    else:
                        with torch.cuda.amp.autocast(enabled=True):
                            outputs = self.model.forward(
                                batch["pixel_values"], 
                                batch["labels"]
                            )
                else:
                    outputs = self.model.forward(
                        batch["pixel_values"], 
                        batch["labels"]
                    )
                
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
                progress_bar.set_postfix({"loss": f"{total_loss / total_steps:.4f}"})
        
        if self.is_distributed and torch.distributed.is_initialized():
            dist_loss = torch.tensor([total_loss / total_steps], device=self.device)
            torch.distributed.all_reduce(dist_loss, op=torch.distributed.ReduceOp.SUM)
            val_loss = (dist_loss / self.world_size).item()
        else:
            val_loss = total_loss / total_steps
        
        return val_loss