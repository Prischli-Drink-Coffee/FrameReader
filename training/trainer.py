from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from core.base import BaseOCRModel
from core.config import TrainingConfig, DataConfig
from training.metrics import MetricsCalculator
from training.visualization import TrainingVisualizer
from training.enhanced_progress import EnhancedProgressDisplay, MetricsCollector

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(self, model: BaseOCRModel, train_config: TrainingConfig, data_config: DataConfig, 
                 output_dir: Path, inference_displayer=None):
        self.model = model
        self.train_config = train_config
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_displayer = inference_displayer
        
        self.device = model.device
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        self._setup_training()
        
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = TrainingVisualizer(self.output_dir, train_config)
        
        self.progress_display = EnhancedProgressDisplay(train_config.num_epochs)
        self.metrics_collector = MetricsCollector()
    
    def _setup_training(self) -> None:
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = AdamW(
            trainable_params, 
            lr=self.train_config.learning_rate, 
            weight_decay=self.train_config.weight_decay
        )
        
        if self.train_config.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        self._set_seed(self.train_config.seed)
        logger.info(f"Training setup complete. Device: {self.device}")
    
    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    def _setup_scheduler(self, total_steps: int) -> None:
        warmup_steps = int(total_steps * self.train_config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        total_steps = len(train_dataloader) * self.train_config.num_epochs
        self._setup_scheduler(total_steps)
        
        history = {'train_loss': [], 'eval_loss': [], 'learning_rates': []}
        
        self.progress_display.start_training()
        self.progress_display.log_info(f"Total steps: {total_steps}")
        self.progress_display.log_info(f"Output directory: {self.output_dir}")
        
        if self.inference_displayer:
            self.progress_display.log_info("Real-time inference engine is active")
        else:
            self.progress_display.log_warning("Real-time inference engine is not available")
        
        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            self.progress_display.start_epoch(epoch, len(train_dataloader))
            
            train_metrics = self._train_epoch(train_dataloader)
            
            epoch_time = time.time() - epoch_start_time
            
            history['train_loss'].append(train_metrics['loss'])
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            eval_metrics = None
            if eval_dataloader and epoch % self.train_config.eval_interval == 0:
                eval_start_time = time.time()
                eval_metrics = self._eval_epoch(eval_dataloader)
                eval_time = time.time() - eval_start_time
                
                history['eval_loss'].append(eval_metrics['loss'])
                
                if eval_metrics['loss'] < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self._save_checkpoint('best_model')
                    self.progress_display.log_info(f"New best model saved! Validation loss: {eval_metrics['loss']:.4f}")
            
            if epoch % self.train_config.save_interval == 0:
                self._save_checkpoint(f'checkpoint-epoch-{epoch}')
            
            try:
                self.visualizer.update_training_progress(epoch, train_metrics, history)
            except Exception as e:
                self.progress_display.log_warning(f"Visualization update failed: {e}")
            
            inference_metrics = None
            if self.inference_displayer and hasattr(self.inference_displayer, 'get_recent_metrics'):
                inference_metrics = self.inference_displayer.get_recent_metrics(last_n=5)
            
            self.metrics_collector.update_epoch_metrics(
                epoch, train_metrics['loss'], 
                eval_metrics['loss'] if eval_metrics else None
            )

            self.progress_display.finish_epoch(
                train_metrics['loss'],
                self.optimizer.param_groups[0]['lr'],
                eval_metrics['loss'] if eval_metrics else None,
                inference_metrics
            )
        
        self._save_checkpoint('final_model')
        try:
            # Передаем metrics_collector при вызове finalize_training
            self.visualizer.finalize_training(history, metrics_collector=self.metrics_collector)
        except Exception as e:
            self.progress_display.log_warning(f"Visualization finalization failed: {e}")

        self.progress_display.display_training_summary()
        
        best_metrics = self.metrics_collector.get_best_metrics()
        if best_metrics:
            self.progress_display.log_info(f"Best metrics achieved:")
            if best_metrics.get('best_loss'):
                self.progress_display.log_info(f"  Best Loss: {best_metrics['best_loss']:.6f}")
            if best_metrics.get('best_cer'):
                self.progress_display.log_info(f"  Best CER: {best_metrics['best_cer']:.3f}")
            if best_metrics.get('best_wer'):
                self.progress_display.log_info(f"  Best WER: {best_metrics['best_wer']:.3f}")
        
        return history
    
    @abstractmethod
    def _train_epoch(self, dataloader: DataLoader, epoch_pbar=None) -> Dict[str, Any]:
        pass
    
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    batch_data = self._prepare_batch(batch)
                    outputs = self.model(**batch_data)
                    
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss = outputs['loss']
                    elif hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        logger.warning("No loss found in model outputs")
                        continue
                    
                    batch_size = batch_data['pixel_values'].size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    
                    if batch_idx % 10 == 0 and self.inference_displayer:
                        try:

                            comparison = self.inference_displayer.inference_engine.compare_prediction_with_ground_truth(
                                batch_data['pixel_values'][0], 
                                batch['texts'][0]
                            )
                            
                            if comparison['status'] == 'success':
                                all_predictions.append(comparison['prediction'])
                                all_targets.append(comparison['ground_truth'])
                                
                                if len(all_predictions) % 5 == 1: 
                                    self.progress_display.log_prediction_comparison(
                                        comparison['prediction'],
                                        comparison['ground_truth'],
                                        comparison['cer'],
                                        comparison['wer']
                                    )
                                    
                        except Exception as e:
                            logger.debug(f"Validation inference error: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(1, total_samples)
        result = {'loss': avg_loss}
        
        if all_predictions and all_targets:
            try:
                detailed_metrics = self.metrics_calculator.calculate_batch_metrics(
                    all_predictions, all_targets, task_type="ocr"
                )
                result.update(detailed_metrics)
                
                self.progress_display.log_validation_summary(
                    self.current_epoch, avg_loss, detailed_metrics
                )
                
            except Exception as e:
                logger.warning(f"Error calculating validation metrics: {e}")
        
        return result
    
    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        try:
            if isinstance(batch, dict):
                prepared = {}
                if 'pixel_values' in batch:
                    prepared['pixel_values'] = batch['pixel_values'].to(self.device)
                if 'labels' in batch and batch['labels'] is not None:
                    prepared['labels'] = batch['labels'].to(self.device)
                return prepared
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                prepared = {
                    'pixel_values': batch[0].to(self.device),
                    'labels': batch[1].to(self.device)
                }
                return prepared
            else:
                logger.error(f"Unsupported batch format: {type(batch)}")
                raise ValueError(f"Unsupported batch format: {type(batch)}")
        except Exception as e:
            logger.error(f"Error preparing batch: {e}")
            raise
    
    def _save_checkpoint(self, checkpoint_name: str) -> None:
        try:
            checkpoint_dir = self.output_dir / checkpoint_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(checkpoint_dir)
            else:
                torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
            
            training_state = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'scaler_state': self.scaler.state_dict() if self.scaler else None
            }
            
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            logger.info(f"Checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_name}: {e}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        try:
            if hasattr(self.model, 'from_pretrained'):
                self.model = self.model.from_pretrained(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path / "model.pt", map_location=self.device)
                self.model.load_state_dict(state_dict)
            
            training_state_path = checkpoint_path / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.device)
                self.current_epoch = training_state.get('epoch', 0)
                self.global_step = training_state.get('global_step', 0)
                self.best_metric = training_state.get('best_metric', float('inf'))
                
                if 'optimizer_state' in training_state:
                    self.optimizer.load_state_dict(training_state['optimizer_state'])
                if 'scheduler_state' in training_state and self.scheduler:
                    self.scheduler.load_state_dict(training_state['scheduler_state'])
                if 'scaler_state' in training_state and self.scaler:
                    self.scaler.load_state_dict(training_state['scaler_state'])
            
            logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise


class StandardTrainer(BaseTrainer):
    def _train_epoch(self, dataloader: DataLoader, epoch_pbar=None) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        num_batches = len(dataloader)
        
        batch_losses = []
        batch_times = []
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            try:
                batch_data = self._prepare_batch(batch)
                
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**batch_data)
                        if isinstance(outputs, dict):
                            loss = outputs.get('loss', outputs.get('logits', None))
                        else:
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                        
                        if loss is None:
                            self.progress_display.log_warning("No loss found in model outputs")
                            continue
                            
                        loss = loss / self.train_config.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(**batch_data)
                    if isinstance(outputs, dict):
                        loss = outputs.get('loss', outputs.get('logits', None))
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                    
                    if loss is None:
                        self.progress_display.log_warning("No loss found in model outputs")
                        continue
                        
                    loss = loss / self.train_config.gradient_accumulation_steps
                    loss.backward()
                
                if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.train_config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.train_config.max_grad_norm)
                        self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                batch_size = batch_data['pixel_values'].size(0)
                if isinstance(outputs, dict):
                    loss_value = outputs.get('loss', loss).item()
                else:
                    loss_value = (outputs.loss if hasattr(outputs, 'loss') else loss).item()
                
                actual_loss = loss_value * self.train_config.gradient_accumulation_steps
                batch_losses.append(actual_loss)
                
                total_loss += actual_loss * batch_size
                total_samples += batch_size
                
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                self.metrics_collector.update_batch_metrics(
                    actual_loss, 
                    self.optimizer.param_groups[0]['lr'], 
                    batch_time
                )
                
                # Логируем метрики по шагам
                if batch_idx % self.train_config.log_interval == 0:
                    inference_metrics = None
                    if self.inference_displayer and batch_idx % 50 == 0:
                        try:
                            inference_display = self.inference_displayer.display_inference(
                                batch_data, self.current_epoch, self.global_step, actual_loss, sample_idx=0
                            )
                            if inference_display:
                                
                                comparison = self.inference_displayer.inference_engine.compare_prediction_with_ground_truth(
                                    batch_data['pixel_values'][0], 
                                    batch['texts'][0]
                                )
                                
                                if comparison['status'] == 'success':
                                    self.progress_display.log_prediction_comparison(
                                        comparison['prediction'],
                                        comparison['ground_truth'],
                                        comparison['cer'],
                                        comparison['wer']
                                    )
                            
                            recent_inference = self.inference_displayer.get_recent_metrics(last_n=5)
                            if recent_inference['count'] > 0:
                                inference_metrics = recent_inference
                                self.metrics_collector.update_inference_metrics(
                                    recent_inference['avg_cer'],
                                    recent_inference['avg_wer'],
                                    recent_inference['count']
                                )
                        except Exception as e:
                            logger.debug(f"Inference display error: {e}")
                    
                    # Обновляем метрики для шага
                    self.metrics_collector.update_step_metrics(
                        self.current_epoch, 
                        batch_idx, 
                        actual_loss, 
                        self.optimizer.param_groups[0]['lr'], 
                        self.global_step,
                        inference_metrics
                    )
                    
                    # Логируем информацию о шаге
                    self.progress_display.log_step_info(
                        batch_idx,
                        self.global_step,
                        actual_loss,
                        self.optimizer.param_groups[0]['lr'],
                        inference_metrics
                    )
                    
                    self.progress_display.update_batch(
                        batch_idx, 
                        actual_loss, 
                        self.optimizer.param_groups[0]['lr'],
                        inference_metrics
                    )
                
                if batch_idx % 100 == 0 and batch_idx > 0:
                    avg_loss = sum(batch_losses[-100:]) / min(100, len(batch_losses))
                    self.progress_display.log_info(
                        f"Step {self.global_step}: Avg Loss (last 100): {avg_loss:.4f}, "
                        f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                    
            except Exception as e:
                error_msg = f"Error in training batch {batch_idx}: {e}"
                self.progress_display.log_error(error_msg)
                continue
        
        avg_epoch_loss = total_loss / max(1, total_samples)
        
        if batch_losses:
            min_loss = min(batch_losses)
            max_loss = max(batch_losses)
            avg_batch_time = sum(batch_times) / len(batch_times)
            
            summary = f"Epoch {self.current_epoch+1} Summary: "
            summary += f"Avg Loss: {avg_epoch_loss:.4f}, "
            summary += f"Min/Max Loss: {min_loss:.4f}/{max_loss:.4f}, "
            summary += f"Avg Batch Time: {avg_batch_time:.2f}s"
            
            self.progress_display.log_info(summary)
        
        return {'loss': avg_epoch_loss}


class TwoStageTrainer(BaseTrainer):
    def __init__(self, model: BaseOCRModel, train_config: TrainingConfig, data_config: DataConfig, 
                 output_dir: Path, synthetic_dataloader: Optional[DataLoader] = None, 
                 real_dataloader: Optional[DataLoader] = None, inference_displayer=None):
        super().__init__(model, train_config, data_config, output_dir, inference_displayer)
        
        self.synthetic_dataloader = synthetic_dataloader
        self.real_dataloader = real_dataloader
        self.stage_transition_epoch = getattr(train_config, 'stage_transition_epochs', train_config.num_epochs // 2)
        
        if getattr(train_config, 'enable_two_stage', False):
            self._setup_two_stage_training()
    
    def _setup_two_stage_training(self) -> None:
        total_epochs = self.train_config.num_epochs
        self.stage1_epochs = min(self.stage_transition_epoch, total_epochs // 2)
        self.stage2_start = self.stage1_epochs
        
        logger.info(f"Stage 1 (synthetic): epochs 0-{self.stage1_epochs-1}")
        logger.info(f"Stage 2 (real): epochs {self.stage2_start}-{total_epochs-1}")
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        if not getattr(self.train_config, 'enable_two_stage', False):
            return super().train(train_dataloader, eval_dataloader)
        
        total_steps = len(train_dataloader) * self.train_config.num_epochs
        self._setup_scheduler(total_steps)
        
        history = {'train_loss': [], 'eval_loss': [], 'learning_rates': [], 'stage_info': []}
        
        logger.info(f"Starting two-stage training for {self.train_config.num_epochs} epochs")
        
        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch
            
            current_dataloader, stage_name = self._get_current_stage_dataloader(epoch, train_dataloader)
            self._adjust_learning_rate_for_stage(stage_name)
            
            train_metrics = self._train_epoch(current_dataloader)
            train_metrics['stage'] = stage_name
            
            history['train_loss'].append(train_metrics['loss'])
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            history['stage_info'].append(stage_name)
            
            if eval_dataloader and epoch % self.train_config.eval_interval == 0:
                eval_metrics = self._eval_epoch(eval_dataloader)
                history['eval_loss'].append(eval_metrics['loss'])
                
                if eval_metrics['loss'] < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self._save_checkpoint('best_model')
            
            if epoch % self.train_config.save_interval == 0:
                self._save_checkpoint(f'checkpoint-epoch-{epoch}')
            
            try:
                if hasattr(self.visualizer, 'update_two_stage_progress'):
                    self.visualizer.update_two_stage_progress(epoch, train_metrics, history)
                else:
                    self.visualizer.update_training_progress(epoch, train_metrics, history)
            except Exception as e:
                logger.warning(f"Visualization update failed: {e}")
            
            logger.info(f"Epoch {epoch} ({stage_name}): loss={train_metrics['loss']:.4f}")
        
        self._save_checkpoint('final_model')
        try:
            if hasattr(self.visualizer, 'finalize_two_stage_training'):
                self.visualizer.finalize_two_stage_training(history, metrics_collector=self.metrics_collector)
            else:
                self.visualizer.finalize_training(history, metrics_collector=self.metrics_collector)
        except Exception as e:
            logger.warning(f"Visualization finalization failed: {e}")
        return history
    
    def _get_current_stage_dataloader(self, epoch: int, default_dataloader: DataLoader):
        if epoch < self.stage1_epochs:
            return (self.synthetic_dataloader or default_dataloader, "synthetic")
        else:
            return (self.real_dataloader or default_dataloader, "real")
    
    def _adjust_learning_rate_for_stage(self, stage_name: str) -> None:
        base_lr = self.train_config.learning_rate
        
        if stage_name == "synthetic":
            lr_factor = getattr(self.train_config, 'synthetic_lr_factor', 1.0)
        elif stage_name == "real":
            lr_factor = getattr(self.train_config, 'real_data_lr_factor', 0.5)
        else:
            lr_factor = 1.0
        
        adjusted_lr = base_lr * lr_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        return StandardTrainer._train_epoch(self, dataloader)


class DistributedTrainer(StandardTrainer):
    def __init__(self, model: BaseOCRModel, train_config: TrainingConfig, data_config: DataConfig, 
                 output_dir: Path, local_rank: int = 0):
        super().__init__(model, train_config, data_config, output_dir)
        self.local_rank = local_rank
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
    
    def _save_checkpoint(self, checkpoint_name: str) -> None:
        if self.local_rank == 0:
            super()._save_checkpoint(checkpoint_name)
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.current_epoch)
        
        return super()._train_epoch(dataloader)


def create_trainer(
    trainer_type: str,
    model: BaseOCRModel,
    train_config: TrainingConfig,
    data_config: DataConfig,
    output_dir: Path,
    **kwargs
) -> BaseTrainer:
    trainer_type = trainer_type.lower()
    
    if trainer_type == "standard":
        return StandardTrainer(model, train_config, data_config, output_dir)
    elif trainer_type == "two_stage":
        return TwoStageTrainer(
            model, train_config, data_config, output_dir,
            synthetic_dataloader=kwargs.get('synthetic_dataloader'),
            real_dataloader=kwargs.get('real_dataloader'),
            inference_displayer=kwargs.get('inference_displayer')
        )
    elif trainer_type == "distributed":
        return DistributedTrainer(
            model, train_config, data_config, output_dir,
            local_rank=kwargs.get('local_rank', 0)
        )
    else:
        logger.warning(f"Unknown trainer type: {trainer_type}, using StandardTrainer")
        return StandardTrainer(model, train_config, data_config, output_dir)