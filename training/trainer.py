"""
Advanced training classes with two-stage training support.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import logging
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_cosine_schedule_with_warmup

from ..core.base import BaseOCRModel
from ..core.config import TrainingConfig, DataConfig
from .metrics import MetricsCalculator
from .visualization import TrainingVisualizer

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer with core training functionality."""
    
    def __init__(
        self,
        model: BaseOCRModel,
        train_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Union[str, Path]
    ):
        self.model = model
        self.train_config = train_config
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = model.device
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = TrainingVisualizer(self.output_dir, train_config)
        
        self._setup_training()
    
    def _setup_training(self) -> None:
        """Setup training components."""
        # Setup optimizer
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = AdamW(
            trainable_params,
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        
        # Setup mixed precision scaler
        if self.train_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup random seed
        self._set_seed(self.train_config.seed)
        
        logger.info(f"Training setup complete. Device: {self.device}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    def _setup_scheduler(self, total_steps: int) -> None:
        """Setup learning rate scheduler."""
        warmup_steps = int(total_steps * self.train_config.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Main training loop."""
        
        total_steps = len(train_dataloader) * self.train_config.num_epochs
        self._setup_scheduler(total_steps)
        
        training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': []
        }
        
        logger.info(f"Starting training for {self.train_config.num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        
        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch(train_dataloader)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Evaluate
            if eval_dataloader is not None and epoch % self.train_config.eval_interval == 0:
                eval_metrics = self._eval_epoch(eval_dataloader)
                training_history['eval_loss'].append(eval_metrics['loss'])
                
                # Check for best model
                if eval_metrics['loss'] < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self._save_checkpoint('best_model')
                
                # Early stopping check
                if self._should_stop_early(training_history):
                    logger.info("Early stopping triggered")
                    break
            
            # Save checkpoint
            if epoch % self.train_config.save_interval == 0:
                self._save_checkpoint(f'checkpoint-epoch-{epoch}')
            
            # Visualize progress
            self.visualizer.update_training_progress(epoch, train_metrics, training_history)
            
            logger.info(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Final save
        self._save_checkpoint('final_model')
        self.visualizer.finalize_training(training_history)
        
        return training_history
    
    @abstractmethod
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Train single epoch - to be implemented by subclasses."""
        pass
    
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate single epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_size = batch['pixel_values'].size(0)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=batch['pixel_values'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                
                total_loss += outputs['loss'].item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return {'loss': avg_loss}
    
    def _should_stop_early(self, history: Dict[str, List]) -> bool:
        """Check early stopping condition."""
        if self.train_config.early_stopping_patience is None:
            return False
        
        eval_losses = history.get('eval_loss', [])
        if len(eval_losses) < self.train_config.early_stopping_patience:
            return False
        
        # Check if loss hasn't improved for patience epochs
        recent_losses = eval_losses[-self.train_config.early_stopping_patience:]
        min_recent = min(recent_losses)
        
        return min_recent >= (eval_losses[-self.train_config.early_stopping_patience-1] - 
                             self.train_config.early_stopping_threshold)
    
    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
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


class TwoStageTrainer(BaseTrainer):
    """Enhanced trainer with two-stage training support."""
    
    def __init__(
        self,
        model: BaseOCRModel,
        train_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Union[str, Path],
        synthetic_dataloader: Optional[DataLoader] = None,
        real_dataloader: Optional[DataLoader] = None
    ):
        super().__init__(model, train_config, data_config, output_dir)
        
        self.synthetic_dataloader = synthetic_dataloader
        self.real_dataloader = real_dataloader
        self.stage_transition_epoch = train_config.stage_transition_epochs
        
        if train_config.enable_two_stage:
            self._setup_two_stage_training()
    
    def _setup_two_stage_training(self) -> None:
        """Setup two-stage training parameters."""
        logger.info("Setting up two-stage training")
        
        # Calculate stage boundaries
        total_epochs = self.train_config.num_epochs
        self.stage1_epochs = min(self.stage_transition_epoch, total_epochs // 2)
        self.stage2_start = self.stage1_epochs
        
        logger.info(f"Stage 1 (synthetic): epochs 0-{self.stage1_epochs-1}")
        logger.info(f"Stage 2 (real): epochs {self.stage2_start}-{total_epochs-1}")
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """Two-stage training with synthetic and real data."""
        
        if not self.train_config.enable_two_stage:
            return super().train(train_dataloader, eval_dataloader)
        
        total_steps = len(train_dataloader) * self.train_config.num_epochs
        self._setup_scheduler(total_steps)
        
        training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
            'stage_info': []
        }
        
        logger.info(f"Starting two-stage training for {self.train_config.num_epochs} epochs")
        
        for epoch in range(self.train_config.num_epochs):
            self.current_epoch = epoch
            
            # Determine current stage and dataloader
            current_dataloader, stage_name = self._get_current_stage_dataloader(epoch, train_dataloader)
            
            # Adjust learning rate for stage
            self._adjust_learning_rate_for_stage(stage_name)
            
            # Train epoch
            train_metrics = self._train_epoch(current_dataloader)
            train_metrics['stage'] = stage_name
            
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            training_history['stage_info'].append(stage_name)
            
            # Evaluate on both synthetic and real data if available
            if eval_dataloader is not None and epoch % self.train_config.eval_interval == 0:
                eval_metrics = self._eval_epoch(eval_dataloader)
                training_history['eval_loss'].append(eval_metrics['loss'])
                
                if eval_metrics['loss'] < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self._save_checkpoint('best_model')
            
            # Save checkpoint
            if epoch % self.train_config.save_interval == 0:
                self._save_checkpoint(f'checkpoint-epoch-{epoch}')
            
            # Visualize progress with stage information
            self.visualizer.update_two_stage_progress(epoch, train_metrics, training_history)
            
            logger.info(f"Epoch {epoch} ({stage_name}): train_loss={train_metrics['loss']:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Final save
        self._save_checkpoint('final_model')
        self.visualizer.finalize_two_stage_training(training_history)
        
        return training_history
    
    def _get_current_stage_dataloader(
        self, 
        epoch: int, 
        default_dataloader: DataLoader
    ) -> Tuple[DataLoader, str]:
        """Get appropriate dataloader for current training stage."""
        
        if epoch < self.stage1_epochs:
            # Stage 1: Synthetic data
            if self.synthetic_dataloader is not None:
                return self.synthetic_dataloader, "synthetic"
            else:
                return default_dataloader, "mixed"
        else:
            # Stage 2: Real data
            if self.real_dataloader is not None:
                return self.real_dataloader, "real"
            else:
                return default_dataloader, "mixed"
    
    def _adjust_learning_rate_for_stage(self, stage_name: str) -> None:
        """Adjust learning rate based on training stage."""
        base_lr = self.train_config.learning_rate
        
        if stage_name == "synthetic":
            lr_factor = self.train_config.synthetic_lr_factor
        elif stage_name == "real":
            lr_factor = self.train_config.real_data_lr_factor
        else:
            lr_factor = 1.0
        
        adjusted_lr = base_lr * lr_factor
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Train single epoch with gradient accumulation and mixed precision."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        accumulated_loss = 0.0
        
        # Progress tracking
        num_batches = len(dataloader)
        log_interval = max(1, num_batches // 10)  # Log 10 times per epoch
        
        for batch_idx, batch in enumerate(dataloader):
            batch_size = batch['pixel_values'].size(0)
            
            # Mixed precision forward pass
            if self.train_config.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        pixel_values=batch['pixel_values'].to(self.device),
                        labels=batch['labels'].to(self.device)
                    )
                    loss = outputs['loss'] / self.train_config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
                
            else:
                outputs = self.model(
                    pixel_values=batch['pixel_values'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                loss = outputs['loss'] / self.train_config.gradient_accumulation_steps
                
                loss.backward()
                accumulated_loss += loss.item()
            
            # Gradient accumulation and optimizer step
            if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                
                if self.train_config.mixed_precision and self.scaler is not None:
                    # Gradient clipping with scaler
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.train_config.max_grad_norm
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard gradient clipping and step
                    torch.nn.utils.clip_grad_norm_(
                        self.model.get_trainable_parameters(),
                        self.train_config.max_grad_norm
                    )
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                
                self.global_step += 1
            
            total_samples += batch_size
            
            # Logging
            if batch_idx % log_interval == 0:
                current_loss = total_loss / max(1, (batch_idx + 1) // self.train_config.gradient_accumulation_steps)
                logger.debug(f"Batch {batch_idx}/{num_batches}: loss={current_loss:.4f}")
        
        # Handle remaining accumulated gradients
        if accumulated_loss > 0:
            total_loss += accumulated_loss
        
        avg_loss = total_loss / max(1, num_batches // self.train_config.gradient_accumulation_steps)
        
        return {
            'loss': avg_loss,
            'total_samples': total_samples
        }