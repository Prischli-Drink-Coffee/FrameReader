"""
Enhanced progress bar and real-time metrics display for training.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import time

try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedProgressDisplay:
    """Enhanced progress display with real-time metrics and beautiful formatting."""
    
    def __init__(self, total_epochs: int, use_rich: bool = True):
        self.total_epochs = total_epochs
        self.use_rich = use_rich and RICH_AVAILABLE
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        
        self.epoch_losses = []
        self.batch_losses = []
        self.learning_rates = []
        self.inference_metrics = []
        
        self.epoch_start_time = None
        self.batch_start_time = None
        self.training_start_time = time.time()
        
        if self.use_rich:
            self.console = Console()
            self._setup_rich_display()
        else:
            logger.info("Rich not available, falling back to standard progress bars")
    
    def _setup_rich_display(self):
        """Setup Rich-based progress display."""
        if not self.use_rich:
            return
            
        self.epoch_progress = Progress(
            TextColumn("[bold blue]Epochs"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        self.batch_progress = Progress(
            TextColumn("[bold green]Batches"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            console=self.console
        )
        
        self.epoch_task = None
        self.batch_task = None
    
    def start_training(self):
        """Start training progress display."""
        self.training_start_time = time.time()
        
        if self.use_rich:
            self.epoch_task = self.epoch_progress.add_task(
                "Training Progress", 
                total=self.total_epochs
            )
            logger.info("🚀 Starting training with enhanced progress display")
        else:
            logger.info("🚀 Starting training")
    
    def start_epoch(self, epoch: int, total_batches: int):
        """Start a new epoch."""
        self.current_epoch = epoch
        self.total_batches = total_batches
        self.current_batch = 0
        self.epoch_start_time = time.time()
        self.batch_losses = []
        
        if self.use_rich:
            if self.batch_task is not None:
                self.batch_progress.remove_task(self.batch_task)
            
            self.batch_task = self.batch_progress.add_task(
                f"Epoch {epoch+1}/{self.total_epochs}",
                total=total_batches
            )
            
            self.epoch_progress.update(
                self.epoch_task,
                completed=epoch,
                description=f"Epoch {epoch+1}/{self.total_epochs}"
            )
        else:
            logger.info(f"📊 Starting epoch {epoch+1}/{self.total_epochs}")
    
    def update_batch(self, batch_idx: int, loss: float, lr: float, 
                    inference_metrics: Optional[Dict] = None):
        """Update batch progress."""
        self.current_batch = batch_idx + 1
        self.batch_losses.append(loss)
        
        if self.use_rich and self.batch_task is not None:
            recent_losses = self.batch_losses[-10:] if len(self.batch_losses) >= 10 else self.batch_losses
            avg_loss = sum(recent_losses) / len(recent_losses)
            
            description = f"Epoch {self.current_epoch+1}/{self.total_epochs} | Loss: {loss:.4f} | Avg: {avg_loss:.4f}"
            
            if inference_metrics:
                if 'cer' in inference_metrics:
                    description += f" | CER: {inference_metrics['cer']:.3f}"
                if 'wer' in inference_metrics:
                    description += f" | WER: {inference_metrics['wer']:.3f}"
            
            self.batch_progress.update(
                self.batch_task,
                completed=self.current_batch,
                description=description
            )
    
    def finish_epoch(self, epoch_loss: float, lr: float, eval_loss: Optional[float] = None,
                    inference_metrics: Optional[Dict] = None):
        """Finish current epoch."""
        self.epoch_losses.append(epoch_loss)
        self.learning_rates.append(lr)
        
        if inference_metrics:
            self.inference_metrics.append(inference_metrics)
        
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        if self.use_rich:
            self.epoch_progress.update(
                self.epoch_task,
                completed=self.current_epoch + 1
            )
            
            self._display_epoch_summary(epoch_loss, lr, eval_loss, epoch_time, inference_metrics)
        else:
            log_msg = f"✅ Epoch {self.current_epoch+1} completed - "
            log_msg += f"Loss: {epoch_loss:.4f}, LR: {lr:.2e}, Time: {epoch_time:.1f}s"
            if eval_loss is not None:
                log_msg += f", Val_Loss: {eval_loss:.4f}"
            if inference_metrics:
                if 'avg_cer' in inference_metrics:
                    log_msg += f", CER: {inference_metrics['avg_cer']:.3f}"
                if 'avg_wer' in inference_metrics:
                    log_msg += f", WER: {inference_metrics['avg_wer']:.3f}"
            
            logger.info(log_msg)
    
    def _display_epoch_summary(self, epoch_loss: float, lr: float, eval_loss: Optional[float],
                              epoch_time: float, inference_metrics: Optional[Dict]):
        """Display rich epoch summary."""
        if not self.use_rich:
            return
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Epoch", f"{self.current_epoch+1}/{self.total_epochs}")
        table.add_row("Training Loss", f"{epoch_loss:.6f}")
        
        if eval_loss is not None:
            table.add_row("Validation Loss", f"{eval_loss:.6f}")

            if len(self.epoch_losses) > 1:
                prev_loss = self.epoch_losses[-2]
                improvement = prev_loss - epoch_loss
                status = "📈" if improvement > 0 else "📉"
                table.add_row("Loss Change", f"{status} {improvement:+.6f}")
        
        table.add_row("Learning Rate", f"{lr:.2e}")
        table.add_row("Epoch Time", f"{epoch_time:.1f}s")
        
        if inference_metrics:
            if 'avg_cer' in inference_metrics:
                cer_status = "🎯" if inference_metrics['avg_cer'] < 0.1 else "⚠️" if inference_metrics['avg_cer'] < 0.3 else "❌"
                table.add_row("Character Error Rate", f"{cer_status} {inference_metrics['avg_cer']:.3f}")
            if 'avg_wer' in inference_metrics:
                wer_status = "🎯" if inference_metrics['avg_wer'] < 0.1 else "⚠️" if inference_metrics['avg_wer'] < 0.3 else "❌"
                table.add_row("Word Error Rate", f"{wer_status} {inference_metrics['avg_wer']:.3f}")
        
        total_time = time.time() - self.training_start_time
        if self.current_epoch > 0:
            avg_epoch_time = total_time / (self.current_epoch + 1)
            remaining_epochs = self.total_epochs - (self.current_epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            table.add_row("Estimated Remaining", f"⏰ {estimated_remaining/60:.1f}m")
            
            progress_pct = ((self.current_epoch + 1) / self.total_epochs) * 100
            table.add_row("Progress", f"📊 {progress_pct:.1f}%")
        
        panel = Panel(
            table,
            title=f"[bold blue]🏆 Epoch {self.current_epoch+1}/{self.total_epochs} Summary[/bold blue]",
            border_style="blue"
        )
        
        self.console.print(panel)
    
    def log_prediction_comparison(self, prediction: str, ground_truth: str, cer: float, wer: float):
        """Log prediction vs ground truth comparison."""
        if self.use_rich:

            table = Table(show_header=True, header_style="bold yellow", box=box.SIMPLE)
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Text", style="white", width=80)
            
            max_len = 100
            
            pred_display = str(prediction)
            gt_display = str(ground_truth)
                
            if not gt_display or gt_display.strip() == "":
                gt_display = "<Пустая строка ground truth!>"
                logger.warning(f"Обнаружена пустая строка ground truth для предсказания: {pred_display}")
                
            pred_display = pred_display[:max_len] + "..." if len(pred_display) > max_len else pred_display
            gt_display = gt_display[:max_len] + "..." if len(gt_display) > max_len else gt_display
            
            table.add_row("🤖 Pred", pred_display)
            table.add_row("✅ Truth", gt_display)
            table.add_row("📊 CER", f"{cer:.3f}")
            table.add_row("📊 WER", f"{wer:.3f}")
            
            if cer < 0.1 and wer < 0.1:
                status = "[green]🎯 Excellent[/green]"
            elif cer < 0.3 and wer < 0.3:
                status = "[yellow]⚠️ Good[/yellow]"
            else:
                status = "[red]❌ Needs Improvement[/red]"
            
            table.add_row("🎯 Quality", status)
            
            panel = Panel(
                table,
                title="[bold cyan]🔍 Real-time Inference Comparison[/bold cyan]",
                border_style="cyan",
                width=120
            )
            
            self.console.print(panel)
        else:
            pred_str = str(prediction)
            gt_str = str(ground_truth)
                
            if not gt_str or gt_str.strip() == "":
                gt_str = "<Пустая строка ground truth!>"
                logger.warning(f"Обнаружена пустая строка ground truth для предсказания: {pred_str}")
                
            logger.info(f"🔍 Prediction vs Ground Truth:")
            logger.info(f"  🤖 Pred: {pred_str[:100]}{'...' if len(pred_str) > 100 else ''}")
            logger.info(f"  ✅ GT:   {gt_str[:100]}{'...' if len(gt_str) > 100 else ''}")
            logger.info(f"  📊 CER: {cer:.3f}, WER: {wer:.3f}")
    
    def log_validation_summary(self, epoch: int, val_loss: float, val_metrics: Optional[Dict] = None):
        """Log validation summary with metrics."""
        if self.use_rich:
            table = Table(show_header=True, header_style="bold green", box=box.DOUBLE_EDGE)
            table.add_column("Validation Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Epoch", f"{epoch+1}/{self.total_epochs}")
            table.add_row("Validation Loss", f"{val_loss:.6f}")
            
            if val_metrics:
                for metric_name, metric_value in val_metrics.items():
                    if isinstance(metric_value, float):
                        if 'error' in metric_name.lower() or 'cer' in metric_name.lower() or 'wer' in metric_name.lower():

                            status = "🎯" if metric_value < 0.1 else "⚠️" if metric_value < 0.3 else "❌"
                        else:

                            status = "🎯" if metric_value > 0.9 else "⚠️" if metric_value > 0.7 else "❌"
                        table.add_row(metric_name.replace('_', ' ').title(), f"{status} {metric_value:.4f}")
                    else:
                        table.add_row(metric_name.replace('_', ' ').title(), str(metric_value))
            
            panel = Panel(
                table,
                title=f"[bold green]📋 Validation Results - Epoch {epoch+1}[/bold green]",
                border_style="green"
            )
            
            self.console.print(panel)
        else:
            log_msg = f"📋 Validation Epoch {epoch+1}: Loss = {val_loss:.6f}"
            if val_metrics:
                metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                       for k, v in val_metrics.items()])
                log_msg += f", {metrics_str}"
            logger.info(log_msg)
    
    def display_training_summary(self):
        """Display final training summary."""
        total_time = time.time() - self.training_start_time
        
        if self.use_rich:

            summary_table = Table(show_header=True, header_style="bold yellow", box=box.DOUBLE)
            summary_table.add_column("Training Summary", style="cyan", width=20)
            summary_table.add_column("Value", style="green", width=15)
            
            summary_table.add_row("Total Epochs", str(self.total_epochs))
            summary_table.add_row("Total Time", f"{total_time/60:.1f}m")
            summary_table.add_row("Avg Time/Epoch", f"{total_time/self.total_epochs:.1f}s")
            
            if self.epoch_losses:
                summary_table.add_row("Final Loss", f"{self.epoch_losses[-1]:.6f}")
                summary_table.add_row("Best Loss", f"{min(self.epoch_losses)::.6f}")
                summary_table.add_row("Loss Improvement", f"{self.epoch_losses[0] - self.epoch_losses[-1]:.6f}")
            
            if self.inference_metrics:
                last_metrics = self.inference_metrics[-1]
                if 'avg_cer' in last_metrics:
                    summary_table.add_row("Final CER", f"{last_metrics['avg_cer']:.3f}")
                if 'avg_wer' in last_metrics:
                    summary_table.add_row("Final WER", f"{last_metrics['avg_wer']:.3f}")
            
            panel = Panel(
                summary_table,
                title="[bold green]🎉 Training Completed Successfully![/bold green]",
                border_style="green"
            )
            
            self.console.print(panel)
        else:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"Total time: {total_time/60:.1f}m")
            if self.epoch_losses:
                logger.info(f"Final loss: {self.epoch_losses[-1]:.6f}")
                logger.info(f"Best loss: {min(self.epoch_losses)::.6f}")
    
    def log_inference_result(self, result: str):
        """Log inference result during training."""
        if self.use_rich:
            self.console.print(f"🔍 [cyan]Inference:[/cyan] {result}")
        else:
            logger.info(f"🔍 Inference: {result}")
    
    def log_error(self, message: str):
        """Log error message."""
        if self.use_rich:
            self.console.print(f"❌ [red]Error:[/red] {message}")
        else:
            logger.error(f"❌ Error: {message}")
    
    def log_warning(self, message: str):
        """Log warning message."""
        if self.use_rich:
            self.console.print(f"⚠️  [yellow]Warning:[/yellow] {message}")
        else:
            logger.warning(f"⚠️  Warning: {message}")
    
    def log_info(self, message: str):
        """Log info message."""
        if self.use_rich:
            self.console.print(f"ℹ️  [blue]Info:[/blue] {message}")
        else:
            logger.info(f"ℹ️  Info: {message}")
    
    def log_step_info(self, step: int, global_step: int, loss: float, lr: float, 
                inference_metrics: Optional[Dict] = None):
        """Log step information during training."""
        description = f"Step {global_step} (Batch {step}) | Loss: {loss:.4f}"
        
        if inference_metrics:
            if 'avg_cer' in inference_metrics:
                description += f" | CER: {inference_metrics['avg_cer']:.3f}"
            if 'avg_wer' in inference_metrics:
                description += f" | WER: {inference_metrics['avg_wer']:.3f}"
                
        if self.use_rich:
            self.console.print(f"🔄 [cyan]Step Info:[/cyan] {description}")
        else:
            logger.info(f"🔄 Step Info: {description}")


class MetricsCollector:
    """Collects and stores metrics during training for visualization and analysis."""
    
    def __init__(self):
        self.step_metrics = []
        self.epoch_metrics = []
        self.inference_metrics = []
        self.best_metrics = {
            'best_loss': float('inf'),
            'best_cer': float('inf'),
            'best_wer': float('inf'),
            'best_exact_match': 0.0,
            'best_epoch': -1
        }
    
    def update_batch_metrics(self, loss: float, learning_rate: float, batch_time: float) -> None:
        """Обновляет метрики для текущего батча."""
        self.step_metrics.append({
            'loss': loss,
            'learning_rate': learning_rate,
            'batch_time': batch_time
        })
    
    def update_step_metrics(self, epoch: int, batch_idx: int, loss: float, learning_rate: float, 
                          global_step: int, inference_metrics: Optional[Dict] = None) -> None:
        """Обновляет метрики для текущего шага."""
        step_data = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'loss': loss,
            'learning_rate': learning_rate,
            'global_step': global_step,
            'timestamp': import_time().time()
        }
        
        if inference_metrics:
            step_data.update({
                'inference_cer': inference_metrics.get('avg_cer', None),
                'inference_wer': inference_metrics.get('avg_wer', None),
                'inference_count': inference_metrics.get('count', 0),
            })
        
        self.step_metrics.append(step_data)
    
    def update_epoch_metrics(self, epoch: int, train_loss: float, 
                           eval_loss: Optional[float] = None) -> None:
        """Обновляет метрики для текущей эпохи."""
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
        }
        
        if eval_loss is not None:
            epoch_data['eval_loss'] = eval_loss
            if eval_loss < self.best_metrics['best_loss']:
                self.best_metrics['best_loss'] = eval_loss
                self.best_metrics['best_epoch'] = epoch
        
        self.epoch_metrics.append(epoch_data)
    
    def update_inference_metrics(self, cer: float, wer: float, sample_count: int) -> None:
        """Обновляет метрики инференса."""
        inference_data = {
            'cer': cer,
            'wer': wer,
            'sample_count': sample_count,
            'timestamp': import_time().time()
        }
        
        # Обновляем лучшие метрики
        if cer < self.best_metrics['best_cer']:
            self.best_metrics['best_cer'] = cer
        
        if wer < self.best_metrics['best_wer']:
            self.best_metrics['best_wer'] = wer
        
        self.inference_metrics.append(inference_data)
    
    def get_step_metrics_history(self) -> List[Dict]:
        """Возвращает историю метрик по шагам."""
        return self.step_metrics
    
    def get_epoch_metrics_history(self) -> List[Dict]:
        """Возвращает историю метрик по эпохам."""
        return self.epoch_metrics
    
    def get_inference_metrics_history(self) -> List[Dict]:
        """Возвращает историю метрик инференса."""
        return self.inference_metrics
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Возвращает лучшие метрики за все обучение."""
        return self.best_metrics
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Возвращает последние метрики."""
        result = {}
        
        if self.epoch_metrics:
            result.update(self.epoch_metrics[-1])
            
        if self.inference_metrics:
            latest_inference = self.inference_metrics[-1]
            result.update({
                'latest_cer': latest_inference['cer'],
                'latest_wer': latest_inference['wer']
            })
        
        return result
    
    def calculate_metrics_trend(self, metric_name: str, window_size: int = 5) -> Dict[str, float]:
        """Рассчитывает тренд для указанной метрики на основе последних значений."""
        if not self.epoch_metrics:
            return {'trend': 0.0, 'improvement': 0.0}
        
        values = []
        for metric in self.epoch_metrics:
            if metric_name in metric:
                values.append(metric[metric_name])
        
        if len(values) < 2:
            return {'trend': 0.0, 'improvement': 0.0}
        
        window = values[-min(window_size, len(values)):]
        
        if len(window) < 2:
            return {'trend': 0.0, 'improvement': 0.0}
            
        # Линейный тренд (положительный = ухудшение для loss, cer, wer)
        trend = (window[-1] - window[0]) / len(window)
        
        # Общее улучшение от начала обучения
        improvement = values[0] - values[-1]
        
        # Для метрик, где больше = лучше (accuracy, f1), инвертируем знак
        if metric_name in ['exact_match', 'accuracy', 'f1', 'precision', 'recall', 'bleu', 'rouge_l']:
            trend = -trend
            improvement = -improvement
        
        return {'trend': trend, 'improvement': improvement}

# Фикс проблемы с импортом time
def import_time():
    import time
    return time