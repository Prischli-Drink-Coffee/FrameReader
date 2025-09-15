"""
Comprehensive visualization system for OCR training and results.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available, training visualization will be disabled")

from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from core.config import TrainingConfig

logger = logging.getLogger(__name__)

if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")


class TrainingVisualizer:
    def __init__(self, output_dir: Path, config: TrainingConfig):
        self.output_dir = output_dir
        self.config = config
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.metrics_history = []
        self.stage_history = []
        
        self.plot_config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'format': 'png',
            'bbox_inches': 'tight'
        }
    
    def update_training_progress(self, epoch: int, metrics: Dict[str, Any], history: Dict[str, List], metrics_collector=None) -> None:
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': metrics.get('stage', 'standard')
        })
        
        if epoch % max(1, self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs - 1:
            if MATPLOTLIB_AVAILABLE:
                # Строим графики обучения
                self._plot_training_curves(history)
                
                # Строим график валидационных метрик на каждой эпохе
                self._plot_validation_metrics(history)
                
                # Строим график метрик по эпохам если metrics_collector доступен
                if metrics_collector is not None:
                    self._plot_metrics_by_epoch(metrics_collector)
            else:
                logger.info(f"Epoch {epoch}: Training progress (visualization disabled)")
    
    def update_two_stage_progress(self, epoch: int, metrics: Dict[str, Any], history: Dict[str, List], metrics_collector=None) -> None:
        stage = metrics.get('stage', 'unknown')
        
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': stage
        })
        
        self.stage_history.append(stage)
        
        if epoch % max(1, self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs - 1:
            if MATPLOTLIB_AVAILABLE:
                # Строим график двухстадийного обучения
                self._plot_two_stage_curves(history)
                
                # Строим график валидационных метрик на каждой эпохе
                self._plot_validation_metrics(history)
                
                # Строим график метрик по эпохам если metrics_collector доступен
                if metrics_collector is not None:
                    self._plot_metrics_by_epoch(metrics_collector)
            else:
                logger.info(f"Epoch {epoch}: Two-stage progress (visualization disabled)")
    
    def _plot_training_curves(self, history: Dict[str, List]) -> None:
        if not MATPLOTLIB_AVAILABLE:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.plot_config['figure_size'])
        
        epochs = list(range(len(history['train_loss'])))
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'eval_loss' in history and history['eval_loss']:
            eval_epochs = [i * max(1, len(epochs) // len(history['eval_loss'])) 
                          for i in range(len(history['eval_loss']))]
            ax1.plot(eval_epochs, history['eval_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        if len(history['train_loss']) > 5:
            window_size = min(5, len(history['train_loss']) // 4)
            smoothed_loss = self._moving_average(history['train_loss'], window_size)
            ax3.plot(epochs, history['train_loss'], alpha=0.3, color='blue', label='Original')
            ax3.plot(epochs, smoothed_loss, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
        else:
            ax3.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Smoothed Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        if self.metrics_history:
            recent_metrics = [m['metrics'] for m in self.metrics_history[-10:]]
            if recent_metrics:
                ax4.text(0.1, 0.5, 'Recent Training\nMetrics', transform=ax4.transAxes)
                ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "training_curves.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()
    
    def _plot_two_stage_curves(self, history: Dict[str, List]) -> None:
        if not MATPLOTLIB_AVAILABLE:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        epochs = list(range(len(history['train_loss'])))
        stage_info = history.get('stage_info', [])
        
        synthetic_epochs = [i for i, stage in enumerate(stage_info) if stage == 'synthetic']
        real_epochs = [i for i, stage in enumerate(stage_info) if stage == 'real']
        mixed_epochs = [i for i, stage in enumerate(stage_info) if stage == 'mixed']
        
        if synthetic_epochs:
            synthetic_losses = [history['train_loss'][i] for i in synthetic_epochs]
            ax1.plot(synthetic_epochs, synthetic_losses, 'b-', label='Synthetic Data', linewidth=2)
        
        if real_epochs:
            real_losses = [history['train_loss'][i] for i in real_epochs]
            ax1.plot(real_epochs, real_losses, 'r-', label='Real Data', linewidth=2)
        
        if mixed_epochs:
            mixed_losses = [history['train_loss'][i] for i in mixed_epochs]
            ax1.plot(mixed_epochs, mixed_losses, 'g-', label='Mixed Data', linewidth=2)
        
        if synthetic_epochs and real_epochs:
            transition_epoch = max(synthetic_epochs) + 0.5
            ax1.axvline(x=transition_epoch, color='black', linestyle='--', alpha=0.7, 
                       label='Stage Transition')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Two-Stage Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        colors = ['blue' if stage == 'synthetic' else 'red' if stage == 'real' else 'green' 
                 for stage in stage_info]
        
        for i, (epoch, lr, color) in enumerate(zip(epochs, history['learning_rates'], colors)):
            ax2.scatter(epoch, lr, c=color, alpha=0.7, s=20)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate by Training Stage')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Synthetic'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Real'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Mixed')
        ]
        ax2.legend(handles=legend_elements)
        
        stage_counts = {}
        for stage in stage_info:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        if stage_counts:
            stages = list(stage_counts.keys())
            counts = list(stage_counts.values())
            colors_pie = ['lightblue' if s == 'synthetic' else 'lightcoral' if s == 'real' else 'lightgreen' 
                         for s in stages]
            
            wedges, texts, autotexts = ax3.pie(counts, labels=stages, colors=colors_pie, autopct='%1.1f%%',
                                              startangle=90)
            ax3.set_title('Training Stage Distribution')
        
        if len(history['train_loss']) > 1:
            loss_improvements = [history['train_loss'][i-1] - history['train_loss'][i] 
                               for i in range(1, len(history['train_loss']))]
            
            ax4.plot(range(1, len(loss_improvements) + 1), loss_improvements, 'purple', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Improvement')
            ax4.set_title('Epoch-to-Epoch Loss Improvement')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "two_stage_training_curves.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i + 1]
            smoothed.append(sum(window_data) / len(window_data))
        
        return smoothed
    
    def visualize_inference_results(self, image: Union[Image.Image, 'np.ndarray'], prediction: str,
                                   ground_truth: Optional[str] = None, confidence: Optional[float] = None,
                                   bounding_boxes: Optional[List[Dict]] = None) -> Image.Image:
        if OPENCV_AVAILABLE and hasattr(image, 'shape'):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = font
        
        if bounding_boxes:
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox.get('coords', [0, 0, 100, 100])
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        img_width, img_height = viz_image.size
        
        text_bg_height = 120
        text_bg = Image.new('RGB', (img_width, text_bg_height), color='white')
        text_draw = ImageDraw.Draw(text_bg)
        
        pred_text = f"Prediction: {prediction[:100]}{'...' if len(prediction) > 100 else ''}"
        text_draw.text((10, 10), pred_text, fill='blue', font=font)
        
        if ground_truth is not None:
            gt_text = f"Ground Truth: {ground_truth[:100]}{'...' if len(ground_truth) > 100 else ''}"
            text_draw.text((10, 40), gt_text, fill='green', font=font)
        
        if confidence is not None:
            conf_text = f"Confidence: {confidence:.3f}"
            text_draw.text((10, 70), conf_text, fill='red', font=font)
        
        combined = Image.new('RGB', (img_width, img_height + text_bg_height))
        combined.paste(viz_image, (0, 0))
        combined.paste(text_bg, (0, img_height))
        
        return combined
    
    def create_attention_visualization(self, image: Image.Image, attention_weights: 'np.ndarray', tokens: List[str]) -> Image.Image:
        if not MATPLOTLIB_AVAILABLE:
            return image
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        if attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)
        
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=False, 
                   cmap='Blues', ax=ax2, cbar=True)
        ax2.set_title('Attention Weights')
        ax2.set_xlabel('Tokens')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return Image.fromarray(plot_image)
    
    def finalize_training(self, history: Dict[str, List], metrics_collector=None) -> None:
        if MATPLOTLIB_AVAILABLE:
            # self._create_training_summary(history)
            # Строим график метрик по эпохам
            # if metrics_collector is not None:
            #     self._plot_metrics_by_epoch(metrics_collector)
            # Строим график валидационных метрик
            self._plot_validation_metrics(history)
        logger.info(f"Training visualizations saved to {self.plots_dir}")
    
    def finalize_two_stage_training(self, history: Dict[str, List], metrics_collector=None) -> None:
        if MATPLOTLIB_AVAILABLE:
            # self._create_two_stage_summary(history)
            # Строим график метрик по эпохам
            # if metrics_collector is not None:
            #     self._plot_metrics_by_epoch(metrics_collector)
            # Строим график валидационных метрик
            self._plot_validation_metrics(history)
        logger.info(f"Two-stage training visualizations saved to {self.plots_dir}")
    
    def _create_training_summary(self, history: Dict[str, List]) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = list(range(len(history['train_loss'])))
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'eval_loss' in history and history['eval_loss']:
            eval_epochs = [i * max(1, len(epochs) // len(history['eval_loss'])) 
                          for i in range(len(history['eval_loss']))]
            ax1.plot(eval_epochs, history['eval_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Final Training Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if self.metrics_history:
            avg_loss = sum(m['metrics'].get('loss', 0) for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
            ax2.text(0.1, 0.5, f'Avg Recent Loss:\n{avg_loss:.4f}', transform=ax2.transAxes)
            ax2.set_title('Training Statistics')
            ax2.axis('off')
        
        ax3.hist(history['train_loss'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Training Summary Report', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "training_summary.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()
    
    def _create_two_stage_summary(self, history: Dict[str, List]) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = list(range(len(history['train_loss'])))
        stage_info = history.get('stage_info', [])
        
        synthetic_epochs = [i for i, stage in enumerate(stage_info) if stage == 'synthetic']
        real_epochs = [i for i, stage in enumerate(stage_info) if stage == 'real']
        
        if synthetic_epochs:
            synthetic_losses = [history['train_loss'][i] for i in synthetic_epochs]
            ax1.plot(synthetic_epochs, synthetic_losses, 'b-', label='Synthetic Data', linewidth=2)
        
        if real_epochs:
            real_losses = [history['train_loss'][i] for i in real_epochs]
            ax1.plot(real_epochs, real_losses, 'r-', label='Real Data', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Two-Stage Training Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        stage_stats = {}
        for stage in ['synthetic', 'real']:
            stage_epochs = [i for i, s in enumerate(stage_info) if s == stage]
            if stage_epochs:
                stage_losses = [history['train_loss'][i] for i in stage_epochs]
                stage_stats[stage] = {
                    'avg_loss': sum(stage_losses) / len(stage_losses),
                    'final_loss': stage_losses[-1],
                    'epochs': len(stage_epochs)
                }
        
        if stage_stats:
            stages = list(stage_stats.keys())
            avg_losses = [stage_stats[s]['avg_loss'] for s in stages]
            ax2.bar(stages, avg_losses, alpha=0.7)
            ax2.set_title('Average Loss by Stage')
            ax2.set_ylabel('Average Loss')
        
        colors = {'synthetic': 'blue', 'real': 'red', 'mixed': 'green'}
        stage_colors = [colors.get(stage, 'gray') for stage in stage_info]
        
        ax3.scatter(epochs, history['train_loss'], c=stage_colors, alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Timeline by Stage')
        ax3.grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Synthetic'),
                          Patch(facecolor='red', label='Real'),
                          Patch(facecolor='green', label='Mixed')]
        ax3.legend(handles=legend_elements)
        
        total_epochs = len(epochs)
        synthetic_epochs_count = len(synthetic_epochs)
        real_epochs_count = len(real_epochs)
        
        summary_text = f"""Two-Stage Training Summary:
Total Epochs: {total_epochs}
Synthetic Stage: {synthetic_epochs_count} epochs
Real Stage: {real_epochs_count} epochs"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.suptitle('Two-Stage Training Summary Report', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "two_stage_training_summary.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()
    
    def _plot_metrics_by_epoch(self, metrics_collector=None):
        """Plot all available metrics by epoch."""
        if not MATPLOTLIB_AVAILABLE or metrics_collector is None:
            return

        # Получение всех метрик с помощью нового метода get_all_metrics
        all_metrics = metrics_collector.get_all_metrics()
        
        # Проверка наличия метрик для графика
        if not all_metrics.get('epoch_metrics'):
            logger.warning("No epoch metrics available for plotting")
            return
            
        # Подготавливаем данные для графика из структуры all_metrics
        epochs = all_metrics.get('epochs', [])
        train_losses = all_metrics.get('train_losses', [])
        eval_losses = all_metrics.get('eval_losses', [])
        
        # Удаление None значений из eval_losses
        eval_epochs = [e for e, v in zip(epochs, eval_losses) if v is not None]
        clean_eval_losses = [v for v in eval_losses if v is not None]
        
        # Создаем график с несколькими осями
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # График 1: Потери при обучении и валидации
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
        if clean_eval_losses:
            ax1.plot(eval_epochs, clean_eval_losses, 'r-o', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss by Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График 2: Метрики CER и WER (если доступны)
        ax2 = axes[0, 1]
        
        epoch_inference_metrics = all_metrics.get('epoch_inference_metrics', {})
        if epoch_inference_metrics:
            # Получаем данные для графика CER и WER
            infer_epochs = sorted(epoch_inference_metrics.keys())
            avg_cers = [epoch_inference_metrics[e]['avg_cer'] for e in infer_epochs 
                      if 'avg_cer' in epoch_inference_metrics[e]]
            avg_wers = [epoch_inference_metrics[e]['avg_wer'] for e in infer_epochs 
                      if 'avg_wer' in epoch_inference_metrics[e]]
            
            if avg_cers:
                ax2.plot(infer_epochs, avg_cers, 'g-o', label='CER', linewidth=2)
            if avg_wers:
                ax2.plot(infer_epochs, avg_wers, 'm-o', label='WER', linewidth=2)
                
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Error Rate')
            ax2.set_title('Character and Word Error Rates by Epoch')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No inference metrics available', 
                   horizontalalignment='center', verticalalignment='center')
            ax2.set_title('Inference Metrics')
        
        # График 3: Скорость обучения по эпохам (если доступна)
        ax3 = axes[1, 0]
        
        # Получаем скорость обучения из метрик шагов
        step_metrics = all_metrics.get('step_metrics', [])
        if step_metrics:
            # Группируем метрики шагов по эпохам, беря среднее значение LR для каждой эпохи
            epoch_to_lr = {}
            for sm in step_metrics:
                epoch = sm.get('epoch')
                if epoch is not None:
                    if epoch not in epoch_to_lr:
                        epoch_to_lr[epoch] = []
                    lr = sm.get('learning_rate')
                    if lr is not None:
                        epoch_to_lr[epoch].append(lr)
            
            lr_epochs = sorted(epoch_to_lr.keys())
            avg_lrs = [sum(epoch_to_lr[e])/len(epoch_to_lr[e]) for e in lr_epochs if epoch_to_lr[e]]
            
            if avg_lrs:
                ax3.plot(lr_epochs, avg_lrs, 'c-o', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.set_title('Learning Rate Schedule')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No learning rate data available', 
                       horizontalalignment='center', verticalalignment='center')
                ax3.set_title('Learning Rate')
        else:
            ax3.text(0.5, 0.5, 'No step metrics available', 
                   horizontalalignment='center', verticalalignment='center')
            ax3.set_title('Learning Rate')
            
        # График 4: Сводная информация
        ax4 = axes[1, 1]
        best_metrics = all_metrics.get('best_metrics', {})
        
        summary_text = "Training Summary:\n\n"
        if 'best_loss' in best_metrics:
            summary_text += f"Best Loss: {best_metrics['best_loss']:.4f} (Epoch {best_metrics.get('best_epoch', '?')})\n"
        if 'best_cer' in best_metrics:
            summary_text += f"Best CER: {best_metrics['best_cer']:.4f}\n"
        if 'best_wer' in best_metrics:
            summary_text += f"Best WER: {best_metrics['best_wer']:.4f}\n"
        
        if train_losses:
            summary_text += f"\nInitial Loss: {train_losses[0]:.4f}\n"
            summary_text += f"Final Loss: {train_losses[-1]:.4f}\n"
            summary_text += f"Improvement: {train_losses[0] - train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%)\n"
            
        ax4.text(0.1, 0.5, summary_text, verticalalignment='center')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "metrics_by_epoch.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()
    
    def _plot_validation_metrics(self, history: Dict[str, List]) -> None:
        """Plot validation metrics over epochs."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        if 'eval_loss' not in history or not history['eval_loss']:
            logger.warning("No validation metrics available for plotting")
            return
            
        # Получаем данные метрик
        eval_losses = history.get('eval_loss', [])
        exact_match = history.get('exact_match', [])
        cer = history.get('cer', [])
        wer = history.get('wer', [])
        bleu = history.get('bleu', [])
        rouge_l = history.get('rouge_l', [])
        sequence_accuracy = history.get('sequence_accuracy', [])
        
        # Логируем доступные метрики для отладки
        logger.info(f"Available metrics for plotting: {', '.join([k for k, v in history.items() if v])}")
        logger.info(f"CER data available: {len(cer)}, WER data available: {len(wer)}")
        
        # Другие валидационные метрики
        extra_metrics = {}
        for key, value in history.items():
            if key not in ['eval_loss', 'train_loss', 'learning_rates', 'stage_info', 
                         'exact_match', 'cer', 'wer', 'bleu', 'rouge_l', 'sequence_accuracy'] and value:
                extra_metrics[key] = value
        
        # Список всех доступных метрик, исключая eval_loss
        metrics_data = {
            'exact_match': exact_match,
            'cer': cer,
            'wer': wer,
            'bleu': bleu,
            'rouge_l': rouge_l,
            'sequence_accuracy': sequence_accuracy
        }
        
        # Добавляем дополнительные метрики
        metrics_data.update(extra_metrics)
        
        # Отфильтровываем пустые метрики
        valid_metrics = {k: v for k, v in metrics_data.items() if v}
        
        if not valid_metrics:
            logger.warning("No validation metrics data available")
            return
            
        # Определяем количество эпох
        epochs = list(range(len(eval_losses)))
        
        # Создаем несколько графиков для разных групп метрик, чтобы избежать наложения
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Цвета для графиков
        colors = {
            'exact_match': 'blue',
            'cer': 'green',
            'wer': 'purple',
            'bleu': 'orange',
            'rouge_l': 'brown',
            'sequence_accuracy': 'cyan'
        }
        
        # Группа 1: Метрики точности
        accuracy_metrics = {k: v for k, v in valid_metrics.items() 
                         if k in ['exact_match', 'sequence_accuracy', 'bleu', 'rouge_l']}
        
        # Построение графиков точности на верхнем графике
        for metric_name, metric_values in accuracy_metrics.items():
            display_name = {
                'exact_match': 'Exact Match',
                'sequence_accuracy': 'Sequence Accuracy',
                'bleu': 'BLEU Score',
                'rouge_l': 'ROUGE-L Score'
            }.get(metric_name, metric_name)
            
            axes[0].plot(
                epochs, 
                metric_values, 
                '-o', 
                label=display_name,
                color=colors.get(metric_name, 'blue'),
                linewidth=2,
                markersize=5
            )
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy Value')
        axes[0].set_title('Validation Accuracy Metrics')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.0)
        
        # Группа 2: Метрики ошибок (CER, WER и др.)
        # Принудительно включаем CER и WER если они есть в истории
        error_metrics = {}
        if 'cer' in history and history['cer']:
            error_metrics['cer'] = history['cer']
        if 'wer' in history and history['wer']:
            error_metrics['wer'] = history['wer']
            
        # Добавляем другие метрики ошибок
        for k, v in valid_metrics.items():
            if k.lower().find('error') >= 0 and k not in ['cer', 'wer']:
                error_metrics[k] = v
                
        logger.info(f"Error metrics for plotting: {', '.join(error_metrics.keys())}")
        
        # Построение графиков ошибок на нижнем графике
        if error_metrics:
            for metric_name, metric_values in error_metrics.items():
                display_name = {
                    'cer': 'Character Error Rate (CER)',
                    'wer': 'Word Error Rate (WER)'
                }.get(metric_name, metric_name)
                
                axes[1].plot(
                    epochs, 
                    metric_values, 
                    '-o', 
                    label=display_name,
                    color=colors.get(metric_name, 'gray'),
                    linewidth=2,
                    markersize=5
                )
            
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Error Rate')
            axes[1].set_title('Validation Error Metrics')
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1.1)  # Увеличиваем до 1.1 чтобы точки со значением 1.0 были хорошо видны
        else:
            # Если метрик ошибок нет, выводим информационное сообщение
            axes[1].text(0.5, 0.5, 'No error metrics available', 
                       horizontalalignment='center', 
                       verticalalignment='center',
                       transform=axes[1].transAxes)
            axes[1].set_title('Validation Error Metrics')
            axes[1].axis('on')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Error Rate')
        
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "validation_metrics.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()