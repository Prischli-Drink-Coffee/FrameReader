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
    
    def update_training_progress(self, epoch: int, metrics: Dict[str, Any], history: Dict[str, List]) -> None:
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': metrics.get('stage', 'standard')
        })
        
        if epoch % max(1, self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs - 1:
            if MATPLOTLIB_AVAILABLE:
                self._plot_training_curves(history)
            else:
                logger.info(f"Epoch {epoch}: Training progress (visualization disabled)")
    
    def update_two_stage_progress(self, epoch: int, metrics: Dict[str, Any], history: Dict[str, List]) -> None:
        stage = metrics.get('stage', 'unknown')
        
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': stage
        })
        
        self.stage_history.append(stage)
        
        if epoch % max(1, self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs - 1:
            if MATPLOTLIB_AVAILABLE:
                self._plot_two_stage_curves(history)
    
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
    
    def finalize_training(self, history: Dict[str, List]) -> None:
        if MATPLOTLIB_AVAILABLE:
            self._create_training_summary(history)
        logger.info(f"Training visualizations saved to {self.plots_dir}")
    
    def finalize_two_stage_training(self, history: Dict[str, List]) -> None:
        if MATPLOTLIB_AVAILABLE:
            self._create_two_stage_summary(history)
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