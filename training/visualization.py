"""
Comprehensive visualization system for OCR training and results.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from core.config import TrainingConfig

logger = logging.getLogger(__name__)

# Configure matplotlib and seaborn
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingVisualizer:
    """Advanced visualization for training progress and results."""
    
    def __init__(self, output_dir: Path, config: TrainingConfig):
        self.output_dir = output_dir
        self.config = config
        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Training history storage
        self.metrics_history = []
        self.stage_history = []
        
        # Visualization configuration
        self.plot_config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'format': 'png',
            'bbox_inches': 'tight'
        }
    
    def update_training_progress(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        history: Dict[str, List]
    ) -> None:
        """Update training progress visualization."""
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': metrics.get('stage', 'standard')
        })
        
        # Plot every N epochs or at the end
        if epoch % max(1, self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs - 1:
            self._plot_training_curves(history)
    
    def update_two_stage_progress(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        history: Dict[str, List]
    ) -> None:
        """Update two-stage training progress visualization."""
        stage = metrics.get('stage', 'unknown')
        
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': stage
        })
        
        self.stage_history.append(stage)
        
        # Plot every N epochs
        if epoch % max(1, self.config.num_epochs // 10) == 0 or epoch == self.config.num_epochs - 1:
            self._plot_two_stage_curves(history)
    
    def _plot_training_curves(self, history: Dict[str, List]) -> None:
        """Plot standard training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.plot_config['figure_size'])
        
        epochs = list(range(len(history['train_loss'])))
        
        # Training Loss
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
        
        # Learning Rate
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Loss Smoothed (rolling average)
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
        
        # Training Statistics
        if self.metrics_history:
            recent_metrics = [m['metrics'] for m in self.metrics_history[-10:]]  # Last 10 epochs
            if recent_metrics:
                # Extract common metrics
                metric_names = ['loss']
                metric_values = []
                
                for name in metric_names:
                    values = [m.get(name, 0) for m in recent_metrics if name in m]
                    if values:
                        metric_values.append(values[-1])  # Latest value
                    else:
                        metric_values.append(0)
                
                bars = ax4.bar(metric_names, metric_values, color=['blue'])
                ax4.set_ylabel('Value')
                ax4.set_title('Latest Training Metrics')
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "training_curves.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()
    
    def _plot_two_stage_curves(self, history: Dict[str, List]) -> None:
        """Plot two-stage training curves with stage visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        epochs = list(range(len(history['train_loss'])))
        stage_info = history.get('stage_info', [])
        
        # Training Loss with Stage Coloring
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
        
        # Add stage transition vertical lines
        if synthetic_epochs and real_epochs:
            transition_epoch = max(synthetic_epochs) + 0.5
            ax1.axvline(x=transition_epoch, color='black', linestyle='--', alpha=0.7, 
                       label='Stage Transition')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Two-Stage Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning Rate with Stage Information
        colors = ['blue' if stage == 'synthetic' else 'red' if stage == 'real' else 'green' 
                 for stage in stage_info]
        
        for i, (epoch, lr, color) in enumerate(zip(epochs, history['learning_rates'], colors)):
            ax2.scatter(epoch, lr, c=color, alpha=0.7, s=20)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate by Training Stage')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Create custom legend for stages
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Synthetic'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Real'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Mixed')
        ]
        ax2.legend(handles=legend_elements)
        
        # Stage Distribution Pie Chart
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
        
        # Loss Improvement Analysis
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
        """Calculate moving average."""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i + 1]
            smoothed.append(sum(window_data) / len(window_data))
        
        return smoothed
    
    def visualize_inference_results(
        self,
        image: Union[Image.Image, np.ndarray],
        prediction: str,
        ground_truth: Optional[str] = None,
        confidence: Optional[float] = None,
        bounding_boxes: Optional[List[Dict]] = None
    ) -> Image.Image:
        """Visualize inference results with bounding boxes and text."""
        
        if isinstance(image, np.ndarray):
            if OPENCV_AVAILABLE:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Create a copy to draw on
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
            small_font = font
        
        # Draw bounding boxes if provided
        if bounding_boxes:
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = bbox.get('coords', [0, 0, 0, 0])
                bbox_confidence = bbox.get('confidence', 1.0)
                
                # Color based on confidence
                if bbox_confidence > 0.8:
                    color = 'green'
                elif bbox_confidence > 0.5:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw confidence score
                conf_text = f"{bbox_confidence:.2f}"
                draw.text((x1, y1-20), conf_text, fill=color, font=small_font)
        
        # Add text annotations at the bottom
        img_width, img_height = viz_image.size
        
        # Background for text
        text_bg_height = 120
        text_bg = Image.new('RGB', (img_width, text_bg_height), color='white')
        text_draw = ImageDraw.Draw(text_bg)
        
        # Add prediction text
        pred_text = f"Prediction: {prediction[:100]}{'...' if len(prediction) > 100 else ''}"
        text_draw.text((10, 10), pred_text, fill='blue', font=font)
        
        # Add ground truth if available
        if ground_truth is not None:
            gt_text = f"Ground Truth: {ground_truth[:100]}{'...' if len(ground_truth) > 100 else ''}"
            text_draw.text((10, 40), gt_text, fill='green', font=font)
        
        # Add confidence if available
        if confidence is not None:
            conf_text = f"Confidence: {confidence:.3f}"
            text_draw.text((10, 70), conf_text, fill='red', font=font)
        
        # Combine image and text
        combined = Image.new('RGB', (img_width, img_height + text_bg_height))
        combined.paste(viz_image, (0, 0))
        combined.paste(text_bg, (0, img_height))
        
        return combined
    
    def create_attention_visualization(
        self,
        image: Image.Image,
        attention_weights: np.ndarray,
        tokens: List[str]
    ) -> Image.Image:
        """Create attention heatmap visualization."""
        
        # Create attention heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Show attention heatmap
        # Average attention across heads if multiple
        if attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)
        
        # Create heatmap
        sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=False, 
                   cmap='Blues', ax=ax2, cbar=True)
        ax2.set_title('Attention Weights')
        ax2.set_xlabel('Tokens')
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        
        return Image.fromarray(plot_image)
    
    def finalize_training(self, history: Dict[str, List]) -> None:
        """Create final training summary visualization."""
        self._create_training_summary(history)
        logger.info(f"Training visualizations saved to {self.plots_dir}")
    
    def finalize_two_stage_training(self, history: Dict[str, List]) -> None:
        """Create final two-stage training summary."""
        self._create_two_stage_summary(history)
        logger.info(f"Two-stage training visualizations saved to {self.plots_dir}")
    
    def _create_training_summary(self, history: Dict[str, List]) -> None:
        """Create comprehensive training summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = list(range(len(history['train_loss'])))
        
        # Final loss curves
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
        
        # Training statistics
        if self.metrics_history:
            final_loss = history['train_loss'][-1] if history['train_loss'] else 0
            min_loss = min(history['train_loss']) if history['train_loss'] else 0
            
            stats_text = f"""Training Summary:
Final Loss: {final_loss:.4f}
Best Loss: {min_loss:.4f}
Total Epochs: {len(epochs)}
Improvement: {((history['train_loss'][0] - final_loss) / history['train_loss'][0] * 100):.1f}%"""
            
            ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
            ax2.set_title('Training Statistics')
            ax2.axis('off')
        
        # Loss distribution histogram
        ax3.hist(history['train_loss'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Learning rate schedule
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
        """Create comprehensive two-stage training summary."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = list(range(len(history['train_loss'])))
        stage_info = history.get('stage_info', [])
        
        # Stage-separated loss curves
        synthetic_epochs = [i for i, stage in enumerate(stage_info) if stage == 'synthetic']
        real_epochs = [i for i, stage in enumerate(stage_info) if stage == 'real']
        
        if synthetic_epochs:
            synthetic_losses = [history['train_loss'][i] for i in synthetic_epochs]
            ax1.plot(synthetic_epochs, synthetic_losses, 'b-', label='Synthetic Stage', linewidth=2, marker='o')
        
        if real_epochs:
            real_losses = [history['train_loss'][i] for i in real_epochs]
            ax1.plot(real_epochs, real_losses, 'r-', label='Real Data Stage', linewidth=2, marker='s')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Two-Stage Training Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Stage comparison statistics
        stage_stats = {}
        for stage in ['synthetic', 'real']:
            stage_epochs = [i for i, s in enumerate(stage_info) if s == stage]
            if stage_epochs:
                stage_losses = [history['train_loss'][i] for i in stage_epochs]
                stage_stats[stage] = {
                    'mean_loss': np.mean(stage_losses),
                    'final_loss': stage_losses[-1],
                    'improvement': (stage_losses[0] - stage_losses[-1]) / stage_losses[0] * 100 if len(stage_losses) > 1 else 0
                }
        
        # Bar chart of stage performance
        if stage_stats:
            stages = list(stage_stats.keys())
            mean_losses = [stage_stats[stage]['mean_loss'] for stage in stages]
            final_losses = [stage_stats[stage]['final_loss'] for stage in stages]
            
            x = np.arange(len(stages))
            width = 0.35
            
            ax2.bar(x - width/2, mean_losses, width, label='Mean Loss', alpha=0.7)
            ax2.bar(x + width/2, final_losses, width, label='Final Loss', alpha=0.7)
            
            ax2.set_xlabel('Training Stage')
            ax2.set_ylabel('Loss')
            ax2.set_title('Stage Performance Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(stages)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Timeline visualization
        colors = {'synthetic': 'blue', 'real': 'red', 'mixed': 'green'}
        stage_colors = [colors.get(stage, 'gray') for stage in stage_info]
        
        ax3.scatter(epochs, history['train_loss'], c=stage_colors, alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Timeline by Stage')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Synthetic'),
                          Patch(facecolor='red', label='Real'),
                          Patch(facecolor='green', label='Mixed')]
        ax3.legend(handles=legend_elements)
        
        # Summary statistics
        total_epochs = len(epochs)
        synthetic_epochs_count = len(synthetic_epochs)
        real_epochs_count = len(real_epochs)
        
        summary_text = f"""Two-Stage Training Summary:
Total Epochs: {total_epochs}
Synthetic Stage: {synthetic_epochs_count} epochs ({synthetic_epochs_count/total_epochs*100:.1f}%)
Real Data Stage: {real_epochs_count} epochs ({real_epochs_count/total_epochs*100:.1f}%)

Performance:"""
        
        if stage_stats:
            for stage, stats in stage_stats.items():
                summary_text += f"\n{stage.title()}: {stats['final_loss']:.4f} (↓{stats['improvement']:.1f}%)"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
        ax4.set_title('Two-Stage Summary')
        ax4.axis('off')
        
        plt.suptitle('Two-Stage Training Summary Report', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "two_stage_summary.png",
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()