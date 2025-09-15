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
    
    def _plot_performance_analysis(self, history: Dict[str, List]) -> None:
        """
        Создает график с анализом производительности модели, включая скорость обучения,
        динамику метрик и эффективность обучения
        """
        if not MATPLOTLIB_AVAILABLE or not self.metrics_history:
            return
            
        # Собираем данные о производительности из истории метрик
        epochs = [m['epoch'] for m in self.metrics_history]
        
        # Собираем значения метрик
        loss_values = [m['metrics'].get('loss', None) for m in self.metrics_history]
        loss_values = [v for v in loss_values if v is not None]
        
        # Собираем метрики производительности OCR, если они есть
        cer_values = [m['metrics'].get('cer', None) for m in self.metrics_history if 'cer' in m['metrics']]
        wer_values = [m['metrics'].get('wer', None) for m in self.metrics_history if 'wer' in m['metrics']]
        
        # Создаем фигуру
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # График 1: Скорость улучшения лосса
        if len(loss_values) > 1:
            loss_improvements = [loss_values[i-1] - loss_values[i] for i in range(1, len(loss_values))]
            epochs_imp = epochs[1:]
            
            # Скользящее среднее для сглаживания
            if len(loss_improvements) > 5:
                window_size = min(5, len(loss_improvements) // 3)
                smoothed_improvements = self._moving_average(loss_improvements, window_size)
                
                axes[0].plot(epochs_imp, loss_improvements, 'o-', alpha=0.5, label='Per Epoch')
                axes[0].plot(epochs_imp, smoothed_improvements, 'r-', linewidth=2, 
                          label=f'Smoothed (w={window_size})')
                
                # Маркируем точки с наибольшим улучшением
                best_idx = smoothed_improvements.index(max(smoothed_improvements))
                axes[0].annotate(f'Best: {smoothed_improvements[best_idx]:.4f}',
                              xy=(epochs_imp[best_idx], smoothed_improvements[best_idx]),
                              xytext=(epochs_imp[best_idx] + 1, smoothed_improvements[best_idx]),
                              arrowprops=dict(facecolor='black', shrink=0.05, width=1))
            else:
                axes[0].plot(epochs_imp, loss_improvements, 'o-', label='Per Epoch')
                
            axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss Improvement')
            axes[0].set_title('Training Speed (Loss Improvement per Epoch)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Добавляем аннотацию с информацией об улучшении
            total_improvement = loss_values[0] - loss_values[-1]
            avg_improvement = total_improvement / (len(loss_values) - 1)
            
            info_text = f"Total improvement: {total_improvement:.4f}\n"
            info_text += f"Avg per epoch: {avg_improvement:.4f}"
            
            axes[0].annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                          va='bottom', ha='left')
            
        else:
            axes[0].text(0.5, 0.5, 'Not enough data for improvement analysis',
                      transform=axes[0].transAxes, ha='center', va='center')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        
        # График 2: Сравнение метрик CER и WER, если они есть
        if cer_values and wer_values:
            min_len = min(len(cer_values), len(wer_values))
            cer_epochs = epochs[:min_len]
            
            # Нормализуем значения для сравнения на одном графике
            max_cer = max(cer_values[:min_len])
            max_wer = max(wer_values[:min_len])
            norm_cer = [v / max_cer for v in cer_values[:min_len]]
            norm_wer = [v / max_wer for v in wer_values[:min_len]]
            
            axes[1].plot(cer_epochs, norm_cer, 'b-', label=f'CER (max={max_cer:.4f})')
            axes[1].plot(cer_epochs, norm_wer, 'g-', label=f'WER (max={max_wer:.4f})')
            
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Normalized Value')
            axes[1].set_title('CER/WER Comparison (Normalized)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Коэффициент корреляции
            from scipy.stats import pearsonr
            try:
                corr, _ = pearsonr(cer_values[:min_len], wer_values[:min_len])
                axes[1].annotate(f"Correlation: {corr:.3f}", xy=(0.02, 0.02), xycoords='axes fraction',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                              va='bottom', ha='left')
            except:
                pass
        elif len(loss_values) > 3:
            # Если нет CER/WER, но есть лосс, показываем скорость сходимости
            convergence_data = []
            
            # Определяем процентное улучшение от начального лосса
            initial_loss = loss_values[0]
            for i, loss in enumerate(loss_values):
                improvement_pct = (initial_loss - loss) / initial_loss * 100
                convergence_data.append(improvement_pct)
            
            axes[1].plot(epochs, convergence_data, 'purple', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Improvement (%)')
            axes[1].set_title('Convergence Speed (% Improvement)')
            axes[1].grid(True, alpha=0.3)
            
            # Добавляем целевые линии
            for target in [25, 50, 75]:
                axes[1].axhline(y=target, color='gray', linestyle='--', alpha=0.5)
                axes[1].annotate(f"{target}%", xy=(-0.5, target), textcoords="offset points",
                              xytext=(-15, 0), ha='right')
        else:
            axes[1].text(0.5, 0.5, 'Not enough OCR metric data',
                      transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        # График 3: Анализ стабильности обучения
        if len(loss_values) > 3:
            # Рассчитываем волатильность (изменение лосса между соседними эпохами)
            volatility = [abs(loss_values[i] - loss_values[i-1]) for i in range(1, len(loss_values))]
            epochs_vol = epochs[1:]
            
            # Скользящее среднее для сглаживания
            if len(volatility) > 5:
                window_size = min(5, len(volatility) // 3)
                smoothed_vol = self._moving_average(volatility, window_size)
                
                # Двойная шкала: волатильность и тренд лосса
                ax3a = axes[2]
                ax3b = ax3a.twinx()
                
                # Волатильность на основной шкале
                ax3a.bar(epochs_vol, volatility, alpha=0.3, color='gray', label='Volatility')
                ax3a.plot(epochs_vol, smoothed_vol, 'r-', linewidth=2, label=f'Smoothed (w={window_size})')
                
                # Тренд лосса на вторичной шкале
                ax3b.plot(epochs, loss_values, 'b--', alpha=0.7, label='Loss')
                
                ax3a.set_xlabel('Epoch')
                ax3a.set_ylabel('Loss Change (Volatility)')
                ax3b.set_ylabel('Loss Value')
                ax3a.set_title('Training Stability Analysis')
                
                # Объединяем легенды
                lines1, labels1 = ax3a.get_legend_handles_labels()
                lines2, labels2 = ax3b.get_legend_handles_labels()
                ax3a.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                
                ax3a.grid(True, alpha=0.3)
                
                # Аннотация со средней волатильностью
                avg_volatility = sum(volatility) / len(volatility)
                ax3a.annotate(f"Avg volatility: {avg_volatility:.4f}", xy=(0.02, 0.02),
                           xycoords='axes fraction', va='bottom', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                axes[2].bar(epochs_vol, volatility, color='gray')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Loss Change (Volatility)')
                axes[2].set_title('Training Stability')
                axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Not enough data for stability analysis',
                      transform=axes[2].transAxes, ha='center', va='center')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
        
        # График 4: Статистика распределения лосса
        if len(loss_values) > 1:
            axes[3].hist(loss_values, bins=min(20, len(loss_values)), color='blue', alpha=0.7)
            axes[3].set_xlabel('Loss Value')
            axes[3].set_ylabel('Frequency')
            axes[3].set_title('Loss Distribution')
            axes[3].grid(True, alpha=0.3)
            
            # Добавляем статистику
            import numpy as np
            mean_loss = np.mean(loss_values)
            std_loss = np.std(loss_values)
            median_loss = np.median(loss_values)
            
            stats_text = f"Mean: {mean_loss:.4f}\nMedian: {median_loss:.4f}\nStd: {std_loss:.4f}"
            axes[3].annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction', 
                          va='top', ha='right', 
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            axes[3].text(0.5, 0.5, 'Not enough data for distribution analysis',
                      transform=axes[3].transAxes, ha='center', va='center')
            axes[3].set_xticks([])
            axes[3].set_yticks([])
        
        plt.suptitle('Training Performance Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Сохраняем график
        perf_plot_path = self.plots_dir / "performance_analysis.png"
        plt.savefig(
            perf_plot_path,
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()

    def update_training_progress(self, epoch: int, metrics: Dict[str, Any], history: Dict[str, List]) -> None:
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': metrics.get('stage', 'standard')
        })
        
        # Строим графики КАЖДЫЕ несколько эпох или на последней эпохе
        plot_frequency = max(1, self.config.num_epochs // 20)  # Минимум каждые 5% эпох
        
        if epoch % plot_frequency == 0 or epoch == self.config.num_epochs - 1:
            if MATPLOTLIB_AVAILABLE:
                self._plot_training_curves(history)
                # Создаем график метрик, если накоплено достаточно данных (минимум 3 эпохи)
                if len(self.metrics_history) >= 3:
                    self._plot_metrics_history(history)
                
                # Анализ производительности, начиная с 5 эпохи
                if len(self.metrics_history) >= 5:
                    self._plot_performance_analysis(history)
                
                logger.info(f"Training plots updated at epoch {epoch+1}")
            else:
                logger.info(f"Epoch {epoch+1}: Training progress (visualization disabled)")
        
        # Дополнительно строим графики каждые 10 эпох для больших экспериментов
        if epoch % 10 == 0 and epoch > 0:
            if MATPLOTLIB_AVAILABLE:
                self._plot_training_curves(history)
                self._plot_metrics_history(history)
                self._plot_performance_analysis(history)
                logger.info(f"Periodic training plots saved at epoch {epoch+1}")

    def update_two_stage_progress(self, epoch: int, metrics: Dict[str, Any], history: Dict[str, List]) -> None:
        stage = metrics.get('stage', 'unknown')
        
        self.metrics_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'stage': stage
        })
        
        self.stage_history.append(stage)
        
        # Строим графики периодически
        plot_frequency = max(1, self.config.num_epochs // 20)
        
        if epoch % plot_frequency == 0 or epoch == self.config.num_epochs - 1:
            if MATPLOTLIB_AVAILABLE:
                self._plot_two_stage_curves(history)
                
                # Создаем график метрик, если накоплено достаточно данных (минимум 3 эпохи)
                if len(self.metrics_history) >= 3:
                    self._plot_metrics_history(history)
                
                # Анализ производительности, начиная с 5 эпохи
                if len(self.metrics_history) >= 5:
                    self._plot_performance_analysis(history)
                
                logger.info(f"Two-stage training plots updated at epoch {epoch+1}")
        
        # Дополнительно строим графики каждые 10 эпох
        if epoch % 10 == 0 and epoch > 0:
            if MATPLOTLIB_AVAILABLE:
                self._plot_two_stage_curves(history)
                self._plot_metrics_history(history)
                self._plot_performance_analysis(history)
                logger.info(f"Periodic two-stage plots saved at epoch {epoch+1}")

    def finalize_training(self, history: Dict[str, List]) -> None:
        if MATPLOTLIB_AVAILABLE:
            self._create_training_summary(history)
            self._plot_metrics_history(history)
            self._plot_performance_analysis(history)
            logger.info(f"Training visualizations saved to {self.plots_dir}")
        else:
            logger.info(f"Training completed (visualization disabled)")
    
    def finalize_two_stage_training(self, history: Dict[str, List]) -> None:
        if MATPLOTLIB_AVAILABLE:
            self._create_two_stage_summary(history)
            self._plot_metrics_history(history)
            self._plot_performance_analysis(history)
            logger.info(f"Two-stage training visualizations saved to {self.plots_dir}")
        else:
            logger.info(f"Two-stage training completed (visualization disabled)")
    
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
        
        # Улучшенное отображение метрик
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]
            if recent_metrics:
                metrics_text = "Recent Training Metrics:\n\n"
                
                # Средний лосс за последние 10 эпох
                avg_loss = sum(m['metrics'].get('loss', 0) for m in recent_metrics) / len(recent_metrics)
                metrics_text += f"Avg Loss (last 10): {avg_loss:.4f}\n"
                
                # Лучший лосс
                if len(recent_metrics) > 1:
                    best_loss = min(m['metrics'].get('loss', float('inf')) for m in recent_metrics)
                    metrics_text += f"Best Loss: {best_loss:.4f}\n"
                
                # Добавляем метрики OCR, если они есть
                cer_values = [m['metrics'].get('cer', None) for m in recent_metrics if 'cer' in m['metrics']]
                wer_values = [m['metrics'].get('wer', None) for m in recent_metrics if 'wer' in m['metrics']]
                
                if cer_values:
                    avg_cer = sum(cer_values) / len(cer_values)
                    best_cer = min(cer_values)
                    metrics_text += f"Avg CER: {avg_cer:.4f}\n"
                    metrics_text += f"Best CER: {best_cer:.4f}\n"
                    
                if wer_values:
                    avg_wer = sum(wer_values) / len(wer_values)
                    best_wer = min(wer_values)
                    metrics_text += f"Avg WER: {avg_wer:.4f}\n"
                    metrics_text += f"Best WER: {best_wer:.4f}\n"
                
                # Текущий прогресс
                current_epoch = recent_metrics[-1]['epoch'] + 1
                total_epochs = self.config.num_epochs
                progress_pct = (current_epoch / total_epochs) * 100
                metrics_text += f"\nProgress: {current_epoch}/{total_epochs} ({progress_pct:.1f}%)"
                
                # Добавляем дополнительные метрики, если они есть
                last_metrics = recent_metrics[-1]['metrics']
                extra_metrics = [k for k in last_metrics.keys() if k not in ('loss', 'cer', 'wer', 'stage')]
                if extra_metrics:
                    metrics_text += "\n\nOther metrics:"
                    for key in extra_metrics[:3]:  # Ограничиваем до 3 дополнительных метрик
                        value = last_metrics[key]
                        if isinstance(value, (int, float)):
                            metrics_text += f"\n{key}: {value:.4f}"
                        else:
                            metrics_text += f"\n{key}: {value}"
                
                ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
                         verticalalignment='top', horizontalalignment='left',
                         fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, 'No metrics data available', transform=ax4.transAxes,
                        horizontalalignment='center', verticalalignment='center')
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, 'No metrics data available', transform=ax4.transAxes,
                    horizontalalignment='center', verticalalignment='center')
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
            self._plot_metrics_history(history)
            self._plot_performance_analysis(history)
        logger.info(f"Training visualizations saved to {self.plots_dir}")
    
    def finalize_two_stage_training(self, history: Dict[str, List]) -> None:
        if MATPLOTLIB_AVAILABLE:
            self._create_two_stage_summary(history)
            self._plot_metrics_history(history)
            self._plot_performance_analysis(history)
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
    
    def _plot_metrics_history(self, history: Dict[str, List]) -> None:
        """
        Создает график с детальными метриками обучения
        """
        if not MATPLOTLIB_AVAILABLE or not self.metrics_history:
            return
        
        # Собираем доступные метрики
        all_metrics = set()
        for m in self.metrics_history:
            all_metrics.update(m['metrics'].keys())
        
        # Исключаем 'stage', это не метрика, а индикатор этапа
        if 'stage' in all_metrics:
            all_metrics.remove('stage')
        
        # Если метрик нет, выходим
        if not all_metrics:
            return
            
        # Ограничиваем до 6 метрик для лучшей читаемости
        if len(all_metrics) > 6:
            priority_metrics = ['loss', 'cer', 'wer', 'accuracy', 'f1', 'precision', 'recall']
            selected_metrics = [m for m in priority_metrics if m in all_metrics]
            remaining = [m for m in all_metrics if m not in priority_metrics]
            selected_metrics.extend(remaining)
            selected_metrics = selected_metrics[:6]
        else:
            selected_metrics = list(all_metrics)
        
        # Создаем фигуру с графиками для каждой метрики
        n_metrics = len(selected_metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2  # Округляем вверх для нечетного количества
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Создаем график для каждой метрики
        for i, metric_name in enumerate(selected_metrics):
            if i < len(axes):
                ax = axes[i]
                
                # Собираем значения для этой метрики
                epochs = []
                values = []
                
                for m in self.metrics_history:
                    if metric_name in m['metrics']:
                        epochs.append(m['epoch'])
                        values.append(m['metrics'][metric_name])
                
                if values:
                    # Построение основного графика
                    ax.plot(epochs, values, 'o-', label=f'{metric_name}', linewidth=2)
                    
                    # Добавляем скользящее среднее, если точек достаточно
                    if len(values) > 3:
                        window_size = min(5, len(values) // 3)
                        smoothed = self._moving_average(values, window_size)
                        ax.plot(epochs, smoothed, 'r-', label=f'Smoothed (w={window_size})', linewidth=2)
                    
                    # Настройка графика
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric_name.capitalize())
                    ax.set_title(f'{metric_name.capitalize()} vs. Epoch')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                
                    # Добавляем аннотацию с лучшим и последним значением
                    best_idx = values.index(min(values)) if metric_name == 'loss' else values.index(max(values))
                    best_epoch = epochs[best_idx]
                    best_value = values[best_idx]
                    
                    last_value = values[-1]
                    
                    info_text = f"Best: {best_value:.4f} (epoch {best_epoch})\nLast: {last_value:.4f}"
                    ax.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                va='bottom', ha='left')
                else:
                    ax.text(0.5, 0.5, f'No data for {metric_name}', 
                            transform=ax.transAxes, ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        # Скрываем неиспользуемые оси
        for i in range(len(selected_metrics), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Detailed Training Metrics', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Сохраняем график
        metrics_plot_path = self.plots_dir / "training_metrics.png"
        plt.savefig(
            metrics_plot_path,
            dpi=self.plot_config['dpi'],
            bbox_inches=self.plot_config['bbox_inches']
        )
        plt.close()