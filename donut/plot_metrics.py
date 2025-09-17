import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_metrics(file_path: Path) -> Dict[str, Any]:
    """Загружает метрики из JSON файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_train_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Предобрабатывает тренировочные метрики: группирует по эпохам, считает среднее и std."""
    train_epochs = np.array(metrics.get('train_epochs', []))
    if len(train_epochs) == 0:
        return {}

    # Округляем эпохи до целого для группировки
    epoch_bins = np.floor(train_epochs).astype(int)

    processed = defaultdict(lambda: defaultdict(list))

    for i, epoch in enumerate(epoch_bins):
        processed[epoch]['train_loss'].append(metrics['train_loss'][i])
        processed[epoch]['train_cer'].append(metrics['train_cer'][i])
        processed[epoch]['train_wer'].append(metrics['train_wer'][i])
        processed[epoch]['learning_rates'].append(metrics['learning_rates'][i])

        if i < len(metrics['rouge_scores']):
            rouge = metrics['rouge_scores'][i]
            for key in ['rouge-2', 'rouge-l']:
                processed[epoch][f'train_{key}'].append(rouge.get(key, 0))

    # Вычисляем среднее и std для каждой эпохи
    result = {
        'epochs': [],
        'train_loss_mean': [], 'train_loss_std': [],
        'train_cer_mean': [], 'train_cer_std': [],
        'train_wer_mean': [], 'train_wer_std': [],
        'learning_rates_mean': [], 'learning_rates_std': [],
        'train_rouge-2_mean': [], 'train_rouge-2_std': [],
        'train_rouge-l_mean': [], 'train_rouge-l_std': []
    }

    for epoch in sorted(processed.keys()):
        result['epochs'].append(epoch)
        for key in ['train_loss', 'train_cer', 'train_wer', 'learning_rates', 'train_rouge-2', 'train_rouge-l']:
            values = processed[epoch][key]
            if values:
                result[f'{key}_mean'].append(np.mean(values))
                result[f'{key}_std'].append(np.std(values))
            else:
                result[f'{key}_mean'].append(0)
                result[f'{key}_std'].append(0)

    return result


def plot_metrics(metrics: Dict[str, Any], output_dir: Path):
    """Строит и сохраняет графики метрик."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.rcParams['font.size'] = 12

    train_processed = preprocess_train_metrics(metrics)

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    # График функции потери
    ax = axes[0, 0]
    if train_processed:
        epochs = train_processed['epochs']
        ax.errorbar(epochs, train_processed['train_loss_mean'], yerr=train_processed['train_loss_std'],
                    label='Train Loss', color='blue', alpha=0.7, capsize=3, ecolor='lightblue')
    if metrics.get('val_epochs') and metrics.get('val_loss'):
        ax.plot(metrics['val_epochs'], metrics['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    # График CER
    ax = axes[0, 1]
    if train_processed:
        ax.errorbar(epochs, train_processed['train_cer_mean'], yerr=train_processed['train_cer_std'],
                    label='Train CER', color='blue', alpha=0.7, capsize=3, ecolor='lightblue')
    if metrics.get('val_epochs') and metrics.get('val_cer'):
        ax.plot(metrics['val_epochs'], metrics['val_cer'], 'r-', label='Val CER', linewidth=2)
    ax.set_title('Character Error Rate (CER)', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CER')
    ax.legend()
    ax.grid(True)

    # График WER
    ax = axes[1, 0]
    if train_processed:
        ax.errorbar(epochs, train_processed['train_wer_mean'], yerr=train_processed['train_wer_std'],
                    label='Train WER', color='blue', alpha=0.7, capsize=3, ecolor='lightblue')
    if metrics.get('val_epochs') and metrics.get('val_wer'):
        ax.plot(metrics['val_epochs'], metrics['val_wer'], 'r-', label='Val WER', linewidth=2)
    ax.set_title('Word Error Rate (WER)', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('WER')
    ax.legend()
    ax.grid(True)

    # График ROUGE метрик
    ax = axes[1, 1]
    if train_processed:
        ax.errorbar(epochs, train_processed['train_rouge-2_mean'], yerr=train_processed['train_rouge-2_std'],
                    label='Train ROUGE-2', color='blue', alpha=0.7, capsize=3, ecolor='lightblue')
        ax.errorbar(epochs, train_processed['train_rouge-l_mean'], yerr=train_processed['train_rouge-l_std'],
                    label='Train ROUGE-L', color='red', alpha=0.7, capsize=3, ecolor='lightcoral')
    if metrics.get('val_epochs') and metrics.get('val_rouge_scores'):
        val_rouge2 = [r.get('rouge-2', 0) for r in metrics['val_rouge_scores']]
        val_rougel = [r.get('rouge-l', 0) for r in metrics['val_rouge_scores']]
        ax.plot(metrics['val_epochs'], val_rouge2, 'b--', label='Val ROUGE-2', linewidth=2)
        ax.plot(metrics['val_epochs'], val_rougel, 'r--', label='Val ROUGE-L', linewidth=2)
    ax.set_title('ROUGE', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    # График Learning Rate
    ax = axes[2, 0]
    if train_processed:
        ax.plot(epochs, train_processed['learning_rates_mean'], 'purple', label='Learning Rate', alpha=0.8)
    ax.set_title('Learning Rate', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LR')
    ax.legend()
    ax.grid(True)

    # Пустой subplot или дополнительный график, если нужно
    axes[2, 1].axis('off')

    plt.tight_layout()

    plot_dir = output_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / "training_metrics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"График сохранен в {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Строит графики обучения на основе файла metrics.json.")
    parser.add_argument('--metrics_file', type=str, required=True,
                        help="Путь к файлу metrics.json.")
    parser.add_argument('--output_dir', type=str, default='./output',
                        help="Директория для сохранения графиков.")

    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        print(f"Файл {metrics_path} не найден.")
        return

    metrics = load_metrics(metrics_path)

    output_dir = Path(args.output_dir)
    plot_metrics(metrics, output_dir)


if __name__ == "__main__":
    main()