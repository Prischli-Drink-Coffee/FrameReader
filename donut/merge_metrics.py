import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_metrics(file_path: Path) -> Dict[str, Any]:
    """Загружает метрики из JSON файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Объединяет список словарей метрик в один, корректируя шаги для последовательности."""
    if not metrics_list:
        return {}

    merged = {
        'train_steps': [],
        'train_epochs': [],
        'train_loss': [],
        'train_cer': [],
        'train_wer': [],
        'learning_rates': [],
        'rouge_scores': [],
        'val_steps': [],
        'val_epochs': [],
        'val_loss': [],
        'val_cer': [],
        'val_wer': [],
        'val_rouge_scores': []
    }

    max_train_step = 0
    max_val_step = 0

    for metrics in metrics_list:
        # Корректировка train_steps
        if 'train_steps' in metrics and metrics['train_steps']:
            adjusted_train_steps = [step + max_train_step for step in metrics['train_steps']]
            merged['train_steps'].extend(adjusted_train_steps)
            max_train_step = max(adjusted_train_steps) if adjusted_train_steps else max_train_step

        # Корректировка val_steps
        if 'val_steps' in metrics and metrics['val_steps']:
            adjusted_val_steps = [step + max_val_step for step in metrics['val_steps']]
            merged['val_steps'].extend(adjusted_val_steps)
            max_val_step = max(adjusted_val_steps) if adjusted_val_steps else max_val_step

        # Остальные метрики просто расширяем
        for key in ['train_epochs', 'train_loss', 'train_cer', 'train_wer', 'learning_rates', 'rouge_scores',
                    'val_epochs', 'val_loss', 'val_cer', 'val_wer', 'val_rouge_scores']:
            if key in metrics:
                merged[key].extend(metrics[key])

    return merged


def main():
    parser = argparse.ArgumentParser(description="Объединяет файлы metrics.json из нескольких экспериментов.")
    parser.add_argument('--experiment_dirs', nargs='+', required=True,
                        help="Список путей к папкам экспериментов в правильном порядке.")
    parser.add_argument('--output_file', type=str, default='merged_metrics.json',
                        help="Путь к выходному файлу с объединенными метриками.")

    args = parser.parse_args()

    metrics_list = []
    for exp_dir in args.experiment_dirs:
        metrics_path = Path(exp_dir) / 'metrics.json'
        if metrics_path.exists():
            metrics = load_metrics(metrics_path)
            metrics_list.append(metrics)
            print(f"Загружены метрики из {metrics_path}")
        else:
            print(f"Файл {metrics_path} не найден, пропускаем.")

    if not metrics_list:
        print("Нет метрик для объединения.")
        return

    merged_metrics = merge_metrics(metrics_list)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_metrics, f, indent=2, ensure_ascii=False)

    print(f"Объединенные метрики сохранены в {output_path}")


if __name__ == "__main__":
    main()