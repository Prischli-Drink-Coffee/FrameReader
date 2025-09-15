from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import math
import json

import torch
import numpy as np

try:
    from nltk.metrics import edit_distance
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCalculator:
    def __init__(self, task_type: str = "ocr"):
        self.task_type = task_type
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, some metrics will be approximated")
    
    def calculate_batch_metrics(self, predictions: List[str], targets: List[str], task_type: Optional[str] = None) -> Dict[str, float]:
        """Calculate metrics for a batch of predictions and targets"""
        if task_type is None:
            task_type = self.task_type
            
        if len(predictions) != len(targets):
            logger.warning(f"Mismatch: {len(predictions)} predictions vs {len(targets)} targets")
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]
        
        if not predictions:
            return self._empty_metrics()
        
        # Базовые метрики OCR
        metrics = {
            'exact_match': self._exact_match_accuracy(predictions, targets),
            'cer': self._character_error_rate(predictions, targets),
            'wer': self._word_error_rate(predictions, targets),
            'bleu': self._bleu_score(predictions, targets),
            'rouge_l': self._rouge_l_score(predictions, targets),
            'sequence_accuracy': self._sequence_accuracy(predictions, targets)
        }
        
        # Специализированные метрики для структурированных данных
        if task_type == "structured" or task_type == "donut":
            metrics.update(self._structured_metrics(predictions, targets))
        
        # Метрики качества для длинных последовательностей
        if task_type in ["document", "long_text"]:
            metrics.update(self._document_metrics(predictions, targets))
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary"""
        return {
            'exact_match': 0.0, 
            'cer': 1.0, 
            'wer': 1.0, 
            'bleu': 0.0, 
            'rouge_l': 0.0,
            'sequence_accuracy': 0.0
        }
    
    def _exact_match_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate exact match accuracy"""
        matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
        return matches / len(predictions) if predictions else 0.0
    
    def _sequence_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate sequence-level accuracy (normalized exact match)"""
        exact_matches = 0
        for pred, target in zip(predictions, targets):
            pred_normalized = self._normalize_text(pred)
            target_normalized = self._normalize_text(target)
            if pred_normalized == target_normalized:
                exact_matches += 1
        return exact_matches / len(predictions) if predictions else 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better comparison"""
        import re
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text.strip())
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление знаков препинания для некоторых метрик
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _character_error_rate(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate Character Error Rate (CER)"""
        total_chars = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            pred = pred.strip()
            target = target.strip()
            
            # ИСПРАВЛЯЕМ: проверяем на placeholder и пустые строки
            if target == "[No Ground Truth Available]" or not target:
                # Если нет ground truth, пропускаем этот образец
                continue
                
            if NLTK_AVAILABLE:
                errors = edit_distance(pred, target)
            else:
                errors = self._levenshtein_distance(pred, target)
            
            total_errors += errors
            total_chars += len(target)
        
        # ИСПРАВЛЯЕМ: если нет валидных образцов, возвращаем 1.0
        if total_chars == 0:
            return 1.0
            
        cer = total_errors / total_chars
        return min(cer, 1.0)  # Ограничиваем максимум 1.0
    
    def _word_error_rate(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate Word Error Rate (WER)"""
        total_words = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            pred = pred.strip()
            target = target.strip()
            
            # ИСПРАВЛЯЕМ: проверяем на placeholder и пустые строки
            if target == "[No Ground Truth Available]" or not target:
                # Если нет ground truth, пропускаем этот образец
                continue
            
            pred_words = pred.split()
            target_words = target.split()
            
            if NLTK_AVAILABLE:
                errors = edit_distance(pred_words, target_words)
            else:
                errors = self._levenshtein_distance(pred_words, target_words)
            
            total_errors += errors
            total_words += len(target_words)
        
        # ИСПРАВЛЯЕМ: если нет валидных образцов, возвращаем 1.0
        if total_words == 0:
            return 1.0
            
        wer = total_errors / total_words
        return min(wer, 1.0)  # Ограничиваем максимум 1.0
    
    def _levenshtein_distance(self, s1: Union[str, List], s2: Union[str, List]) -> int:
        """Calculate Levenshtein distance between two sequences"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _bleu_score(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate BLEU score (simplified unigram version)"""
        total_precision = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_tokens = pred.strip().split()
            target_tokens = target.strip().split()
            
            if not pred_tokens or not target_tokens:
                continue
            
            # Подсчет совпадений токенов
            pred_counts = {}
            for token in pred_tokens:
                pred_counts[token] = pred_counts.get(token, 0) + 1
            
            target_counts = {}
            for token in target_tokens:
                target_counts[token] = target_counts.get(token, 0) + 1
            
            matches = 0
            for token, count in pred_counts.items():
                matches += min(count, target_counts.get(token, 0))
            
            precision = matches / len(pred_tokens) if pred_tokens else 0.0
            
            # Brevity penalty
            if len(pred_tokens) < len(target_tokens):
                brevity_penalty = math.exp(1 - len(target_tokens) / max(1, len(pred_tokens)))
                precision *= brevity_penalty
            
            total_precision += precision
        
        return total_precision / len(predictions) if predictions else 0.0
    
    def _rouge_l_score(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate ROUGE-L score"""
        total_score = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_tokens = pred.strip().split()
            target_tokens = target.strip().split()
            
            if not pred_tokens and not target_tokens:
                score = 1.0
            elif not pred_tokens or not target_tokens:
                score = 0.0
            else:
                lcs_length = self._lcs_length(pred_tokens, target_tokens)
                
                if lcs_length == 0:
                    score = 0.0
                else:
                    precision = lcs_length / len(pred_tokens)
                    recall = lcs_length / len(target_tokens)
                    score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            total_score += score
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _structured_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate metrics for structured data (JSON/XML)"""
        json_exact_matches = 0
        valid_json_predictions = 0
        key_accuracy_scores = []
        value_accuracy_scores = []
        
        for pred, target in zip(predictions, targets):
            try:
                # Попытка парсинга как JSON
                pred_data = json.loads(pred) if pred.strip() else {}
                target_data = json.loads(target) if target.strip() else {}
                
                valid_json_predictions += 1
                
                # Точное совпадение структуры
                if pred_data == target_data:
                    json_exact_matches += 1
                
                # Метрики точности ключей и значений
                if isinstance(pred_data, dict) and isinstance(target_data, dict):
                    key_acc = self._calculate_key_accuracy(pred_data, target_data)
                    value_acc = self._calculate_value_accuracy(pred_data, target_data)
                    key_accuracy_scores.append(key_acc)
                    value_accuracy_scores.append(value_acc)
                    
            except (json.JSONDecodeError, TypeError):
                # Если не JSON, пробуем простое сравнение полей
                try:
                    # Попытка извлечь поля из текста
                    pred_fields = self._extract_fields_from_text(pred)
                    target_fields = self._extract_fields_from_text(target)
                    
                    if pred_fields or target_fields:
                        key_acc = self._calculate_key_accuracy(pred_fields, target_fields)
                        value_acc = self._calculate_value_accuracy(pred_fields, target_fields)
                        key_accuracy_scores.append(key_acc)
                        value_accuracy_scores.append(value_acc)
                except:
                    continue
        
        total_predictions = len(predictions)
        json_validity = valid_json_predictions / total_predictions if total_predictions > 0 else 0.0
        json_accuracy = json_exact_matches / valid_json_predictions if valid_json_predictions > 0 else 0.0
        
        result = {
            'json_validity': json_validity,
            'json_accuracy': json_accuracy,
            'structure_exact_match': json_exact_matches / total_predictions if total_predictions > 0 else 0.0
        }
        
        if key_accuracy_scores:
            result['key_accuracy'] = sum(key_accuracy_scores) / len(key_accuracy_scores)
        if value_accuracy_scores:
            result['value_accuracy'] = sum(value_accuracy_scores) / len(value_accuracy_scores)
        
        return result
    
    def _calculate_key_accuracy(self, pred_dict: Dict, target_dict: Dict) -> float:
        """Calculate accuracy of extracted keys"""
        pred_keys = set(pred_dict.keys()) if isinstance(pred_dict, dict) else set()
        target_keys = set(target_dict.keys()) if isinstance(target_dict, dict) else set()
        
        if not target_keys:
            return 1.0 if not pred_keys else 0.0
        
        intersection = pred_keys.intersection(target_keys)
        return len(intersection) / len(target_keys)
    
    def _calculate_value_accuracy(self, pred_dict: Dict, target_dict: Dict) -> float:
        """Calculate accuracy of extracted values"""
        if not isinstance(pred_dict, dict) or not isinstance(target_dict, dict):
            return 0.0
        
        common_keys = set(pred_dict.keys()).intersection(set(target_dict.keys()))
        if not common_keys:
            return 0.0
        
        correct_values = 0
        for key in common_keys:
            pred_val = str(pred_dict[key]).strip().lower()
            target_val = str(target_dict[key]).strip().lower()
            if pred_val == target_val:
                correct_values += 1
        
        return correct_values / len(common_keys)
    
    def _extract_fields_from_text(self, text: str) -> Dict[str, str]:
        """Extract fields from unstructured text"""
        import re
        
        # Простые паттерны для извлечения пар ключ-значение
        patterns = [
            r'(\w+):\s*([^,\n]+)',  # key: value
            r'(\w+)\s*=\s*([^,\n]+)',  # key = value
            r'<s_(\w+)>(.*?)</s_\1>',  # Donut-style tokens
        ]
        
        fields = {}
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                fields[key.strip().lower()] = value.strip()
        
        return fields
    
    def _document_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate metrics specific to document-level OCR"""
        avg_length_accuracy = 0.0
        paragraph_accuracy = 0.0
        
        for pred, target in zip(predictions, targets):
            # Точность длины документа
            pred_len = len(pred.split())
            target_len = len(target.split())
            length_acc = 1.0 - abs(pred_len - target_len) / max(target_len, 1)
            avg_length_accuracy += max(0.0, length_acc)
            
            # Точность абзацев
            pred_paragraphs = [p.strip() for p in pred.split('\n\n') if p.strip()]
            target_paragraphs = [p.strip() for p in target.split('\n\n') if p.strip()]
            
            if target_paragraphs:
                para_matches = 0
                for target_para in target_paragraphs:
                    best_match = 0.0
                    for pred_para in pred_paragraphs:
                        # Используем ROUGE-L для сравнения абзацев
                        target_tokens = target_para.split()
                        pred_tokens = pred_para.split()
                        if target_tokens and pred_tokens:
                            lcs_len = self._lcs_length(pred_tokens, target_tokens)
                            precision = lcs_len / len(pred_tokens)
                            recall = lcs_len / len(target_tokens)
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                            best_match = max(best_match, f1)
                    para_matches += best_match
                
                paragraph_accuracy += para_matches / len(target_paragraphs)
        
        return {
            'length_accuracy': avg_length_accuracy / len(predictions) if predictions else 0.0,
            'paragraph_accuracy': paragraph_accuracy / len(predictions) if predictions else 0.0
        }
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple batches"""
        if not metrics_list:
            return self._empty_metrics()
        
        aggregated = {}
        
        # Собираем все возможные метрики
        metric_names = set()
        for metrics in metrics_list:
            if metrics:
                metric_names.update(metrics.keys())
        
        # Вычисляем среднее для каждой метрики
        for metric_name in metric_names:
            values = []
            for metrics in metrics_list:
                if metrics and metric_name in metrics:
                    value = metrics[metric_name]
                    if not (math.isnan(value) or math.isinf(value)):
                        values.append(value)
            
            if values:
                aggregated[metric_name] = sum(values) / len(values)
            else:
                # Значения по умолчанию для различных типов метрик
                if metric_name in ['cer', 'wer']:
                    aggregated[metric_name] = 1.0
                else:
                    aggregated[metric_name] = 0.0
        
        return aggregated
    
    def format_metrics(self, metrics: Dict[str, float], precision: int = 3) -> str:
        """Format metrics for display"""
        if not metrics:
            return "No metrics available"
        
        formatted_parts = []
        
        # Группируем метрики по типам для лучшего отображения
        primary_metrics = ['exact_match', 'cer', 'wer', 'bleu', 'rouge_l']
        structure_metrics = ['json_validity', 'json_accuracy', 'key_accuracy', 'value_accuracy']
        
        # Основные метрики
        for metric_name in primary_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                formatted_parts.append(f"{metric_name}: {value:.{precision}f}")
        
        # Структурированные метрики
        structure_parts = []
        for metric_name in structure_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                structure_parts.append(f"{metric_name}: {value:.{precision}f}")
        
        if structure_parts:
            formatted_parts.append(f"[{' | '.join(structure_parts)}]")
        
        # Остальные метрики
        other_metrics = [k for k in metrics.keys() 
                        if k not in primary_metrics and k not in structure_metrics]
        
        for metric_name in sorted(other_metrics):
            value = metrics[metric_name]
            formatted_parts.append(f"{metric_name}: {value:.{precision}f}")
        
        return " | ".join(formatted_parts)
    
    def calculate_confidence_intervals(self, metrics_list: List[Dict[str, float]], confidence: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics"""
        import scipy.stats as stats
        
        confidence_intervals = {}
        
        # Получаем все метрики
        all_metrics = set()
        for metrics in metrics_list:
            if metrics:
                all_metrics.update(metrics.keys())
        
        alpha = 1 - confidence
        
        for metric_name in all_metrics:
            values = []
            for metrics in metrics_list:
                if metrics and metric_name in metrics:
                    value = metrics[metric_name]
                    if not (math.isnan(value) or math.isinf(value)):
                        values.append(value)
            
            if len(values) > 1:
                mean = np.mean(values)
                sem = stats.sem(values)  # Standard error of the mean
                
                if len(values) < 30:
                    # t-distribution for small samples
                    t_val = stats.t.ppf(1 - alpha/2, len(values) - 1)
                    margin_error = t_val * sem
                else:
                    # Normal distribution for large samples  
                    z_val = stats.norm.ppf(1 - alpha/2)
                    margin_error = z_val * sem
                
                confidence_intervals[metric_name] = (
                    max(0.0, mean - margin_error),
                    min(1.0, mean + margin_error)
                )
        
        return confidence_intervals
    
    def compare_model_performance(self, model_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare performance across multiple models"""
        if not model_metrics:
            return {}
        
        comparison = {
            'best_model': {},
            'model_rankings': {},
            'performance_gaps': {}
        }
        
        # Получаем все метрики
        all_metrics = set()
        for metrics in model_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Для каждой метрики определяем лучшую модель
        for metric_name in all_metrics:
            metric_values = {}
            for model_name, metrics in model_metrics.items():
                if metric_name in metrics:
                    metric_values[model_name] = metrics[metric_name]
            
            if metric_values:
                # Для ошибок (CER, WER) лучше меньше, для остальных - больше
                if metric_name in ['cer', 'wer']:
                    best_model = min(metric_values.items(), key=lambda x: x[1])
                    rankings = sorted(metric_values.items(), key=lambda x: x[1])
                else:
                    best_model = max(metric_values.items(), key=lambda x: x[1])
                    rankings = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
                comparison['best_model'][metric_name] = best_model[0]
                comparison['model_rankings'][metric_name] = [model for model, _ in rankings]
                
                # Вычисляем разрыв в производительности
                if len(rankings) > 1:
                    best_score = rankings[0][1]
                    worst_score = rankings[-1][1]
                    gap = abs(best_score - worst_score)
                    comparison['performance_gaps'][metric_name] = gap
        
        return comparison