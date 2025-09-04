"""
Enhanced metrics calculation for OCR training and evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import math

import torch
import numpy as np

try:
    from nltk.metrics import edit_distance
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Comprehensive metrics calculation for OCR tasks."""
    
    def __init__(self):
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, some metrics will be approximated")
    
    def calculate_batch_metrics(
        self,
        predictions: List[str],
        targets: List[str],
        task_type: str = "ocr"
    ) -> Dict[str, float]:
        """Calculate metrics for a batch of predictions."""
        
        if len(predictions) != len(targets):
            logger.warning(f"Mismatch: {len(predictions)} predictions vs {len(targets)} targets")
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]
        
        if not predictions:
            return self._empty_metrics()
        
        metrics = {
            'exact_match': self._exact_match_accuracy(predictions, targets),
            'cer': self._character_error_rate(predictions, targets),
            'wer': self._word_error_rate(predictions, targets),
            'bleu': self._bleu_score(predictions, targets),
            'rouge_l': self._rouge_l_score(predictions, targets)
        }
        
        if task_type == "structured":
            metrics.update(self._structured_metrics(predictions, targets))
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics when no data available."""
        return {
            'exact_match': 0.0,
            'cer': 1.0,
            'wer': 1.0,
            'bleu': 0.0,
            'rouge_l': 0.0
        }
    
    def _exact_match_accuracy(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate exact match accuracy."""
        matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
        return matches / len(predictions) if predictions else 0.0
    
    def _character_error_rate(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate Character Error Rate (CER)."""
        total_chars = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            pred = pred.strip()
            target = target.strip()
            
            if NLTK_AVAILABLE:
                errors = edit_distance(pred, target)
            else:
                errors = self._levenshtein_distance(pred, target)
            
            total_errors += errors
            total_chars += len(target)
        
        return total_errors / max(1, total_chars)
    
    def _word_error_rate(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate Word Error Rate (WER)."""
        total_words = 0
        total_errors = 0
        
        for pred, target in zip(predictions, targets):
            pred_words = pred.strip().split()
            target_words = target.strip().split()
            
            if NLTK_AVAILABLE:
                errors = edit_distance(pred_words, target_words)
            else:
                errors = self._levenshtein_distance(pred_words, target_words)
            
            total_errors += errors
            total_words += len(target_words)
        
        return total_errors / max(1, total_words)
    
    def _levenshtein_distance(self, s1: Union[str, List], s2: Union[str, List]) -> int:
        """Compute Levenshtein distance when NLTK is not available."""
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
        """Calculate approximate BLEU score."""
        total_precision = 0.0
        
        for pred, target in zip(predictions, targets):
            pred_tokens = pred.strip().split()
            target_tokens = target.strip().split()
            
            if not pred_tokens or not target_tokens:
                continue
            
            # Calculate 1-gram precision
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
            
            # Apply brevity penalty
            if len(pred_tokens) < len(target_tokens):
                brevity_penalty = math.exp(1 - len(target_tokens) / max(1, len(pred_tokens)))
                precision *= brevity_penalty
            
            total_precision += precision
        
        return total_precision / len(predictions) if predictions else 0.0
    
    def _rouge_l_score(self, predictions: List[str], targets: List[str]) -> float:
        """Calculate ROUGE-L score based on longest common subsequence."""
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
        """Calculate longest common subsequence length."""
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
        """Calculate metrics specific to structured output (e.g., JSON)."""
        json_exact_matches = 0
        valid_json_predictions = 0
        
        for pred, target in zip(predictions, targets):
            try:
                import json
                pred_data = json.loads(pred) if pred.strip() else {}
                target_data = json.loads(target) if target.strip() else {}
                
                valid_json_predictions += 1
                
                if pred_data == target_data:
                    json_exact_matches += 1
                    
            except json.JSONDecodeError:
                continue
        
        total_predictions = len(predictions)
        json_validity = valid_json_predictions / total_predictions if total_predictions > 0 else 0.0
        json_accuracy = json_exact_matches / valid_json_predictions if valid_json_predictions > 0 else 0.0
        
        return {
            'json_validity': json_validity,
            'json_accuracy': json_accuracy,
            'structure_exact_match': json_exact_matches / total_predictions if total_predictions > 0 else 0.0
        }
    
    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple batches."""
        if not metrics_list:
            return self._empty_metrics()
        
        aggregated = {}
        
        # Get all metric names from first non-empty metrics dict
        metric_names = set()
        for metrics in metrics_list:
            if metrics:
                metric_names.update(metrics.keys())
                break
        
        # Calculate mean for each metric
        for metric_name in metric_names:
            values = [metrics.get(metric_name, 0.0) for metrics in metrics_list if metrics]
            aggregated[metric_name] = sum(values) / len(values) if values else 0.0
        
        return aggregated
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for logging."""
        formatted_parts = []
        
        for metric_name, value in metrics.items():
            if metric_name in ['cer', 'wer']:
                # Lower is better for error rates
                formatted_parts.append(f"{metric_name}: {value:.3f}")
            else:
                # Higher is better for accuracy/precision metrics
                formatted_parts.append(f"{metric_name}: {value:.3f}")
        
        return " | ".join(formatted_parts)