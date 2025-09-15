"""
Real-time inference utilities during training.
Incorporates TextCleanup and DonutInferenceEngine from train_donut/inference.py
"""

import logging
import re
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from ast import literal_eval

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, realtime inference visualization will be disabled")

try:
    from nltk import edit_distance
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not available, some text comparison features will be disabled")

logger = logging.getLogger(__name__)


class TextCleanup:
    """Text cleanup utilities for OCR output processing."""
    
    @staticmethod
    def cleanup_donut_output(text: Any) -> str:
        """Clean up Donut model output by removing special tokens."""
        text_str = str(text if text is not None else "")
        text_str = re.sub(r"<s_([^>]*)>", "", text_str)
        text_str = re.sub(r"</s_[^>]*>", "", text_str)
        text_str = text_str.replace("<sep/>", ", ")
        text_str = re.sub(r"\s+", " ", text_str).strip()
        return text_str

    @staticmethod
    def extract_fields_from_donut_output(text: Any) -> Dict:
        """Extract structured fields from Donut output."""
        output = {}
        # Ensure text is a string
        processed_text = str(text if text is not None else "")
        
        original_text_for_fallback = processed_text

        while processed_text:
            match = re.search(r"<s_(.*?)>", processed_text, re.IGNORECASE)
            if not match:
                break 

            key = match.group(1)
            original_start_token_val = match.group(0)
            
            content_block_start = match.end(0)
            
            # Simple regex match for the content of the current key
            start_token_escaped = re.escape(original_start_token_val)
            end_token_escaped = re.escape(fr"</s_{key}>")

            content_re_match = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", processed_text, re.IGNORECASE | re.DOTALL)

            if content_re_match:
                content = content_re_match.group(1).strip()
                # Advance processed_text past this entire matched block
                processed_text = processed_text[content_re_match.end(0):].strip()
            else:
                # No matching end token found for this start token, remove start token and continue
                processed_text = processed_text.replace(original_start_token_val, "", 1).strip()
                continue

            # Recursive call for nested structures
            if r"<s_" in content and r"</s_" in content:
                value = TextCleanup.extract_fields_from_donut_output(content)
            else:
                value_parts = [part.strip() for part in content.split(r"<sep/>") if part.strip()]
                if not value_parts:
                    value = ""
                elif len(value_parts) == 1:
                    value = value_parts[0]
                else:
                    value = value_parts

            if value or isinstance(value, str):
                if key in output:
                    if not isinstance(output[key], list):
                        output[key] = [output[key]]
                    
                    if isinstance(value, list):
                        output[key].extend(value)
                    else:
                        output[key].append(value)
                else:
                    output[key] = value
        
        if not output and original_text_for_fallback.strip():
            cleaned_fallback = TextCleanup.cleanup_donut_output(original_text_for_fallback.strip())
            if cleaned_fallback:
                return {"text_sequence": cleaned_fallback}
        return output


def calculate_cer(prediction: str, reference: str) -> float:
    """Calculate Character Error Rate (CER)."""
    if not isinstance(prediction, str): 
        prediction = str(prediction)
    if not isinstance(reference, str): 
        reference = str(reference)

    if not reference:
        return 1.0 if prediction else 0.0
    if not prediction:
        return 1.0

    if NLTK_AVAILABLE:
        distance = edit_distance(prediction, reference)
    else:
        # Fallback: simple character-by-character comparison
        distance = _levenshtein_distance(prediction, reference)
    
    return min(distance / len(reference), 1.0)  # Ограничиваем максимум 1.0


def calculate_wer(prediction: str, reference: str) -> float:
    """Calculate Word Error Rate (WER)."""
    if not isinstance(prediction, str): 
        prediction = str(prediction)
    if not isinstance(reference, str): 
        reference = str(reference)

    pred_words = prediction.split()
    ref_words = reference.split()

    if not ref_words:
        return 1.0 if pred_words else 0.0
    if not pred_words:
        return 1.0

    if NLTK_AVAILABLE:
        distance = edit_distance(pred_words, ref_words)
    else:
        # Fallback: simple word-by-word comparison
        distance = _levenshtein_distance(pred_words, ref_words)
    
    return min(distance / len(ref_words), 1.0)  # Ограничиваем максимум 1.0


def _levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two sequences (fallback implementation)."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
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


class RealtimeInferenceEngine:
    """
    Real-time inference engine for displaying model predictions during training.
    Based on DonutInferenceEngine from train_donut/inference.py
    """
    
    def __init__(
        self,
        model,
        device: Optional[Union[str, torch.device]] = None,
        task_start_token: Optional[str] = "<s_ocr>", 
        prompt_end_token: Optional[str] = "<s_prompt>",
        precision: str = "bf16", 
        max_length: int = 64,   
        num_beams: int = 1
    ):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.precision = precision
        self.max_length = max_length
        self.num_beams = num_beams
        
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token
        
        if hasattr(self.model, 'processor') and hasattr(self.model.processor, 'tokenizer'):
            tokenizer = self.model.processor.tokenizer
            self.eos_token_str_for_cleanup = getattr(tokenizer, 'eos_token', "</s>")
            if not isinstance(self.eos_token_str_for_cleanup, str): 
                self.eos_token_str_for_cleanup = "</s>"
        else:
            self.eos_token_str_for_cleanup = "</s>"

        logger.info(f"Initialized realtime inference engine with max_length={max_length}, num_beams={num_beams}")
    
    def prepare_prompt(self, prompt: Optional[str] = None) -> str:
        """Prepare prompt for model input."""
        if prompt is None or prompt.strip() == "":
            return self.task_start_token
        else:
            _prompt = prompt.strip()
            return f"{self.task_start_token}{_prompt}{self.prompt_end_token}"

    def quick_inference(
        self, 
        image: Union[Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        return_json: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Quick inference for real-time display during training.
        Optimized for speed over accuracy.
        """
        
        if not hasattr(self.model, 'processor'):
            logger.error("Model processor not available for realtime inference")
            return {"error": "Processor not available"} if return_json else "Error: Processor not available"
        
        processor = self.model.processor
        
        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    pixel_values = image.unsqueeze(0).to(self.device)
                else:
                    pixel_values = image.to(self.device)
            else:
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(self.device)

        except Exception as e:
            logger.error(f"Error processing image for realtime inference: {e}")
            return {"error": f"Image processing error: {e}"} if return_json else f"Error: Image processing error"

        self.model.eval()
        with torch.no_grad():
            try:
                encoder_outputs = self.model.encoder(pixel_values)
                
                task_start_token = self.task_start_token or "<s_ocr>"
                if hasattr(processor.tokenizer, 'bos_token_id'):
                    initial_token_id = processor.tokenizer.bos_token_id
                else:
                    initial_tokens = processor.tokenizer(
                        task_start_token,
                        add_special_tokens=False,
                        return_tensors="pt"
                    )["input_ids"]
                    initial_token_id = initial_tokens[0, 0].item() if initial_tokens.numel() > 0 else 0
                
                generated_tokens = torch.tensor([[initial_token_id]], device=self.device)
                
                for step in range(self.max_length - 1):
                    try:
                        decoder_outputs = self.model.decoder(
                            input_ids=generated_tokens,
                            encoder_hidden_states=encoder_outputs
                        )
                        
                        next_token_logits = decoder_outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                        
                        if hasattr(processor.tokenizer, 'eos_token_id') and next_token.item() == processor.tokenizer.eos_token_id:
                            break
                            
                        if hasattr(processor.tokenizer, 'pad_token_id') and next_token.item() == processor.tokenizer.pad_token_id:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error in decode step {step}: {e}")
                        break
                
                raw_text_output = processor.tokenizer.decode(
                    generated_tokens[0], 
                    skip_special_tokens=True
                )

                # logger.info(f"Realtime inference raw output: {raw_text_output}")

            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                raw_text_output = ""

        if return_json:
            try:
                result = TextCleanup.extract_fields_from_donut_output(raw_text_output)
                if not result and raw_text_output.strip():
                    result = {"text_sequence": TextCleanup.cleanup_donut_output(raw_text_output)}
                return result or {"text_sequence": ""}
            except Exception as e:
                logger.warning(f"Error parsing output to JSON: {e}")
                return {"text_sequence": TextCleanup.cleanup_donut_output(raw_text_output)}
        else:
            # logger.info(f"Realtime inference cleaned output: {TextCleanup.cleanup_donut_output(raw_text_output)}")
            return TextCleanup.cleanup_donut_output(raw_text_output)
    
    def _simple_greedy_decode(self, encoder_outputs, prompt_tokens, tokenizer, max_steps=20):
        """Simple greedy decoding for real-time inference."""
        batch_size = encoder_outputs.size(0)
        generated_tokens = prompt_tokens.clone()
        
        for _ in range(max_steps):
            try:
                decoder_outputs = self.model.decoder(
                    input_ids=generated_tokens,
                    encoder_hidden_states=encoder_outputs
                )
                
                next_token_logits = decoder_outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
                
                if next_tokens.item() == tokenizer.eos_token_id:
                    break
                    
            except Exception as e:
                logger.debug(f"Error in greedy decode step: {e}")
                break
        
        return generated_tokens

    def compare_prediction_with_ground_truth(
        self,
        image: Union[Image.Image, torch.Tensor],
        ground_truth: str,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare model prediction with ground truth for training monitoring.
        Returns metrics and formatted text for display.
        """
        
        prediction = self.quick_inference(image, prompt, return_json=False)
        
        if isinstance(prediction, dict) and "error" in prediction:
            return {
                "prediction": str(prediction),
                "ground_truth": ground_truth,
                "cer": 1.0,
                "wer": 1.0,
                "status": "error"
            }
        
        prediction_str = str(prediction)
        ground_truth_str = str(ground_truth)

        # logger.info(f"Prediction: '{prediction_str}' | Ground Truth: '{ground_truth_str}'")
        
        cer = calculate_cer(prediction_str, ground_truth_str)
        wer = calculate_wer(prediction_str, ground_truth_str)
        
        return {
            "prediction": prediction_str,
            "ground_truth": ground_truth_str,
            "cer": cer,
            "wer": wer,
            "status": "success"
        }

    def format_training_display(
        self,
        comparison_result: Dict[str, Any],
        epoch: int,
        step: int,
        loss: float
    ) -> str:
        """Format comparison result for training display."""
        
        prediction = comparison_result.get("prediction", "")
        ground_truth = comparison_result.get("ground_truth", "")
        cer = comparison_result.get("cer", 1.0)
        wer = comparison_result.get("wer", 1.0)
        status = comparison_result.get("status", "unknown")
        
        # Truncate long texts for display
        max_len = 60
        pred_display = prediction[:max_len] + "..." if len(prediction) > max_len else prediction
        gt_display = ground_truth[:max_len] + "..." if len(ground_truth) > max_len else ground_truth
        
        if status == "error":
            return f"[E{epoch:03d}|S{step:05d}|L{loss:.4f}] INFERENCE ERROR: {pred_display}"
        
        return (
            f"[E{epoch:03d}|S{step:05d}|L{loss:.4f}] "
            f"CER:{cer:.3f} WER:{wer:.3f} | "
            f"PRED: {pred_display} | GT: {gt_display}"
        )


class TrainingInferenceDisplayer:
    """Manages real-time inference display during training."""
    
    def __init__(
        self,
        inference_engine: RealtimeInferenceEngine,
        display_interval: int = 100,
        max_history: int = 1000
    ):
        self.inference_engine = inference_engine
        self.display_interval = display_interval
        self.max_history = max_history
        
        self.inference_history = []
        self.step_counter = 0
        
    def should_display(self, step: int) -> bool:
        """Check if inference should be displayed for this step."""
        return step % self.display_interval == 0
    
    def display_inference(
        self,
        batch_data: Dict[str, torch.Tensor],
        epoch: int,
        step: int,
        loss: float,
        sample_idx: int = 0
    ) -> Optional[str]:
        """
        Display model inference for a sample from the batch.
        Returns formatted display string if successful.
        """
        
        if not self.should_display(step):
            return None
            
        try:
            # Extract sample from batch
            pixel_values = batch_data.get('pixel_values', None)
            labels = batch_data.get('labels', None)
            texts = batch_data.get('texts', None)  # Получаем тексты из батча
            
            # ОТЛАДКА: Подробное логирование структуры батча
            logger.warning(f"Batch keys: {list(batch_data.keys())}")
            logger.warning(f"Texts available: {texts is not None}")
            if texts is not None:
                logger.warning(f"Texts type: {type(texts)}, length: {len(texts)}")
                if len(texts) > 0:
                    text_sample = texts[0]
                    logger.warning(f"Sample text type: {type(text_sample)}, content: '{text_sample}'")
                    # Проверка на escape последовательности
                    try:
                        decoded = text_sample.encode().decode('unicode_escape')
                        logger.warning(f"Unicode decoded: '{decoded}'")
                    except:
                        logger.warning(f"Failed to decode unicode escapes")
            
            if pixel_values is None:
                logger.warning("No pixel_values found in batch for inference display")
                return None
                
            # Get single sample
            if pixel_values.dim() == 4:  # Batch dimension exists
                sample_image = pixel_values[sample_idx] if sample_idx < pixel_values.shape[0] else pixel_values[0]
            else:
                sample_image = pixel_values
                
            # Get ground truth if available
            ground_truth = ""
            
            # Сначала пробуем получить из texts (если есть)
            if texts is not None and len(texts) > sample_idx:
                raw_text = texts[sample_idx]
                
                # Обработка строки с учетом возможных escape-последовательностей
                if isinstance(raw_text, str):
                    try:
                        # Пробуем декодировать unicode escape-последовательности
                        ground_truth = raw_text.encode().decode('unicode_escape')
                    except:
                        ground_truth = raw_text
                else:
                    ground_truth = str(raw_text)
                    
                logger.warning(f"Got ground truth from texts: '{ground_truth}'")
            
            # Если texts пустые или отсутствуют, пробуем декодировать labels
            if not ground_truth and labels is not None and hasattr(self.inference_engine.model, 'processor'):
                try:
                    if labels.dim() == 2:  # Batch dimension exists
                        sample_labels = labels[sample_idx] if sample_idx < labels.shape[0] else labels[0]
                    else:
                        sample_labels = labels
                    
                    # Используем более надежную логику декодирования labels
                    pad_token_id = self.inference_engine.model.processor.tokenizer.pad_token_id
                    ignore_idx = -100
                    
                    # Создаем копию для безопасности
                    decode_tokens = sample_labels.clone()
                    
                    # Заменяем ignored tokens на pad tokens для корректного декодирования
                    if pad_token_id is not None:
                        decode_tokens[decode_tokens == ignore_idx] = pad_token_id
                        
                    # Декодируем только если есть хотя бы один не-pad токен
                    valid_tokens = (decode_tokens != pad_token_id).sum().item() if pad_token_id is not None else len(decode_tokens)
                    if valid_tokens > 0:
                        decoded_text = self.inference_engine.model.processor.tokenizer.decode(
                            decode_tokens, skip_special_tokens=True
                        )
                        ground_truth = decoded_text.strip()
                        logger.warning(f"Got ground truth from labels: '{ground_truth}'")
                        
                except Exception as e:
                    logger.warning(f"Could not decode ground truth labels: {e}")
            
            # Если все еще пустая, проверяем другие возможные поля батча
            if not ground_truth and isinstance(batch_data, dict):
                for key in batch_data.keys():
                    if key.lower().find('text') >= 0 or key.lower().find('gt') >= 0 or key.lower().find('ground') >= 0:
                        value = batch_data[key]
                        if isinstance(value, (list, tuple)) and len(value) > sample_idx:
                            potential_text = value[sample_idx]
                            if isinstance(potential_text, str) and potential_text.strip():
                                ground_truth = potential_text.strip()
                                logger.warning(f"Found ground truth in field '{key}': '{ground_truth}'")
                                break
            
            # Если все еще пустая, используем placeholder
            if not ground_truth:
                ground_truth = ""
                logger.warning("No ground truth found, using empty string")
            
            # Get prediction
            comparison = self.inference_engine.compare_prediction_with_ground_truth(
                sample_image, ground_truth
            )
            
            # Format and display
            display_text = self.inference_engine.format_training_display(
                comparison, epoch, step, loss
            )
            
            # Store in history
            self.inference_history.append({
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'comparison': comparison,
                'timestamp': time.time()
            })
            
            # Trim history if too long
            if len(self.inference_history) > self.max_history:
                self.inference_history = self.inference_history[-self.max_history:]
            
            logger.info(display_text)
            return display_text
            
        except Exception as e:
            logger.error(f"Error during inference display: {e}")
            return f"[E{epoch:03d}|S{step:05d}|L{loss:.4f}] INFERENCE DISPLAY ERROR: {str(e)}"
    
    def get_recent_metrics(self, last_n: int = 10) -> Dict[str, float]:
        """Get average metrics from recent inferences."""
        if not self.inference_history:
            return {"avg_cer": 1.0, "avg_wer": 1.0, "count": 0}
        
        recent = self.inference_history[-last_n:]
        successful = [h for h in recent if h['comparison']['status'] == 'success']
        
        if not successful:
            return {"avg_cer": 1.0, "avg_wer": 1.0, "count": 0}
        
        avg_cer = sum(h['comparison']['cer'] for h in successful) / len(successful)
        avg_wer = sum(h['comparison']['wer'] for h in successful) / len(successful)
        
        return {
            "avg_cer": avg_cer,
            "avg_wer": avg_wer,
            "count": len(successful),
            "total_attempts": len(recent)
        }