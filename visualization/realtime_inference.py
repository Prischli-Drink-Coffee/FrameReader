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

    distance = edit_distance(prediction, reference)
    return distance / len(reference)


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

    distance = edit_distance(pred_words, ref_words)
    return distance / len(ref_words)


class RealtimeInferenceEngine:
    """
    Real-time inference engine for displaying model predictions during training.
    Based on DonutInferenceEngine from train_donut/inference.py
    """
    
    def __init__(
        self,
        model,
        device: Optional[Union[str, torch.device]] = None,
        task_start_token: Optional[str] = "<s_500k>", 
        prompt_end_token: Optional[str] = "<s_prompt>",
        precision: str = "fp32", 
        max_length: int = 64,   
        num_beams: int = 1  # Faster inference for real-time display
    ):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.precision = precision
        self.max_length = max_length
        self.num_beams = num_beams
        
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token
        
        # Get EOS token for cleanup
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
            # Handle different input types
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # Add batch dimension
                    pixel_values = image.unsqueeze(0).to(self.device)
                else:
                    pixel_values = image.to(self.device)
            else:
                # PIL Image
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(self.device)

        except Exception as e:
            logger.error(f"Error processing image for realtime inference: {e}")
            return {"error": f"Image processing error: {e}"} if return_json else f"Error: Image processing error"

        input_prompt = self.prepare_prompt(prompt)
        
        try:
            decoder_input_ids = processor.tokenizer(
                input_prompt,
                add_special_tokens=False,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )["input_ids"].to(self.device)
        except Exception as e:
            logger.error(f"Error tokenizing prompt for realtime inference: {e}")
            return {"error": f"Tokenization error: {e}"} if return_json else f"Error: Tokenization error"

        gen_kwargs = {
            "no_repeat_ngram_size": 2,
            "do_sample": False,  # Greedy decoding for speed
            "early_stopping": True
        }
        
        self.model.eval()
        with torch.no_grad():
            try:
                if self.device.type == 'cuda' and self.precision in ["fp16", "bf16"]:
                    dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
                    with torch.autocast(device_type=self.device.type, dtype=dtype):
                        # Use model's generate method directly for speed
                        if hasattr(self.model, 'generate'):
                            model_output = self.model.generate(
                                pixel_values,
                                decoder_input_ids=decoder_input_ids,
                                max_length=self.max_length, 
                                num_beams=self.num_beams,
                                **gen_kwargs 
                            )
                        else:
                            # Fallback for custom models
                            model_output = self.model(
                                pixel_values=pixel_values,
                                decoder_input_ids=decoder_input_ids
                            ).logits.argmax(dim=-1)
                else:
                    if hasattr(self.model, 'generate'):
                        model_output = self.model.generate(
                            pixel_values,
                            decoder_input_ids=decoder_input_ids,
                            max_length=self.max_length,
                            num_beams=self.num_beams,
                            **gen_kwargs
                        )
                    else:
                        model_output = self.model(
                            pixel_values=pixel_values,
                            decoder_input_ids=decoder_input_ids
                        ).logits.argmax(dim=-1)

            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                return {"error": f"Model inference error: {e}"} if return_json else f"Error: Model inference error"
        
        # Decode output
        try:
            if isinstance(model_output, torch.Tensor):
                raw_text_outputs = processor.batch_decode(model_output, skip_special_tokens=False)
                raw_text_output = raw_text_outputs[0] if raw_text_outputs else ""
            elif isinstance(model_output, list):
                raw_text_output = model_output[0] if model_output else ""
            else:
                raw_text_output = str(model_output)

        except Exception as e:
            logger.error(f"Error decoding model output: {e}")
            return {"error": f"Decoding error: {e}"} if return_json else f"Error: Decoding error"

        # Process output
        if return_json:
            try:
                result = TextCleanup.extract_fields_from_donut_output(raw_text_output)
                if not result and raw_text_output.strip() and raw_text_output.strip() != self.eos_token_str_for_cleanup:
                    result = {"text_sequence": TextCleanup.cleanup_donut_output(raw_text_output)}
                return result
            except Exception as e:
                logger.warning(f"Error parsing output to JSON: {e}")
                return {"text_sequence": TextCleanup.cleanup_donut_output(raw_text_output), "parsing_error": str(e)}
        else:
            return TextCleanup.cleanup_donut_output(raw_text_output)

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
        
        # Calculate metrics
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
            if labels is not None and hasattr(self.inference_engine.model, 'processor'):
                try:
                    if labels.dim() == 2:  # Batch dimension exists
                        sample_labels = labels[sample_idx] if sample_idx < labels.shape[0] else labels[0]
                    else:
                        sample_labels = labels
                    
                    # Decode labels to text
                    # Filter out ignore_id and special tokens
                    valid_tokens = sample_labels[sample_labels != -100]
                    if len(valid_tokens) > 0:
                        ground_truth = self.inference_engine.model.processor.tokenizer.decode(
                            valid_tokens, skip_special_tokens=True
                        )
                except Exception as e:
                    logger.debug(f"Could not decode ground truth labels: {e}")
            
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