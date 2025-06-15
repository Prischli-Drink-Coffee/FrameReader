import logging
import os
import sys
import argparse
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ast import literal_eval
from nltk import edit_distance # For CER and WER

# Assuming model.py is in the same directory or accessible via PYTHONPATH
from model import DonutModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TextCleanup:
    @staticmethod
    def cleanup_donut_output(text: Any) -> str: # Added type hint for input
        text_str = str(text if text is not None else "") # Ensure input is string
        text_str = re.sub(r"<s_([^>]*)>", "", text_str)
        text_str = re.sub(r"</s_[^>]*>", "", text_str)
        text_str = text_str.replace("<sep/>", ", ")
        text_str = re.sub(r"\s+", " ", text_str).strip()
        return text_str

    @staticmethod
    def extract_fields_from_donut_output(text: Any) -> Dict: # Added type hint for input
        output = {}
        # Ensure text is a string, as it might be None or other types from model output
        processed_text = str(text if text is not None else "")
        
        original_text_for_fallback = processed_text # Keep original for fallback

        while processed_text:
            match = re.search(r"<s_(.*?)>", processed_text, re.IGNORECASE)
            if not match:
                break 

            key = match.group(1)
            original_start_token_val = match.group(0)
            
            content_block_start = match.end(0)
            
            end_token_search_string = fr"</s_{re.escape(key)}>"
            end_token_match = None
            
            # More careful end token finding: needs to handle nesting if keys are reused
            # This simple version finds the *next* one. A proper parser would use a stack.
            # For Donut's typical structure, this might be okay if keys are unique at each level
            # or nesting is shallow.
            
            # Try to find a balanced pair of <s_key>...</s_key>
            # This is a simplified approach for finding the content of the current key
            # It does not fully handle arbitrary nesting of the *same* key.
            nesting_level = 1
            current_pos = content_block_start
            content_end_idx = -1
            
            # Search for the end token, considering simple nesting of *other* keys
            # This is still not a full parser for arbitrary nesting of the *same* key.
            temp_text_to_search = processed_text[content_block_start:]
            
            # Find all occurrences of start and end tokens for *any* key
            # This is complex to do correctly with regex for true nesting.
            # The original regex was: content_match = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", text, re.IGNORECASE | re.DOTALL)
            # This implies a non-greedy match up to the first corresponding end token.

            # Reverting to a simpler regex match for the content of the current key,
            # similar to the original structure, assuming it worked for the user's format.
            start_token_escaped = re.escape(original_start_token_val)
            end_token_escaped = re.escape(fr"</s_{key}>") # Specific to the current key

            # This regex will find the shortest content between the first original_start_token_val
            # and the first corresponding end_token_val for that key.
            content_re_match = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", processed_text, re.IGNORECASE | re.DOTALL)

            if content_re_match:
                content = content_re_match.group(1).strip()
                # Advance processed_text past this entire matched block
                processed_text = processed_text[content_re_match.end(0):].strip()
            else:
                # No matching end token found for this start token, remove start token and continue
                processed_text = processed_text.replace(original_start_token_val, "", 1).strip()
                continue # Try to find next token

            # Recursive call for nested structures
            if r"<s_" in content and r"</s_" in content: # Check if content itself contains Donut tags
                value = TextCleanup.extract_fields_from_donut_output(content)
            else: # Leaf node or list of leaf nodes
                value_parts = [part.strip() for part in content.split(r"<sep/>") if part.strip()]
                if not value_parts:
                    value = "" # Empty string if no parts, or could be None
                elif len(value_parts) == 1:
                    value = value_parts[0]
                else:
                    value = value_parts # as a list

            if value or isinstance(value, str): # Add if value is not empty list/dict or an empty string
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
             # If no structured fields extracted, and original string was not just special tokens
             cleaned_fallback = TextCleanup.cleanup_donut_output(original_text_for_fallback.strip())
             if cleaned_fallback: # only if there's actual text content left
                return {"text_sequence": cleaned_fallback}
        return output


def calculate_cer(prediction: str, reference: str) -> float:
    """Calculates Character Error Rate (CER)."""
    if not isinstance(prediction, str): prediction = str(prediction)
    if not isinstance(reference, str): reference = str(reference)

    if not reference: # If reference is empty
        return 1.0 if prediction else 0.0 # 100% error if prediction is not empty, 0% if both empty
    if not prediction: # If prediction is empty but reference is not
        return 1.0 # 100% error (all deletions)

    distance = edit_distance(prediction, reference)
    return distance / len(reference)


def calculate_wer(prediction: str, reference: str) -> float:
    """Calculates Word Error Rate (WER)."""
    if not isinstance(prediction, str): prediction = str(prediction)
    if not isinstance(reference, str): reference = str(reference)

    pred_words = prediction.split()
    ref_words = reference.split()

    if not ref_words: # If reference is empty
        return 1.0 if pred_words else 0.0
    if not pred_words: # If prediction is empty but reference is not
        return 1.0

    distance = edit_distance(pred_words, ref_words)
    return distance / len(ref_words)


class DonutInferenceEngine:
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        task_start_token: Optional[str] = "<s_500k>", 
        prompt_end_token: Optional[str] = "<s_prompt>",
        precision: str = "fp32", 
        max_length: int = 64,   
        num_beams: int = 5
    ):
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.precision = precision
        self.max_length = max_length
        self.num_beams = num_beams
        
        logger.info(f"Инициализация движка инференса из {model_path}")
        logger.info(f"Параметры движка: device={self.device}, precision={self.precision}, max_length={self.max_length}, num_beams={self.num_beams}")

        start_time = time.time()
        self.model = DonutModel.from_pretrained(
            model_path,
            device=self.device,       
            precision=self.precision, 
            max_length=self.max_length 
        )

        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token
        
        try:
            if hasattr(self.model, 'processor') and hasattr(self.model.processor, 'tokenizer'):
                tokenizer = self.model.processor.tokenizer
                # Pad Token ID Patch
                if hasattr(tokenizer, 'pad_token_id') and not isinstance(tokenizer.pad_token_id, int):
                    pad_token_str = getattr(tokenizer, 'pad_token', None)
                    if pad_token_str and isinstance(pad_token_str, str):
                        try:
                            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token_str)
                            logger.info(f"Patched tokenizer.pad_token_id to: {tokenizer.pad_token_id} (from token: '{pad_token_str}')")
                        except Exception as e: logger.error(f"Failed to convert pad_token '{pad_token_str}' for patching: {e}")
                    elif tokenizer.pad_token_id is None: # Check if it's None and needs setting (e.g. to EOS)
                        eos_id_val = getattr(tokenizer, 'eos_token_id', None)
                        if isinstance(eos_id_val, int):
                            logger.warning(f"tokenizer.pad_token_id is None. Setting to eos_token_id: {eos_id_val}")
                            tokenizer.pad_token_id = eos_id_val
                        else: logger.error(f"Cannot patch pad_token_id (None): eos_token_id is also not a valid int ({eos_id_val}).")
                    else: logger.error(f"Cannot patch pad_token_id: type {type(tokenizer.pad_token_id)}, pad_token string is '{pad_token_str}'.")

                # EOS Token ID Patch
                if hasattr(tokenizer, 'eos_token_id') and not isinstance(tokenizer.eos_token_id, int):
                    eos_token_str = getattr(tokenizer, 'eos_token', None)
                    if eos_token_str and isinstance(eos_token_str, str):
                        try:
                            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token_str)
                            logger.info(f"Patched tokenizer.eos_token_id to: {tokenizer.eos_token_id} (from token: '{eos_token_str}')")
                        except Exception as e: logger.error(f"Failed to convert eos_token '{eos_token_str}' for patching: {e}")
                    else: logger.error(f"Cannot patch eos_token_id: type {type(tokenizer.eos_token_id)}, eos_token string is '{eos_token_str}'.")

                # UNK Token ID Patch
                unk_token_str = getattr(tokenizer, 'unk_token', None)
                if unk_token_str and isinstance(unk_token_str, str): # Only if unk_token string exists
                    if hasattr(tokenizer, 'unk_token_id') and not isinstance(tokenizer.unk_token_id, int):
                        try:
                            tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids(unk_token_str)
                            logger.info(f"Patched tokenizer.unk_token_id to: {tokenizer.unk_token_id} (from token: '{unk_token_str}')")
                        except Exception as e: logger.error(f"Failed to convert unk_token '{unk_token_str}' for patching: {e}")
                elif unk_token_str is None and getattr(tokenizer, 'unk_token_id', None) is not None:
                     logger.warning(f"tokenizer.unk_token is None, but unk_token_id is {tokenizer.unk_token_id}. This might be inconsistent.")
                
                # Set string for cleanup reference
                self.eos_token_str_for_cleanup = getattr(tokenizer, 'eos_token', "</s>")
                if not isinstance(self.eos_token_str_for_cleanup, str): self.eos_token_str_for_cleanup = "</s>"
            else:
                logger.error("self.model.processor.tokenizer not found. Cannot patch/verify special token IDs.")
                self.eos_token_str_for_cleanup = "</s>"
        except Exception as e:
            logger.error(f"Error during tokenizer patch/verification: {e}", exc_info=True)
            self.eos_token_str_for_cleanup = "</s>"

        logger.info(f"Модель загружена за {time.time() - start_time:.2f} с")
        logger.info(f"Токены движка: task_start_token='{self.task_start_token}', prompt_end_token='{self.prompt_end_token}', self.eos_token_str_for_cleanup='{self.eos_token_str_for_cleanup}'")

        self.model.eval()
        logger.info(f"Модель инициализирована на устройстве {self.device}")
    
    def prepare_prompt(self, prompt: Optional[str] = None) -> str:
        if prompt is None or prompt.strip() == "":
            return self.task_start_token
        else:
            _prompt = prompt.strip()
            return f"{self.task_start_token}{_prompt}{self.prompt_end_token}"

    def process_image(
        self, 
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        return_json: bool = True, # If True, attempts to parse Donut output to JSON dict
        save_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Dict[str, Any]]:

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Изображение не найдено: {image_path}")
            try:
                loaded_image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения {image_path}: {e}", exc_info=True)
                return {"error": f"Failed to load image {image_path}: {e}"} if return_json else f"Error: Failed to load image {image_path}"
        elif isinstance(image, Image.Image):
            loaded_image = image.convert("RGB")
        else:
            raise TypeError("image argument must be a file path or PIL.Image.Image object")

        if not hasattr(self.model, 'processor') or not hasattr(self.model.processor, 'tokenizer'):
            logger.error("Processor or tokenizer not found on self.model. Cannot proceed.")
            return {"error": "Processor/tokenizer not available"} if return_json else "Error: Processor/tokenizer not available"
        
        processor = self.model.processor
        
        try:
            pixel_values = processor(loaded_image, return_tensors="pt").pixel_values
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения процессором: {e}", exc_info=True)
            return {"error": f"Processor error: {e}"} if return_json else f"Error: Processor error"

        pixel_values = pixel_values.to(self.device)
        input_prompt = self.prepare_prompt(prompt)
        
        decoder_input_ids = processor.tokenizer(
            input_prompt,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )["input_ids"].to(self.device)

        gen_kwargs = { "no_repeat_ngram_size": 3 } # Minimal, specific overrides
        
        model_output_from_generate = None
        with torch.no_grad():
            # Assuming self.model.generate is the custom method from model.py
            # It expects: pixel_values, decoder_input_ids, max_length, num_beams, return_json, **kwargs
            if self.device.type == 'cuda' and self.precision in ["fp16", "bf16"]:
                dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type=self.device.type, dtype=dtype):
                    model_output_from_generate = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.max_length, 
                        num_beams=self.num_beams,   
                        return_json=False, # Ask DonutModel.generate for List[str]
                        **gen_kwargs 
                    )
            else:
                model_output_from_generate = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    return_json=False, 
                    **gen_kwargs
                )
        
        # Process the output of DonutModel.generate
        raw_text_outputs_list = []
        if isinstance(model_output_from_generate, list) and all(isinstance(s, str) for s in model_output_from_generate):
            raw_text_outputs_list = model_output_from_generate
        elif isinstance(model_output_from_generate, torch.Tensor):
             logger.info("DonutModel.generate returned a tensor, decoding.")
             raw_text_outputs_list = processor.batch_decode(model_output_from_generate, skip_special_tokens=False)
        elif isinstance(model_output_from_generate, list) and model_output_from_generate and isinstance(model_output_from_generate[0], dict):
             logger.warning("DonutModel.generate returned List[Dict] when List[str] was expected. Using string representation of first item.")
             raw_text_outputs_list = [str(model_output_from_generate[0])] # Fallback
        else:
            logger.error(f"Unexpected output type from DonutModel.generate: {type(model_output_from_generate)}. Expected List[str] or Tensor.")
            return {"error": "Unexpected output from model"} if return_json else "Error: Unexpected output from model"

        # Assuming batch size 1 for single image processing
        raw_text_output_for_this_image = raw_text_outputs_list[0] if raw_text_outputs_list else ""

        final_result_to_return = None
        if return_json: # User of this function (process_image) wants JSON
            try:
                # TextCleanup.extract_fields_from_donut_output needs the raw string with Donut tags
                final_result_to_return = TextCleanup.extract_fields_from_donut_output(raw_text_output_for_this_image)
                if not final_result_to_return and raw_text_output_for_this_image.strip() and \
                   raw_text_output_for_this_image.strip() != self.eos_token_str_for_cleanup:
                     final_result_to_return = {"text_sequence": TextCleanup.cleanup_donut_output(raw_text_output_for_this_image)}
            except Exception as e:
                logger.warning(f"Ошибка при преобразовании вывода в JSON: {e}. Raw output: '{raw_text_output_for_this_image}'", exc_info=True)
                final_result_to_return = {"text_sequence": TextCleanup.cleanup_donut_output(raw_text_output_for_this_image), "parsing_error": str(e)}
        else: # User wants cleaned string
            final_result_to_return = TextCleanup.cleanup_donut_output(raw_text_output_for_this_image)
        
        if save_path:
            save_path_obj = Path(save_path) 
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path_obj, "w", encoding="utf-8") as f:
                if isinstance(final_result_to_return, dict):
                    json.dump(final_result_to_return, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(final_result_to_return)) # Ensure it's string
        return final_result_to_return
    
    def process_batch(
        self, 
        image_paths: List[Union[str, Path]],
        prompt: Optional[str] = None,
        # batch_size arg is not used for actual batching in process_image, but for tqdm
        batch_size_for_tqdm_step: int = 1, # Renamed for clarity
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        return_json: bool = True
    ) -> List[Dict[str, Any]]:

        results_list = [] 
        output_dir_path = Path(output_dir) if output_dir else None

        if save_results and output_dir_path:
            output_dir_path.mkdir(parents=True, exist_ok=True)

        # process_image handles one image at a time. This loop iterates.
        for idx, image_path_item in enumerate(tqdm(image_paths, desc="Обработка изображений")):
            try:
                save_path_item = None 
                if save_results and output_dir_path:
                    image_name = Path(image_path_item).stem
                    ext = ".json" if return_json else ".txt"
                    save_path_item = output_dir_path / f"{image_name}_result{ext}" 
                
                result_data = self.process_image(
                    image_path_item, 
                    prompt=prompt, 
                    return_json=return_json,
                    save_path=save_path_item
                )
                
                results_list.append({
                    "image_path": str(image_path_item),
                    "result": result_data,
                    "sample_index": idx # Sequential index for plotting
                })
                        
            except Exception as e:
                logger.error(f"Ошибка при обработке {image_path_item}: {e}", exc_info=True)
                results_list.append({
                    "image_path": str(image_path_item),
                    "error": str(e),
                    "sample_index": idx
                })
        return results_list
    
    def evaluate_on_dataset(
        self,
        dataset_path: Union[str, Path],
        ground_truth_file: Optional[Union[str, Path]] = None,
        prompt: Optional[str] = None,
        batch_size_for_tqdm_step: int = 1, # For tqdm in process_batch
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        return_json_for_processing: bool = True # How process_image should handle output
    ) -> Dict[str, Any]:

        dataset_path_obj = Path(dataset_path) 
        ground_truth_map = {} 
        output_dir_path = Path(output_dir) if output_dir else None

        if ground_truth_file is not None:
            ground_truth_file_path = Path(ground_truth_file) 
            if ground_truth_file_path.exists():
                with open(ground_truth_file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            item = json.loads(line.strip())
                            file_name = item.get("file_name")
                            gt_data_payload = item.get("ground_truth")
                            if not file_name or gt_data_payload is None: continue
                            image_stem = Path(file_name).stem
                            # Ground truth can be a JSON string or an actual JSON object
                            if isinstance(gt_data_payload, str):
                                try:
                                    actual_gt_data = json.loads(gt_data_payload)
                                except json.JSONDecodeError: # If not JSON, try literal_eval
                                    try: actual_gt_data = literal_eval(gt_data_payload)
                                    except: actual_gt_data = {"text_sequence": gt_data_payload} # Fallback to text
                            else: actual_gt_data = gt_data_payload # Assumed dict/list
                            ground_truth_map[image_stem] = actual_gt_data
                        except Exception as e_json:
                            logger.warning(f"Error parsing GT line {line_num} in {ground_truth_file_path}: {e_json}")
            else:
                logger.warning(f"Ground truth file not found: {ground_truth_file_path}")

        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]: 
            image_paths.extend(list(dataset_path_obj.glob(ext)))
        
        logger.info(f"Найдено {len(image_paths)} изображений для оценки в {dataset_path_obj}")
        if not image_paths: return {"error": f"No images found in {dataset_path_obj}"}

        results_data = self.process_batch(
            image_paths, prompt, batch_size_for_tqdm_step, 
            save_results, output_dir_path, return_json_for_processing
        )

        metrics = {
            "total_images_found": len(image_paths),
            "processed_successfully": len([r for r in results_data if "error" not in r]),
            "errors_in_processing": sum(1 for r in results_data if "error" in r),
            "cer_scores": [], 
            "wer_scores": [],
            "per_sample_details": [],
        }

        for res_item in results_data: 
            sample_idx = res_item.get("sample_index", -1)
            current_image_path = res_item.get("image_path", f"unknown_image_idx_{sample_idx}")
            image_name_stem = Path(current_image_path).stem 

            if "error" in res_item:
                metrics["per_sample_details"].append({"idx": sample_idx, "cer": 1.0, "wer": 1.0, "name": image_name_stem, "error": res_item["error"]})
                metrics["cer_scores"].append(1.0) # Max error
                metrics["wer_scores"].append(1.0) # Max error
                continue
            
            cer_current_sample = 1.0 # Default to max error
            wer_current_sample = 1.0 # Default to max error

            if image_name_stem in ground_truth_map:
                prediction_data_obj = res_item["result"] 
                reference_data_obj = ground_truth_map[image_name_stem]

                # For CER/WER, we need canonical string representations
                # If prediction_data_obj is a dict (from JSON parsing), dump it. If string, use as is.
                pred_str_for_eval = json.dumps(prediction_data_obj, ensure_ascii=False, sort_keys=True) if isinstance(prediction_data_obj, dict) else str(prediction_data_obj)
                # Same for reference
                ref_str_for_eval = json.dumps(reference_data_obj, ensure_ascii=False, sort_keys=True) if isinstance(reference_data_obj, dict) else str(reference_data_obj)
                
                # If the ground truth or prediction was intended to be a simple text sequence,
                # and it got wrapped in {"text_sequence": "..."}, extract that.
                if isinstance(prediction_data_obj, dict) and "text_sequence" in prediction_data_obj and len(prediction_data_obj) == 1:
                    pred_str_for_eval = str(prediction_data_obj["text_sequence"])
                if isinstance(reference_data_obj, dict) and "text_sequence" in reference_data_obj and len(reference_data_obj) == 1:
                    ref_str_for_eval = str(reference_data_obj["text_sequence"])

                cer_current_sample = calculate_cer(pred_str_for_eval, ref_str_for_eval)
                wer_current_sample = calculate_wer(pred_str_for_eval, ref_str_for_eval)
                
                metrics["cer_scores"].append(cer_current_sample)
                metrics["wer_scores"].append(wer_current_sample)
            else: # No ground truth for this sample
                metrics["cer_scores"].append(None) # Or 1.0 if prefer to count as max error
                metrics["wer_scores"].append(None) # Or 1.0
            
            metrics["per_sample_details"].append({"idx": sample_idx, "cer": cer_current_sample, "wer": wer_current_sample, "name": image_name_stem})

        # Calculate averages and stddevs for valid scores
        valid_cer_scores = [s for s in metrics["cer_scores"] if s is not None]
        if valid_cer_scores:
            metrics["avg_cer"] = sum(valid_cer_scores) / len(valid_cer_scores)
            metrics["std_cer"] = torch.std(torch.tensor(valid_cer_scores)).item() if len(valid_cer_scores) > 1 else 0.0
        
        valid_wer_scores = [s for s in metrics["wer_scores"] if s is not None]
        if valid_wer_scores:
            metrics["avg_wer"] = sum(valid_wer_scores) / len(valid_wer_scores)
            metrics["std_wer"] = torch.std(torch.tensor(valid_wer_scores)).item() if len(valid_wer_scores) > 1 else 0.0
        
        if output_dir_path:
            self.plot_cer_wer_metrics(metrics, output_dir_path) # New plotting function
            metrics_path = output_dir_path / "evaluation_metrics_summary.json"
            summary_metrics = {k: v for k, v in metrics.items() if k != "per_sample_details"}
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(summary_metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"Метрики (сводка) сохранены в {metrics_path}")
        return metrics

    def plot_cer_wer_metrics(self, metrics_data: Dict[str, Any], output_dir: Path):
        """Plots distribution of CER and WER scores."""
        sample_details = metrics_data.get("per_sample_details", [])
        if not sample_details:
            logger.info("Нет данных по образцам для построения графиков CER/WER.")
            return

        indices = [item["idx"] for item in sample_details]
        
        # Plot for CER
        cer_scores_plot = [item["cer"] for item in sample_details if item.get("cer") is not None] # Filter None for plotting
        indices_cer = [item["idx"] for item in sample_details if item.get("cer") is not None]

        if cer_scores_plot:
            avg_cer = metrics_data.get("avg_cer")
            std_cer = metrics_data.get("std_cer")

            plt.figure(figsize=(15, 7))
            plt.bar(indices_cer, cer_scores_plot, color='cyan', label='CER per sample')
            if avg_cer is not None:
                plt.axhline(avg_cer, color='r', linestyle='--', label=f'Mean CER: {avg_cer:.3f}')
            if avg_cer is not None and std_cer is not None:
                plt.axhline(avg_cer + std_cer, color='g', linestyle=':', label=f'Mean+StdDev ({avg_cer + std_cer:.3f})')
                plt.axhline(max(0, avg_cer - std_cer), color='g', linestyle=':', label=f'Mean-StdDev ({max(0, avg_cer - std_cer):.3f})')
            
            plt.xlabel("Sample Index")
            plt.ylabel("Character Error Rate (CER)")
            plt.title("Распределение метрики Character Error Rate (CER)")
            plt.legend(loc='upper right')
            plt.ylim(min(0, min(cer_scores_plot)-0.05) if cer_scores_plot else 0, max(1.05, max(cer_scores_plot)+0.05 if cer_scores_plot else 1.05) ) # Adjust ylim based on data
            plt.tight_layout()
            plot_path = output_dir / "cer_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"График CER сохранен: {plot_path}")

        # Plot for WER
        wer_scores_plot = [item["wer"] for item in sample_details if item.get("wer") is not None]
        indices_wer = [item["idx"] for item in sample_details if item.get("wer") is not None]

        if wer_scores_plot:
            avg_wer = metrics_data.get("avg_wer")
            std_wer = metrics_data.get("std_wer")

            plt.figure(figsize=(15, 7))
            plt.bar(indices_wer, wer_scores_plot, color='magenta', label='WER per sample')
            if avg_wer is not None:
                plt.axhline(avg_wer, color='b', linestyle='--', label=f'Mean WER: {avg_wer:.3f}')
            if avg_wer is not None and std_wer is not None:
                plt.axhline(avg_wer + std_wer, color='purple', linestyle=':', label=f'Mean+StdDev ({avg_wer + std_wer:.3f})')
                plt.axhline(max(0, avg_wer - std_wer), color='purple', linestyle=':', label=f'Mean-StdDev ({max(0, avg_wer - std_wer):.3f})')

            plt.xlabel("Sample Index")
            plt.ylabel("Word Error Rate (WER)")
            plt.title("Распределение метрики Word Error Rate (WER)")
            plt.legend(loc='upper right')
            plt.ylim(min(0, min(wer_scores_plot)-0.05) if wer_scores_plot else 0, max(1.05, max(wer_scores_plot)+0.05 if wer_scores_plot else 1.05) ) # WER can exceed 1.0
            plt.tight_layout()
            plot_path = output_dir / "wer_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"График WER сохранен: {plot_path}")
    
    def visualize_prediction(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None, 
        output_dir: Optional[Union[str, Path]] = None, 
        return_json: bool = True
    ) -> None:
        pil_image = None
        image_source_path_str = None 
        output_dir_path = Path(output_dir) if output_dir else None

        if isinstance(image, (str, Path)):
            image_path_obj = Path(image) 
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Изображение не найдено: {image_path_obj}")
            pil_image = Image.open(image_path_obj).convert("RGB")
            image_source_path_str = str(image_path_obj)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB") 
            image_source_path_str = "pil_image_input" 
        else:
            raise TypeError("image argument must be a file path or PIL.Image.Image object")

        viz_save_path = Path(save_path) if save_path else None
        if viz_save_path is None and output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_source_path_str).stem
            viz_save_path = output_dir_path / f"{base_name}_visualized.png"

        json_result_save_path = None
        if output_dir_path is not None:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_source_path_str).stem
            ext = ".json" if return_json else ".txt"
            json_result_save_path = output_dir_path / f"{base_name}_result{ext}"

        result_data = self.process_image(
            pil_image, 
            prompt=prompt, 
            return_json=return_json,
            save_path=json_result_save_path 
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(pil_image)
        ax.axis('off')
        display_text = ""
        if isinstance(result_data, str):
            display_text = result_data
        elif isinstance(result_data, dict):
            display_text = json.dumps(result_data, ensure_ascii=False, indent=2)
        else:
            display_text = str(result_data) 
        plt.figtext(0.05, 0.01, display_text, wrap=True, horizontalalignment='left', fontsize=9, va='bottom')
        plt.tight_layout(rect=[0, 0.1, 1, 1]) 
        if viz_save_path: 
            viz_save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_save_path, bbox_inches='tight')
            logger.info(f"Визуализация сохранена в {viz_save_path}")
        else:
            plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Donut Inference Engine")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Путь к модели или имя модели на Hugging Face Hub")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Путь к изображению для обработки")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Путь к директории с изображениями для пакетной обработки")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Путь к файлу с эталонными данными для оценки (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="results_donut_inference",
                        help="Директория для сохранения результатов")
    parser.add_argument("--device", type=str, default=None,
                        help="Устройство для вычислений ('cpu' или 'cuda')")
    
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Точность вычислений")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Максимальная длина генерируемой последовательности")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Количество лучей для поиска по лучам")
    
    # Renamed batch_size for clarity as it's for tqdm iteration in process_batch
    parser.add_argument("--iteration_batch_size", type=int, default=1, 
                        help="Batch size for tqdm iteration in process_batch (process_image still handles one-by-one).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Промпт для модели (инструкция, которая будет вставлена)")
    parser.add_argument("--visualize", action="store_true",
                        help="Визуализировать результаты для каждого изображения в датасете (если --dataset_path указан)")
    parser.add_argument("--save_results", action="store_true",
                        help="Сохранять результаты обработки и метрики в файлы")
    # Renamed --no_json for clarity
    parser.add_argument("--output_raw_string", action="store_true", 
                        help="Выводить результат process_image как очищенную строку, а не пытаться парсить в JSON.")
    
    args = parser.parse_args()

    output_dir_path = Path(args.output_dir) 
    output_dir_path.mkdir(parents=True, exist_ok=True)

    engine = DonutInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        precision=args.precision,     
        max_length=args.max_length,   
        num_beams=args.num_beams      
    )
    
    # This flag determines if process_image tries to parse to JSON or returns cleaned string
    process_image_return_json_flag = not args.output_raw_string 
    
    if args.image_path:
        if args.visualize:
            engine.visualize_prediction(
                image=args.image_path,
                prompt=args.prompt,
                output_dir=output_dir_path if args.save_results else None, 
                return_json=process_image_return_json_flag 
            )
        else:
            save_path_single = None 
            if args.save_results:
                ext = ".json" if process_image_return_json_flag else ".txt"
                save_path_single = output_dir_path / f"{Path(args.image_path).stem}_result{ext}"
            
            result = engine.process_image(
                args.image_path, 
                prompt=args.prompt, 
                return_json=process_image_return_json_flag,
                save_path=save_path_single
            )
            
            print("\nРезультат обработки:")
            if isinstance(result, dict):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(result)

    if args.dataset_path:
        if args.ground_truth: 
            logger.info(f"Запуск оценки на датасете: {args.dataset_path} с GT: {args.ground_truth}")
            metrics = engine.evaluate_on_dataset(
                dataset_path=args.dataset_path,
                ground_truth_file=args.ground_truth,
                prompt=args.prompt,
                batch_size_for_tqdm_step=args.iteration_batch_size,
                save_results=args.save_results,
                output_dir=output_dir_path, 
                return_json_for_processing=process_image_return_json_flag
            )
            print("\nРезультаты оценки (сводка):")
            summary_metrics_display = {k: v for k, v in metrics.items() if k != "per_sample_details"}
            print(json.dumps(summary_metrics_display, ensure_ascii=False, indent=2))
            if args.save_results:
                 logger.info(f"Результаты оценки, метрики и графики сохранены в {output_dir_path}")
        else: 
            logger.info(f"Запуск пакетной обработки датасета: {args.dataset_path} (без оценки)")
            # Collect images first for process_batch
            image_paths_for_batch = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]:
                image_paths_for_batch.extend(list(Path(args.dataset_path).glob(ext)))

            results_batch = engine.process_batch( 
                image_paths=image_paths_for_batch,
                prompt=args.prompt,
                batch_size_for_tqdm_step=args.iteration_batch_size,
                save_results=args.save_results,
                output_dir=output_dir_path, 
                return_json=process_image_return_json_flag
            )
            print(f"\nОбработано {len(results_batch)} изображений из датасета.")
            if args.save_results:
                logger.info(f"Результаты пакетной обработки сохранены в {output_dir_path}")

        if args.visualize: 
            logger.info(f"Запуск визуализации для изображений из датасета: {args.dataset_path}")
            image_paths_for_viz = [] # Re-collect for clarity or use image_paths_for_batch if already collected
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]: 
                image_paths_for_viz.extend(list(Path(args.dataset_path).glob(ext)))

            for image_path_viz in tqdm(image_paths_for_viz, desc="Визуализация изображений датасета"): 
                try:
                    viz_plot_save_path = output_dir_path / f"{Path(image_path_viz).stem}_visualized.png" if args.save_results else None
                    engine.visualize_prediction(
                        image=image_path_viz,
                        prompt=args.prompt,
                        save_path=viz_plot_save_path, 
                        output_dir=output_dir_path if args.save_results else None, 
                        return_json=process_image_return_json_flag 
                    )
                except Exception as e:
                    logger.error(f"Ошибка при визуализации {image_path_viz}: {e}", exc_info=True)
            print(f"\nВизуализировано {len(image_paths_for_viz)} изображений из датасета.")

if __name__ == "__main__":
    main()