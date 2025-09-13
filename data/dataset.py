import json
import logging
import os
import sys
import random
import re
import io
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import DonutProcessor, TrOCRProcessor
from torchvision import transforms
import torchvision.transforms.functional as TF

from .augmentations import ImageAugmentator, TrOCRAugmentator, NoAugmentationTransform
from .cache import ImageCache

try:
   from zss import Node
   import zss
   from nltk import edit_distance
   ZSS_AVAILABLE = True
except ImportError:
   ZSS_AVAILABLE = False
   logging.warning("zss and/or nltk not available, some evaluation features will be disabled")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class DonutDataset(Dataset):
   def __init__(self, processor: DonutProcessor, data_dir: Union[str, Path], split: str = "train", max_length: int = 768,
                ignore_id: int = -100, task_start_token: str = "<s>", prompt_end_token: Optional[str] = None,
                sort_json_key: bool = True, image_size: Tuple[int, int] = (1280, 960), apply_augmentation: bool = True,
                cache_images: bool = False, limit_samples: Optional[int] = None, augmentation_prob: float = 0.5,
                max_rotation: float = 5.0, brightness_range: Tuple[float, float] = (0.8, 1.2),
                contrast_range: Tuple[float, float] = (0.8, 1.2), blur_range: Tuple[int, int] = (0, 2),
                noise_level: float = 0.05, sharpness_range: Tuple[float, float] = (0.8, 1.5), config: Optional[Dict] = None):
       super().__init__()
       
       self.processor = processor
       self.data_dir = Path(data_dir)
       self.split = split
       self.max_length = max_length
       self.ignore_id = ignore_id
       self.task_start_token = task_start_token
       self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
       self.sort_json_key = sort_json_key
       self.image_size = image_size
       self.apply_augmentation = apply_augmentation and split == "train"
       
       self.processor.image_processor.size = image_size[::-1]
       self.processor.image_processor.do_align_long_axis = False

       self.image_cache = ImageCache(enabled=cache_images)
       
       if self.apply_augmentation:
           self.augmentator = ImageAugmentator(
               augmentation_prob=augmentation_prob, max_rotation=max_rotation, brightness_range=brightness_range,
               contrast_range=contrast_range, blur_range=blur_range, noise_level=noise_level, sharpness_range=sharpness_range
           )

       self.split_dir = self.data_dir / split
       if not self.split_dir.exists():
           raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

       all_samples = self._load_metadata()
       logger.info(f"Loaded {len(all_samples)} metadata records from {self.split_dir}")

       if limit_samples is not None and limit_samples > 0:
           all_samples = all_samples[:limit_samples]
           logger.info(f"Limited to {len(all_samples)} samples for debugging")
       
       self.samples = all_samples
       
       self.added_tokens = []
       self.gt_token_sequences = []
       
       for sample in self.samples:
           ground_truth = sample["ground_truth"]
           
           if "gt_parses" in ground_truth:
               assert isinstance(ground_truth["gt_parses"], list)
               gt_jsons = ground_truth["gt_parses"]
           elif "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], str):
               gt_jsons = [{"text_sequence": ' '.join(ground_truth["gt_parse"].split())[:self.max_length]}] 
           else:
               assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
               gt_jsons = [ground_truth["gt_parse"]]
       
           self.gt_token_sequences.append([
               self.task_start_token + self.json2token(gt_json, update_special_tokens_for_json_key=self.split == "train", sort_json_key=self.sort_json_key) + processor.tokenizer.eos_token
               for gt_json in gt_jsons
           ])
       
       self.add_tokens([self.task_start_token, self.prompt_end_token])
       self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
       
       logger.info(f"Initialized Donut dataset for split '{split}' with {len(self.samples)} samples")
       logger.info(f"Added {len(self.added_tokens)} special tokens")
   
   def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True) -> str:
       if type(obj) == dict:
           if len(obj) == 1 and "text_sequence" in obj:
               return obj["text_sequence"]
           else:
               output = ""
               keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
               for k in keys:
                   if update_special_tokens_for_json_key:
                       self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                   output += fr"<s_{k}>" + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key) + fr"</s_{k}>"
               return output
       elif type(obj) == list:
           return r"<sep/>".join([self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj])
       else:
           obj = str(obj)
           if f"<{obj}/>" in self.added_tokens:
               obj = f"<{obj}/>"
           return obj
   
   def add_tokens(self, list_of_tokens: List[str]) -> int:
       newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
       if newly_added_num > 0:
           if hasattr(self.processor, "model") and hasattr(self.processor.model, "decoder"):
               self.processor.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
           self.added_tokens.extend(list_of_tokens)
       return newly_added_num
   
   def _load_metadata(self) -> List[Dict[str, Any]]:
       samples = []
       jsonl_files = list(self.split_dir.glob("*.jsonl"))
       
       if not jsonl_files:
           raise FileNotFoundError(f"JSONL metadata files not found in {self.split_dir}")
       
       for jsonl_file in jsonl_files:
           logger.info(f"Loading metadata from {jsonl_file}")
           with open(jsonl_file, "r", encoding="utf-8") as f:
               for line_num, line in enumerate(f, 1):
                   try:
                       item = json.loads(line.strip())
                       file_name = item["file_name"]

                       if isinstance(item["ground_truth"], str):
                           try:
                               ground_truth = json.loads(item["ground_truth"])
                           except json.JSONDecodeError:
                               try:
                                   ground_truth = literal_eval(item["ground_truth"])
                               except (SyntaxError, ValueError):
                                   logger.warning(f"Cannot parse ground_truth for {file_name}")
                                   continue
                       else:
                           ground_truth = item["ground_truth"]
                       
                       image_path = self.split_dir / file_name
                       if not image_path.exists():
                           potential_paths = list(self.split_dir.glob(f"{file_name}.*"))
                           if potential_paths:
                               image_path = potential_paths[0]
                           else:
                               logger.warning(f"Image not found: {image_path} (line {line_num} in {jsonl_file})")
                               continue
                       
                       if "gt_parse" not in ground_truth and "gt_parses" not in ground_truth:
                           logger.warning(f"Missing gt_parse/gt_parses in ground_truth: {file_name}")
                           continue
                           
                       samples.append({"image_path": image_path, "ground_truth": ground_truth})
                   except (json.JSONDecodeError, KeyError) as e:
                       logger.warning(f"Error parsing metadata line {line_num} in {jsonl_file}: {e}")
                       continue
       
       return samples
   
   def __len__(self) -> int:
       return len(self.samples)

   def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
       sample = self.samples[idx]
       image_path = sample["image_path"]

       image = self.image_cache.load_image_with_cache(image_path, self.image_size)

       if self.apply_augmentation and hasattr(self, 'augmentator'):
           image = self.augmentator.apply(image)

       pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

       processed_parse = random.choice(self.gt_token_sequences[idx])

       input_ids = self.processor.tokenizer(
           processed_parse, add_special_tokens=False, max_length=self.max_length,
           padding="max_length", truncation=True, return_tensors="pt"
       )["input_ids"].squeeze(0)
       
       if self.split == "train":
           prompt_end_token_pos = None
           if self.prompt_end_token in processed_parse:
               prompt_end_tokens = [i for i, id in enumerate(input_ids) 
                                   if id == self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)]
               if prompt_end_tokens:
                   prompt_end_token_pos = prompt_end_tokens[0]

           input_ids[input_ids == self.processor.tokenizer.pad_token_id] = self.ignore_id

           if prompt_end_token_pos is not None:
               input_ids[:prompt_end_token_pos+1] = self.ignore_id
           
           return pixel_values, input_ids, processed_parse
       else:
           prompt_end_index = None
           if self.prompt_end_token in processed_parse:
               prompt_end_index_list = [i for i, id in enumerate(input_ids) 
                                      if id == self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)]
               if prompt_end_index_list:
                   prompt_end_index = prompt_end_index_list[0]
                   
           if prompt_end_index is not None:
               input_ids[:prompt_end_index+1] = self.ignore_id

           input_ids[input_ids == self.processor.tokenizer.pad_token_id] = self.ignore_id

           if prompt_end_index is None:
               prompt_end_index = -1

           return pixel_values, input_ids, prompt_end_index, processed_parse
   
   def visualize_sample(self, idx: int) -> Tuple[Image.Image, str, str]:
       if 0 <= idx < len(self.samples):
           sample = self.samples[idx]
           image_path = sample["image_path"]
           try:
               image = Image.open(image_path).convert("RGB")
           except Exception as e:
               logger.error(f"Error visualizing sample {idx}: {e}")
               image = Image.new('RGB', (100, 100), color='white')
               
           if "gt_parse" in sample["ground_truth"]:
               json_data = sample["ground_truth"]["gt_parse"]
           else:
               json_data = sample["ground_truth"].get("gt_parses", [{}])[0]
           
           token_sequence = self.gt_token_sequences[idx][0]
           
           return image, json.dumps(json_data, ensure_ascii=False, indent=2), token_sequence
       
       return Image.new('RGB', (100, 100), color='white'), "{}", ""
   
   def get_data_statistics(self) -> Dict[str, Any]:
       return {"total_samples": len(self.samples), "data_types": ["mixed"]}


class TrOCRDataset(Dataset):
   def __init__(self, processor: TrOCRProcessor, data_dir: Union[str, Path], split: str = "train", max_length: int = 64,
                ignore_id: int = -100, image_size: Tuple[int, int] = (384, 384), fraction: float = 1.0,
                apply_augmentation: bool = True, cache_images: bool = False, limit_samples: Optional[int] = None, config: Optional[Dict] = None):
       super().__init__()
       self.processor = processor
       self.data_dir = Path(data_dir)
       self.split = split
       self.max_length = max_length
       self.ignore_id = ignore_id
       self.image_size = image_size
       self.fraction = fraction
       self.apply_augmentation = apply_augmentation and split == "train"
       
       self.image_cache = ImageCache(enabled=cache_images)
       
       if self.apply_augmentation:
           self.augmentator = TrOCRAugmentator(image_size=image_size)
       else:
           self.augmentator = NoAugmentationTransform(image_size=image_size)
       
       self.split_dir = self.data_dir / split
       
       if not self.split_dir.exists():
           raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

       all_samples = self._load_metadata()
       logger.info(f"Loaded {len(all_samples)} metadata records from {self.split_dir}")

       if self.fraction < 1.0:
           random.seed(42)
           random.shuffle(all_samples)
           num_samples = max(1, int(len(all_samples) * self.fraction))
           self.samples = all_samples[:num_samples]
           logger.info(f"Using {len(self.samples)} samples ({self.fraction:.1%} of {len(all_samples)}) for split {split}")
       else:
           self.samples = all_samples
           logger.info(f"Loaded {len(self.samples)} samples for split {split}")
       
       if limit_samples is not None and limit_samples > 0:
           self.samples = self.samples[:limit_samples]
           logger.info(f"Limited to {len(self.samples)} samples for debugging")
   
   def _load_metadata(self) -> List[Dict[str, Any]]:
       samples = []
       jsonl_files = list(self.split_dir.glob("*.jsonl"))
       
       if not jsonl_files:
           raise FileNotFoundError(f"JSONL metadata files not found in {self.split_dir}")
       
       for jsonl_file in jsonl_files:
           logger.info(f"Loading metadata from {jsonl_file}")
           with open(jsonl_file, "r", encoding="utf-8") as f:
               for line_num, line in enumerate(f, 1):
                   try:
                       item = json.loads(line.strip())
                       file_name = item["file_name"]
                       ground_truth = json.loads(item["ground_truth"]) if isinstance(item["ground_truth"], str) else item["ground_truth"]
                       
                       image_path = self.split_dir / file_name
                       if not image_path.exists():
                           potential_paths = list(self.split_dir.glob(f"{file_name}.*"))
                           if potential_paths:
                               image_path = potential_paths[0]
                           else:
                               logger.warning(f"Image not found: {image_path} (line {line_num} in {jsonl_file})")
                               continue
                       
                       if "gt_parse" not in ground_truth:
                           logger.warning(f"Missing gt_parse in ground_truth: {file_name} (line {line_num} in {jsonl_file})")
                           continue
                           
                       samples.append({"image_path": image_path, "ground_truth": ground_truth})
                   except (json.JSONDecodeError, KeyError) as e:
                       logger.warning(f"Error parsing metadata line {line_num} in {jsonl_file}: {e}")
                       continue
       
       return samples
   
   def __len__(self) -> int:
       return len(self.samples)
   
   def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
       sample = self.samples[idx]
       image_path = sample["image_path"]
       ground_truth = sample["ground_truth"]
       text = ground_truth["gt_parse"]
       
       image = self.image_cache.load_image_with_cache(image_path, self.image_size)
       
       if hasattr(self.augmentator, 'apply'):
           pixel_values = self.augmentator.apply(image)
       else:
           pixel_values = self.augmentator(image)
       
       encoding = self.processor(text=text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
       
       labels = encoding["input_ids"].squeeze()
       labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
       
       return {"pixel_values": pixel_values, "labels": labels}
   
   def get_data_statistics(self) -> Dict[str, Any]:
       return {"total_samples": len(self.samples), "data_types": ["mixed"]}


if ZSS_AVAILABLE:
   class JSONParseEvaluator:
       def __init__(self):
           self.results = []
       
       def add_prediction(self, prediction: str, ground_truth: str):
           try:
               pred_tree = self._json_to_tree(json.loads(prediction))
               gt_tree = self._json_to_tree(json.loads(ground_truth))
               distance = zss.simple_distance(pred_tree, gt_tree)
               self.results.append({'prediction': prediction, 'ground_truth': ground_truth, 'tree_edit_distance': distance})
           except (json.JSONDecodeError, Exception) as e:
               distance = edit_distance(prediction, ground_truth)
               self.results.append({'prediction': prediction, 'ground_truth': ground_truth, 'string_edit_distance': distance})
       
       def _json_to_tree(self, obj, name="root"):
           if isinstance(obj, dict):
               node = Node(name)
               for key, value in obj.items():
                   child = self._json_to_tree(value, key)
                   node.addkid(child)
               return node
           elif isinstance(obj, list):
               node = Node(name)
               for i, item in enumerate(obj):
                   child = self._json_to_tree(item, f"item_{i}")
                   node.addkid(child)
               return node
           else:
               return Node(f"{name}:{str(obj)}")
       
       def compute_metrics(self) -> Dict[str, float]:
           if not self.results:
               return {}
           
           tree_distances = [r.get('tree_edit_distance', 0) for r in self.results if 'tree_edit_distance' in r]
           string_distances = [r.get('string_edit_distance', 0) for r in self.results if 'string_edit_distance' in r]
           
           metrics = {'num_samples': len(self.results), 'num_tree_parsed': len(tree_distances), 'num_string_fallback': len(string_distances)}
           
           if tree_distances:
               metrics['avg_tree_edit_distance'] = sum(tree_distances) / len(tree_distances)
           
           if string_distances:
               metrics['avg_string_edit_distance'] = sum(string_distances) / len(string_distances)
           
           return metrics
else:
   class JSONParseEvaluator:
       def __init__(self):
           self.results = []
       
       def add_prediction(self, prediction: str, ground_truth: str):
           self.results.append({'prediction': prediction, 'ground_truth': ground_truth})
       
       def compute_metrics(self) -> Dict[str, float]:
           return {'num_samples': len(self.results), 'evaluation_disabled': True}