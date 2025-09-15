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

from .augmentations import AdaptiveAugmentations
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


class BaseOCRDataset(Dataset):
    def __init__(self, processor, data_dir: Union[str, Path], split: str, config: Dict[str, Any]):
        self.processor = processor
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.config = config
        self.max_length = config.get('max_length', 512)
        self.image_size = config.get('image_size', (384, 384))

        if isinstance(self.image_size, list) and len(self.image_size) == 2:
            self.image_size = tuple(self.image_size)
        elif isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        self.ignore_id = -100
        
        self.apply_augmentation = config.get('apply_augmentation', True) and split == 'train'
        self.cache_images = config.get('enable_caching', True)
        self.sort_json_key = config.get('sort_json_key', True)
        
        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            real_split_dir = self.data_dir / "real" / split
            synth_split_dir = self.data_dir / "synth" / split
            
            if real_split_dir.exists():
                self.split_dir = real_split_dir
            elif synth_split_dir.exists():
                self.split_dir = synth_split_dir
            else:
                self.split_dir = self.data_dir / split
        
        self.task_start_token = config.get('task_start_token', '<s_ocr>')
        self.prompt_end_token = '<s_prompt>'
        self.added_tokens = []
        
        if self.apply_augmentation:
            self.augmentations = AdaptiveAugmentations(config)
        
        self.samples = self._load_metadata()
        self.gt_token_sequences = []

        # logger.info(self.samples[0])
        
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
        
            self.gt_token_sequences.append(
                [
                    self.task_start_token
                    + self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons
                ]
            )

        # logger.info(self.gt_token_sequences[0])
        
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.image_cache = {} if self.cache_images else None
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True) -> str:
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                text = obj["text_sequence"]
                text = ' '.join(text.split())
                return text
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"
            return obj
    
    def add_tokens(self, list_of_tokens: List[str]) -> int:
        try:
            newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
            if newly_added_num > 0:
                if hasattr(self.processor, "model") and hasattr(self.processor.model, "decoder"):
                    self.processor.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
                self.added_tokens.extend(list_of_tokens)
            return newly_added_num
        except Exception as e:
            logger.warning(f"Failed to add tokens: {e}")
            return 0
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        samples = []
        
        logger.info(f"Загрузка метаданных для split: {self.split}")
        
        max_samples = self.config.get('max_samples_per_split', None)
        if max_samples:
            logger.info(f"Ограничение выборки: {max_samples} образцов для split '{self.split}'")
        
        jsonl_files = []
        
        for category in ["real", "synth"]:
            category_split_dir = self.data_dir / category / self.split
            
            if category_split_dir.exists():
                category_jsonl_files = list(category_split_dir.glob("*.jsonl"))
                jsonl_files.extend(category_jsonl_files)
                all_files = list(category_split_dir.iterdir())
        
        if not jsonl_files:
            logger.info("JSONL файлы не найдены в структуре real/synth, проверяем другие варианты...")
            
            direct_split_dir = self.data_dir / self.split
            logger.info(f"Проверяем прямую директорию split: {direct_split_dir}")
            
            if direct_split_dir.exists():
                direct_jsonl_files = list(direct_split_dir.glob("*.jsonl"))
                jsonl_files.extend(direct_jsonl_files)
            
            if not jsonl_files:
                logger.info(f"Проверяем корневую директорию: {self.data_dir}")
                root_jsonl_files = list(self.data_dir.glob(f"*{self.split}*.jsonl"))
                jsonl_files.extend(root_jsonl_files)
        
        if jsonl_files:
            for jsonl_file in jsonl_files:
                logger.info(f"Загрузка метаданных из {jsonl_file}")
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                        
                            if max_samples and len(samples) >= max_samples:
                                logger.info(f"Достигнуто ограничение: {max_samples} образцов для split '{self.split}'")
                                break
                                
                            if not line.strip():
                                continue
                            try:
                                item = json.loads(line.strip())
                                file_name = item.get("file_name", item.get("image", ""))
                                
                                if not file_name:
                                    logger.warning(f"Отсутствует file_name в строке {line_num} файла {jsonl_file}")
                                    continue
                                
                                ground_truth = json.loads(item["ground_truth"])
                                
                                if isinstance(ground_truth, str):
                                    ground_truth = {"text_sequence": ground_truth}
                                
                                image_path = self._find_image_path(file_name, jsonl_file.parent)
                                if not image_path or not image_path.exists():
                                    logger.warning(f"Изображение не найдено: {file_name} (строка {line_num} в {jsonl_file})")
                                    continue
                                
                                if not ground_truth or ("gt_parse" not in ground_truth and "gt_parses" not in ground_truth and "text_sequence" not in ground_truth):
                                    if "text" in item:
                                        ground_truth = {"gt_parse": {"text_sequence": item["text"]}}
                                    else:
                                        logger.warning(f"Отсутствуют данные разметки для {file_name}")
                                        continue
                                
                                if "text_sequence" in ground_truth and "gt_parse" not in ground_truth:
                                    ground_truth = {"gt_parse": {"text_sequence": ground_truth["text_sequence"]}}
                                    
                                samples.append({
                                    "image_path": image_path,
                                    "ground_truth": ground_truth
                                })
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.warning(f"Ошибка при разборе строки метаданных {line_num} в {jsonl_file}: {e}")
                                continue
                        
                        if max_samples and len(samples) >= max_samples:
                            break
                            
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла {jsonl_file}: {e}")
                    continue
        else:
            logger.error("JSONL файлы метаданных не найдены!")
            
            for category in ["real", "synth"]:
                category_dir = self.data_dir / category
                if category_dir.exists():
                    category_split_dir = category_dir / self.split
            
            raise FileNotFoundError(f"JSONL файлы метаданных не найдены в структуре {self.data_dir}/real|synth/{self.split}")
        
        if not samples:
            logger.warning(f"Не найдено валидных образцов для split: {self.split}")
        else:
            logger.info(f"Загружено {len(samples)} образцов для split: {self.split}")
            if max_samples and len(samples) >= max_samples:
                logger.info(f"Применено ограничение max_samples_per_split={max_samples}")
        
        return samples
    
    def _find_image_path(self, file_name: str, metadata_dir: Optional[Path] = None) -> Optional[Path]:
        search_dirs = []
        
        if metadata_dir and metadata_dir.exists():
            search_dirs.append(metadata_dir)
        
        for category in ["real", "synth"]:
            category_split_dir = self.data_dir / category / self.split
            if category_split_dir.exists():
                search_dirs.append(category_split_dir)
        
        fallback_dirs = [
            self.data_dir / self.split,
            self.data_dir,
            self.data_dir / "images",
            self.data_dir / "img"
        ]
        
        for fallback_dir in fallback_dirs:
            if fallback_dir.exists():
                search_dirs.append(fallback_dir)
        
        for search_dir in search_dirs:
            if search_dir.exists():
                exact_path = search_dir / file_name
                if exact_path.exists():
                    return exact_path
                
                name_without_ext = Path(file_name).stem
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    path_with_ext = search_dir / (name_without_ext + ext)
                    if path_with_ext.exists():
                        return path_with_ext
                
                pattern_files = list(search_dir.glob(f"{name_without_ext}.*"))
                if pattern_files:
                    return pattern_files[0]
        
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]

        image_path = sample["image_path"]
        ground_truth = sample["ground_truth"]
        original_text = ""
        
        if "gt_parse" in ground_truth:
            if isinstance(ground_truth["gt_parse"], str):
                original_text = ground_truth["gt_parse"]
                # logger.info(f"[DEBUG-GT] Извлечен текст из gt_parse (строка): '{original_text}'")
            elif isinstance(ground_truth["gt_parse"], dict):
                if "text_sequence" in ground_truth["gt_parse"]:
                    original_text = ground_truth["gt_parse"]["text_sequence"]
                    # logger.info(f"[DEBUG-GT] Извлечен текст из gt_parse.text_sequence: '{original_text}'")
                else:
                    original_text = json.dumps(ground_truth["gt_parse"], ensure_ascii=False)
                    # logger.info(f"[DEBUG-GT] Извлечен JSON из gt_parse: '{original_text}'")
            else:
                original_text = str(ground_truth["gt_parse"])
                # logger.info(f"[DEBUG-GT] Извлечен строковый текст из gt_parse: '{original_text}'")
        elif "text_sequence" in ground_truth:
            original_text = ground_truth["text_sequence"]
            # logger.info(f"[DEBUG-GT] Извлечен текст из text_sequence: '{original_text}'")

        if isinstance(original_text, dict):
            original_text = json.dumps(original_text, ensure_ascii=False)
            # logger.info(f"[DEBUG-GT] Преобразован словарь в JSON строку: '{original_text}'")
        
        original_text = str(original_text).strip()

        # logger.info(f"[DEBUG-GT] Окончательный исходный текст: '{original_text}'")

        if self.image_cache is not None and str(image_path) in self.image_cache:
            image = self.image_cache[str(image_path)].copy()
        else:
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image = image.resize(self.image_size[::-1], Image.LANCZOS)
                if self.image_cache is not None:
                    self.image_cache[str(image_path)] = image.copy()
            except Exception as e:
                logger.error(f"Ошибка при загрузке изображения {image_path}: {e}")
                image = Image.new('RGB', self.image_size[::-1], color='white')

        if self.apply_augmentation:
            image = self.augmentations.apply_augmentations(image)

        try:
            if hasattr(self.processor, 'image_processor'):
                pixel_values = self.processor.image_processor(
                    image, 
                    size={"height": self.image_size[0], "width": self.image_size[1]},
                    return_tensors="pt"
                ).pixel_values.squeeze()
            elif hasattr(self.processor, 'feature_extractor'):
                pixel_values = self.processor.feature_extractor(
                    image, 
                    size={"height": self.image_size[0], "width": self.image_size[1]},
                    return_tensors="pt"
                ).pixel_values.squeeze()
            else:
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            pixel_values = torch.zeros((3, self.image_size[0], self.image_size[1]))

        if len(self.gt_token_sequences[idx]) > 0:
            processed_parse = random.choice(self.gt_token_sequences[idx])
        else:
            processed_parse = self.task_start_token + self.processor.tokenizer.eos_token

        # logger.info(processed_parse)

        try:
            input_ids = self.processor.tokenizer(
                processed_parse,
                add_special_tokens=False,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
        except Exception as e:
            logger.error(f"Ошибка токенизации: {e}")
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
        
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
            
            task_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)
            task_token_positions = [i for i, id in enumerate(input_ids) if id == task_start_token_id]
            
            eos_token_id = self.processor.tokenizer.eos_token_id
            eos_positions = [i for i, id in enumerate(input_ids) if id == eos_token_id]

            masking = self.config.get('enable_masking', True)

            if masking and task_token_positions and random.random() < 0.3:
                task_start_pos = task_token_positions[0]
                end_pos = eos_positions[0] if eos_positions else len(input_ids)
                
                word_start_positions = []
                for pos in range(task_start_pos+1, end_pos):
                    if pos < len(input_ids):
                        try:
                            token = self.processor.tokenizer.convert_ids_to_tokens(input_ids[pos].item())
                            if isinstance(token, str) and (token.startswith('▁') or pos == task_start_pos+1):
                                word_start_positions.append(pos)
                        except:
                            continue
                
                if len(word_start_positions) >= 3:
                    content_positions = list(range(task_start_pos+1, end_pos))
                    
                    special_token_ids = {
                        task_start_token_id, 
                        eos_token_id, 
                        self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token),
                        self.processor.tokenizer.pad_token_id,
                        self.ignore_id
                    }
                    
                    valid_positions = [
                        pos for pos in content_positions 
                        if input_ids[pos].item() not in special_token_ids
                    ]
                    
                    valid_word_start_positions = [
                        pos for pos in word_start_positions 
                        if pos in valid_positions
                    ]

                    if valid_word_start_positions:
                        mask_start_pos = random.choice(valid_word_start_positions)
                        next_word_idx = valid_word_start_positions.index(mask_start_pos) + 1
                        mask_end_pos = valid_word_start_positions[next_word_idx] if next_word_idx < len(valid_word_start_positions) else end_pos
                        input_ids[mask_start_pos:mask_end_pos] = self.ignore_id
            
            return pixel_values, input_ids, original_text
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

            return pixel_values, input_ids, original_text


class DonutDataset(BaseOCRDataset):
    def __init__(self, processor, data_dir: Union[str, Path], split: str, config: Dict[str, Any]):
        super().__init__(processor, data_dir, split, config)


class TrOCRDataset(BaseOCRDataset):
    def __init__(self, processor, data_dir: Union[str, Path], split: str, config: Dict[str, Any]):
        super().__init__(processor, data_dir, split, config)


class TwoStageDataset(BaseOCRDataset):
    def __init__(self, real_dataset: BaseOCRDataset, synthetic_dataset: Optional[BaseOCRDataset] = None):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.current_stage = "real"
        
        self.samples = real_dataset.samples if real_dataset else []
        if synthetic_dataset:
            self.samples.extend(synthetic_dataset.samples)
    
    def set_stage(self, stage: str):
        self.current_stage = stage
        if stage == "real" and self.real_dataset:
            self.samples = self.real_dataset.samples
        elif stage == "synthetic" and self.synthetic_dataset:
            self.samples = self.synthetic_dataset.samples
        else:
            if self.real_dataset:
                self.samples = self.real_dataset.samples
            elif self.synthetic_dataset:
                self.samples = self.synthetic_dataset.samples
    
    def __getitem__(self, idx: int):
        if self.current_stage == "real" and self.real_dataset:
            return self.real_dataset[idx % len(self.real_dataset)]
        elif self.current_stage == "synthetic" and self.synthetic_dataset:
            return self.synthetic_dataset[idx % len(self.synthetic_dataset)]
        else:
            return self.real_dataset[idx % len(self.real_dataset)] if self.real_dataset else None
    
    def __len__(self) -> int:
        if self.current_stage == "real" and self.real_dataset:
            return len(self.real_dataset)
        elif self.current_stage == "synthetic" and self.synthetic_dataset:
            return len(self.synthetic_dataset)
        else:
            return len(self.real_dataset) if self.real_dataset else 0


def create_dataset(model_type: str, processor, data_dir: Union[str, Path], split: str, config: Dict[str, Any]) -> BaseOCRDataset:
    model_type = model_type.lower()
    if model_type in ["donut", "donut-ocr"]:
        return DonutDataset(processor, data_dir, split, config)
    elif model_type in ["trocr", "trocr-ocr"]:
        return TrOCRDataset(processor, data_dir, split, config)
    else:
        logger.warning(f"Unknown model type: {model_type}, defaulting to DonutDataset")
        return DonutDataset(processor, data_dir, split, config)


def create_dataloader(dataset: Dataset, config: Dict[str, Any], split: str = "train") -> DataLoader:
    batch_size = config.get('batch_size', 8)
    if split != "train":
        batch_size = config.get('eval_batch_size', batch_size)
    
    num_workers = config.get('num_workers', 4)
    shuffle = split == "train" and not config.get('distributed', False)
    
    sampler = None
    if config.get('distributed', False):
        sampler = DistributedSampler(dataset, shuffle=(split == "train"))
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train")
    )


def collate_fn(batch):

    if isinstance(batch[0], tuple):
        pixel_values = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        
        texts = []
        for item in batch:
            text = item[2]
            texts.append(str(text))
        return {"pixel_values": pixel_values, "labels": labels, "texts": texts}
            
    elif isinstance(batch[0], dict):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        texts = []
        if "texts" in batch[0]:
            for item in batch:
                text_val = item.get("texts")
                texts.append(str(text_val))
            return {"pixel_values": pixel_values, "labels": labels, "texts": texts}


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
