import json
import logging
import os
import sys
import random
import re
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from nltk import edit_distance

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import DonutProcessor
from torchvision import transforms
from zss import Node
import zss

import io
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
        
try:
    import cv2
except ImportError:
    logger.warning("OpenCV (cv2) не установлен, некоторые аугментации будут недоступны")
    cv2 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DonutDataset(Dataset):
    
    def __init__(
        self,
        processor: DonutProcessor,
        data_dir: Union[str, Path],
        split: str = "train",
        max_length: int = 768,
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: Optional[str] = None,
        sort_json_key: bool = True,
        image_size: Tuple[int, int] = (1280, 960),
        apply_augmentation: bool = True,
        cache_images: bool = False,
        limit_samples: Optional[int] = None,
        augmentation_prob: float = 0.5,
        max_rotation: float = 5.0,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        blur_range: Tuple[int, int] = (0, 2),
        noise_level: float = 0.05,
        sharpness_range: Tuple[float, float] = (0.8, 1.5)
    ):
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
        self.cache_images = cache_images

        self.augmentation_prob = augmentation_prob
        self.max_rotation = max_rotation
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_range = blur_range
        self.noise_level = noise_level
        self.sharpness_range = sharpness_range
        
        self.processor.image_processor.size = image_size[::-1]  # (width, height)
        self.processor.image_processor.do_align_long_axis = False

        if self.apply_augmentation:
            self._setup_augmentations()

        self.split_dir = self.data_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Директория разделения не найдена: {self.split_dir}")

        all_samples = self._load_metadata()
        logger.info(f"Загружено {len(all_samples)} записей метаданных из {self.split_dir}")

        if limit_samples is not None and limit_samples > 0:
            all_samples = all_samples[:limit_samples]
            logger.info(f"Ограничено до {len(all_samples)} образцов для отладки")
        
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
        
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.image_cache = {} if cache_images else None
        
        logger.info(f"Инициализирован датасет Donut для разделения '{split}' с {len(self.samples)} образцами")
        logger.info(f"Добавлено {len(self.added_tokens)} специальных токенов")
    
    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True) -> str:
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
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
            raise FileNotFoundError(f"JSONL файлы метаданных не найдены в {self.split_dir}")
        
        for jsonl_file in jsonl_files:
            logger.info(f"Загрузка метаданных из {jsonl_file}")
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
                                    logger.warning(f"Невозможно разобрать ground_truth для {file_name}")
                                    continue
                        else:
                            ground_truth = item["ground_truth"]
                        
                        image_path = self.split_dir / file_name
                        if not image_path.exists():
                            potential_paths = list(self.split_dir.glob(f"{file_name}.*"))
                            if potential_paths:
                                image_path = potential_paths[0]
                            else:
                                logger.warning(f"Изображение не найдено: {image_path} (строка {line_num} в {jsonl_file})")
                                continue
                        
                        if "gt_parse" not in ground_truth and "gt_parses" not in ground_truth:
                            logger.warning(f"Отсутствует gt_parse/gt_parses в ground_truth: {file_name}")
                            continue
                            
                        samples.append({
                            "image_path": image_path,
                            "ground_truth": ground_truth
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Ошибка при разборе строки метаданных {line_num} в {jsonl_file}: {e}")
                        continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)

    def _setup_augmentations(self):
        self.augmentations = []
        
        def apply_rotation(img):
            if random.random() < self.augmentation_prob:
                angle = random.uniform(-self.max_rotation, self.max_rotation)
                return TF.rotate(img, angle)
            return img
        self.augmentations.append(apply_rotation)
        
        def apply_brightness(img):
            if random.random() < self.augmentation_prob:
                brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
                return TF.adjust_brightness(img, brightness_factor)
            return img
        self.augmentations.append(apply_brightness)
        
        def apply_contrast(img):
            if random.random() < self.augmentation_prob:
                contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
                return TF.adjust_contrast(img, contrast_factor)
            return img
        self.augmentations.append(apply_contrast)
        
        def apply_blur(img):
            if random.random() < self.augmentation_prob and cv2 is not None:
                img_np = np.array(img)
                kernel_size = random.randint(self.blur_range[0], self.blur_range[1]) * 2 + 1
                if kernel_size > 1:
                    img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
                return Image.fromarray(img_np)
            return img
        self.augmentations.append(apply_blur)
        
        def apply_sharpness(img):
            if random.random() < self.augmentation_prob:
                sharpness_factor = random.uniform(self.sharpness_range[0], self.sharpness_range[1])
                return TF.adjust_sharpness(img, sharpness_factor)
            return img
        self.augmentations.append(apply_sharpness)
        
        def apply_salt_pepper_noise(img):
            if random.random() < self.augmentation_prob:
                img_np = np.array(img)
                h, w, c = img_np.shape
                salt_mask = np.random.random((h, w)) < self.noise_level/2
                img_np[salt_mask] = 255
                pepper_mask = np.random.random((h, w)) < self.noise_level/2
                img_np[pepper_mask] = 0
                
                return Image.fromarray(img_np)
            return img
        self.augmentations.append(apply_salt_pepper_noise)
        
        def apply_perspective(img):
            if random.random() < self.augmentation_prob:
                width, height = img.size
                factor = 0.1
                startpoints = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]
                endpoints = []
                
                for point in startpoints:
                    dx = random.uniform(-factor, factor) * width
                    dy = random.uniform(-factor, factor) * height
                    endpoints.append((point[0] + dx, point[1] + dy))
                
                return TF.perspective(img, startpoints, endpoints, TF.InterpolationMode.BILINEAR)
            return img
        self.augmentations.append(apply_perspective)
        
        def apply_elastic_transform(img):
            if random.random() < self.augmentation_prob and cv2 is not None:
                img_np = np.array(img)
                h, w, c = img_np.shape
                alpha = random.uniform(w*0.5, w*1.5)
                sigma = random.uniform(w*0.05, w*0.1)
                dx = cv2.GaussianBlur(np.random.rand(h, w) * 2 - 1, (0, 0), sigma) * alpha
                dy = cv2.GaussianBlur(np.random.rand(h, w) * 2 - 1, (0, 0), sigma) * alpha
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + dx).astype(np.float32)
                map_y = (y + dy).astype(np.float32)
                distorted = cv2.remap(img_np, map_x, map_y, 
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                
                return Image.fromarray(distorted)
            return img
        self.augmentations.append(apply_elastic_transform)
        
        def apply_color_jitter(img):
            if random.random() < self.augmentation_prob:
                hue_factor = random.uniform(-0.1, 0.1)
                saturation_factor = random.uniform(0.5, 1.5)
                
                img = TF.adjust_hue(img, hue_factor)
                img = TF.adjust_saturation(img, saturation_factor)
                
                return img
            return img
        self.augmentations.append(apply_color_jitter)
        
        def apply_cutout(img):
            if random.random() < self.augmentation_prob:
                img_np = np.array(img)
                h, w, c = img_np.shape
                num_cutouts = random.randint(1, 3)
                
                for _ in range(num_cutouts):
                    cutout_width = random.randint(int(w * 0.05), int(w * 0.2))
                    cutout_height = random.randint(int(h * 0.05), int(h * 0.2))
                    x = random.randint(0, w - cutout_width)
                    y = random.randint(0, h - cutout_height)
                    color = random.randint(100, 200)
                    img_np[y:y+cutout_height, x:x+cutout_width, :] = color
                    
                return Image.fromarray(img_np)
            return img
        self.augmentations.append(apply_cutout)
        
        logger.info(f"Настроено {len(self.augmentations)} типов аугментаций для датасета '{self.split}'")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        image_path = sample["image_path"]

        if self.image_cache is not None and str(image_path) in self.image_cache:
            image = self.image_cache[str(image_path)].copy()
        else:
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                if self.image_size:
                    image.thumbnail(self.image_size[::-1], Image.LANCZOS)
                if self.image_cache is not None:
                    self.image_cache[str(image_path)] = image.copy()
            except Exception as e:
                logger.error(f"Ошибка при загрузке изображения {image_path}: {e}")
                image = Image.new('RGB', self.image_size[::-1], color='white')

        if self.apply_augmentation:
            for augmentation_fn in self.augmentations:
                image = augmentation_fn(image)

        pixel_values = self.processor(
            image, 
            return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.squeeze()

        processed_parse = random.choice(self.gt_token_sequences[idx])

        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        
        if self.split == "train":
            prompt_end_token_pos = None
            if self.prompt_end_token in processed_parse:
                prompt_end_tokens = [i for i, id in enumerate(input_ids) 
                                    if id == self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)]
                if prompt_end_tokens:
                    prompt_end_token_pos = prompt_end_tokens[0]

            input_ids[
                input_ids == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id

            if prompt_end_token_pos is not None:
                input_ids[:prompt_end_token_pos+1] = self.ignore_id
            
            task_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.task_start_token)
            task_token_positions = [i for i, id in enumerate(input_ids) if id == task_start_token_id]
            
            eos_token_id = self.processor.tokenizer.eos_token_id
            eos_positions = [i for i, id in enumerate(input_ids) if id == eos_token_id]

            masking = True

            if masking and task_token_positions and random.random() < 0.3:
                task_start_pos = task_token_positions[0]
                end_pos = eos_positions[0] if eos_positions else len(input_ids)
                
                word_start_positions = []
                for pos in range(task_start_pos+1, end_pos):
                    if pos < len(input_ids):
                        token = self.processor.tokenizer.convert_ids_to_tokens(input_ids[pos].item())
                        if token.startswith('▁') or pos == task_start_pos+1:
                            word_start_positions.append(pos)
                
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

            input_ids[
                input_ids == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id

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
                logger.error(f"Ошибка при визуализации образца {idx}: {e}")
                image = Image.new('RGB', (100, 100), color='white')
            if "gt_parse" in sample["ground_truth"]:
                json_data = sample["ground_truth"]["gt_parse"]
            else:
                json_data = sample["ground_truth"].get("gt_parses", [{}])[0]
            
            token_sequence = self.gt_token_sequences[idx][0]
            
            return image, json.dumps(json_data, ensure_ascii=False, indent=2), token_sequence
        
        return Image.new('RGB', (100, 100), color='white'), "{}", ""


class DonutDataModule:
 
    def __init__(
        self,
        processor: DonutProcessor,
        data_dir: Union[str, Path],
        batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 768,
        task_start_token: str = "<s>",
        prompt_end_token: Optional[str] = None,
        sort_json_key: bool = True,
        image_size: Tuple[int, int] = (1280, 960),
        apply_augmentation: bool = True,
        distributed: bool = False,
        pin_memory: bool = True,
        cache_images: bool = False,
        drop_last: bool = True,
        seed: int = 42,
        train_limit_samples: Optional[int] = None,
        val_limit_samples: Optional[int] = None,
        augmentation_prob: float = 0.5,
        max_rotation: float = 5.0,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        blur_range: Tuple[int, int] = (0, 2),
        noise_level: float = 0.05,
        sharpness_range: Tuple[float, float] = (0.8, 1.5)
    ):
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.distributed = distributed
        self.pin_memory = pin_memory
        self.cache_images = cache_images
        self.drop_last = drop_last
        self.seed = seed
        self.train_limit_samples = train_limit_samples
        self.val_limit_samples = val_limit_samples

        self.augmentation_prob = augmentation_prob
        self.max_rotation = max_rotation
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_range = blur_range
        self.noise_level = noise_level
        self.sharpness_range = sharpness_range

        self.train_dataset = self._create_dataset("train", apply_augmentation=True, limit_samples=self.train_limit_samples)
        self.val_dataset = self._create_dataset("valid", apply_augmentation=False, limit_samples=self.val_limit_samples) or self._create_dataset("validation", apply_augmentation=False, limit_samples=self.val_limit_samples)
        self.test_dataset = self._create_dataset("test", apply_augmentation=False)
        
        logger.info(
            f"Инициализирован модуль данных Donut: "
            f"размер пакета={batch_size}, "
            f"рабочие процессы={num_workers}, "
            f"размер изображения={image_size}, "
            f"макс. длина={max_length}"
        )
    
    def _create_dataset(
        self, 
        split: str, 
        apply_augmentation: bool = False,
        limit_samples: Optional[int] = None
    ) -> Optional[DonutDataset]:

        split_dir = self.data_dir / split
        if not split_dir.exists():
            logger.warning(f"Директория разделения не найдена: {split_dir}")
            return None
        
        try:
            return DonutDataset(
                processor=self.processor,
                data_dir=self.data_dir,
                split=split,
                max_length=self.max_length,
                task_start_token=self.task_start_token,
                prompt_end_token=self.prompt_end_token,
                sort_json_key=self.sort_json_key,
                image_size=self.image_size,
                apply_augmentation=apply_augmentation and split == "train",
                cache_images=self.cache_images and split != "train",
                limit_samples=limit_samples,
                augmentation_prob=self.augmentation_prob,
                max_rotation=self.max_rotation,
                brightness_range=self.brightness_range,
                contrast_range=self.contrast_range,
                blur_range=self.blur_range,
                noise_level=self.noise_level,
                sharpness_range=self.sharpness_range
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Не удалось создать датасет для разделения {split}: {e}")
            return None
    
    def _create_dataloader(
        self, 
        dataset: Optional[DonutDataset], 
        shuffle: bool = False
    ) -> Optional[DataLoader]:

        if dataset is None:
            return None
        
        sampler = None
        if self.distributed:
            sampler = DistributedSampler(
                dataset, 
                shuffle=shuffle,
                seed=self.seed
            )
            logger.info(f"Создан распределенный семплер для {len(dataset)} образцов")
        
        prefetch_factor = 2
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).name.find("A100") >= 0:
            prefetch_factor = 4
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last and dataset.split == "train",
            persistent_workers=self.num_workers > 0,
            prefetch_factor=prefetch_factor if self.num_workers > 0 else None,
        )
    
    def train_dataloader(self) -> Optional[DataLoader]:
        return self._create_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self) -> Optional[DataLoader]:
        return self._create_dataloader(self.val_dataset, shuffle=False)
    
    def test_dataloader(self) -> Optional[DataLoader]:
        return self._create_dataloader(self.test_dataset, shuffle=False)
    
    def resampler_for_epoch(self, epoch: int) -> None:
        if self.distributed:
            for dataset, loader_fn in [
                (self.train_dataset, self.train_dataloader),
                (self.val_dataset, self.val_dataloader),
                (self.test_dataset, self.test_dataloader)
            ]:
                loader = loader_fn()
                if loader is not None and hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                    loader.sampler.set_epoch(epoch)
                    logger.info(f"Обновлен семплер для эпохи {epoch}")
    
    def get_sample_info(self) -> Dict[str, Any]:
        return {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "steps_per_epoch": len(self.train_dataloader()) if self.train_dataloader() else 0,
            "train_limit_samples": self.train_limit_samples,
            "val_limit_samples": self.val_limit_samples
        }


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )