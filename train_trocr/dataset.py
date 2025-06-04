import json
import logging
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import TrOCRProcessor
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrOCRDataset(Dataset):
    
    def __init__(
        self,
        processor: TrOCRProcessor,
        data_dir: Union[str, Path],
        split: str = "train",
        max_length: int = 64,
        ignore_id: int = -100,
        image_size: Tuple[int, int] = (384, 384),
        fraction: float = 1.0,
        apply_augmentation: bool = True,
        cache_images: bool = False,
        limit_samples: Optional[int] = None
    ):
        super().__init__()
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.ignore_id = ignore_id
        self.image_size = image_size
        self.fraction = fraction
        self.apply_augmentation = apply_augmentation and split == "train"
        self.cache_images = cache_images
        
        if self.apply_augmentation:
            self._setup_augmentations()
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Директория разделения не найдена: {self.split_dir}")

        all_samples = self._load_metadata()
        logger.info(f"Загружено {len(all_samples)} записей метаданных из {self.split_dir}")

        if self.fraction < 1.0:
            random.seed(42)
            random.shuffle(all_samples)
            
            num_samples = max(1, int(len(all_samples) * self.fraction))
            self.samples = all_samples[:num_samples]
            logger.info(f"Используется {len(self.samples)} образцов ({self.fraction:.1%} от {len(all_samples)}) для разделения {split}")
        else:
            self.samples = all_samples
            logger.info(f"Загружено {len(self.samples)} образцов для разделения {split}")
        
        if limit_samples is not None and limit_samples > 0:
            self.samples = self.samples[:limit_samples]
            logger.info(f"Ограничено до {len(self.samples)} образцов для отладки")
        
        self.image_cache = {} if cache_images else None
    
    def _setup_augmentations(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.1),
            transforms.RandomRotation(degrees=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Настроены аугментации для обучающих данных")
    
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
                        ground_truth = json.loads(item["ground_truth"]) if isinstance(item["ground_truth"], str) else item["ground_truth"]
                        
                        image_path = self.split_dir / file_name
                        if not image_path.exists():
                            potential_paths = list(self.split_dir.glob(f"{file_name}.*"))
                            if potential_paths:
                                image_path = potential_paths[0]
                            else:
                                logger.warning(f"Изображение не найдено: {image_path} (строка {line_num} в {jsonl_file})")
                                continue
                        
                        if "gt_parse" not in ground_truth:
                            logger.warning(f"Отсутствует gt_parse в ground_truth: {file_name} (строка {line_num} в {jsonl_file})")
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        ground_truth = sample["ground_truth"]
        text = ground_truth["gt_parse"]
        
        if self.image_cache is not None and str(image_path) in self.image_cache:
            image = self.image_cache[str(image_path)]
        else:
            try:
                image = Image.open(image_path).convert("RGB")
                if self.image_cache is not None:
                    self.image_cache[str(image_path)] = image
            except Exception as e:
                logger.error(f"Ошибка при загрузке изображения {image_path}: {e}")
                image = Image.new('RGB', self.image_size, color='white')
        
        if self.apply_augmentation:
            image_tensor = self.transform(image)
        else:
            image_tensor = self.transform(image)
        
        if idx == 0 and not getattr(self, '_logged_sample', False):
            logger.info(f"Путь к изображению: {image_path}, Текст: {text}")
            logger.info(f"Токенизированный текст: {self.processor.tokenizer.tokenize(text)}")
            self._logged_sample = True

        encoding = self.processor.tokenizer(
            text=text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        result = {
            "pixel_values": image_tensor,  # Форма: [C, H, W]
            "labels": encoding["input_ids"].squeeze(0)  # Форма: [seq_len]
        }

        labels = result["labels"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
        result["labels"] = labels
        
        return result
    
    def get_text_sample(self, idx: int) -> str:
        if 0 <= idx < len(self.samples):
            return self.samples[idx]["ground_truth"]["gt_parse"]
        return ""
    
    def visualize_sample(self, idx: int) -> Tuple[Image.Image, str]:
        if 0 <= idx < len(self.samples):
            sample = self.samples[idx]
            image_path = sample["image_path"]
            text = sample["ground_truth"]["gt_parse"]
            try:
                image = Image.open(image_path).convert("RGB")
                return image, text
            except Exception as e:
                logger.error(f"Ошибка при визуализации образца {idx}: {e}")
        return Image.new('RGB', (100, 100), color='white'), "Ошибка"


class TrOCRDataModule:

    def __init__(
        self,
        processor: TrOCRProcessor,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 8,
        max_length: int = 64,
        image_size: Tuple[int, int] = (384, 384),
        fraction: float = 1.0,
        apply_augmentation: bool = True,
        distributed: bool = False,
        pin_memory: bool = True,
        cache_images: bool = False,
        drop_last: bool = True,
        seed: int = 42
    ):
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.image_size = image_size
        self.fraction = fraction
        self.apply_augmentation = apply_augmentation
        self.distributed = distributed
        self.pin_memory = pin_memory
        self.cache_images = cache_images
        self.drop_last = drop_last
        self.seed = seed
        
        self.train_dataset = self._create_dataset("train", apply_augmentation=True)
        self.val_dataset = self._create_dataset("valid", apply_augmentation=False)
        self.test_dataset = self._create_dataset("test", apply_augmentation=False)
        
        logger.info(
            f"Инициализирован модуль данных: "
            f"размер пакета={batch_size}, "
            f"рабочие процессы={num_workers}, "
            f"размер изображения={image_size}, "
            f"макс. длина={max_length}"
        )
    
    def _create_dataset(
        self, 
        split: str, 
        apply_augmentation: bool = False
    ) -> Optional[TrOCRDataset]:
        split_dir = self.data_dir / split
        if not split_dir.exists():
            logger.warning(f"Директория разделения не найдена: {split_dir}")
            return None
        
        try:
            return TrOCRDataset(
                processor=self.processor,
                data_dir=self.data_dir,
                split=split,
                max_length=self.max_length,
                image_size=self.image_size,
                fraction=self.fraction,
                apply_augmentation=apply_augmentation and split == "train",
                cache_images=self.cache_images and split != "train"  # Кэшируем только валидационные/тестовые данные
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Не удалось создать датасет для разделения {split}: {e}")
            return None
    
    def _create_dataloader(
        self, 
        dataset: Optional[TrOCRDataset], 
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
    
    def get_sample_info(self) -> Dict[str, Any]:
        return {
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "steps_per_epoch": len(self.train_dataloader()) if self.train_dataloader() else 0
        }