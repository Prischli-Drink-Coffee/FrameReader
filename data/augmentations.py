import random
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms.functional as TF
from torchvision import transforms

try:
   import cv2
except ImportError:
   cv2 = None
   logging.warning("OpenCV (cv2) not installed, some augmentations will be unavailable")

logger = logging.getLogger(__name__)


class ImageAugmentations:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prob = config.get('augmentation_prob', 0.3)
        self.max_rotation = config.get('max_rotation', 5.0)
        self.noise_level = config.get('noise_level', 0.02)
        self.color_jitter = config.get('color_jitter', 0.1)
        self.elastic_transform = config.get('elastic_transform', True)
        self.random_perspective = config.get('random_perspective', True)
    
    def apply_augmentations(self, image: Image.Image) -> Image.Image:
        if not self.config.get('apply_augmentation', True):
            return image
        
        if random.random() > self.prob:
            return image
        
        augmented = image.copy()
        
        if random.random() < 0.3:
            augmented = self._rotate(augmented)
        
        if random.random() < 0.2:
            augmented = self._add_noise(augmented)
        
        if random.random() < 0.3:
            augmented = self._color_jitter(augmented)
        
        if random.random() < 0.2:
            augmented = self._blur(augmented)
        
        if random.random() < 0.1:
            augmented = self._perspective_transform(augmented)
        
        return augmented
    
    def _rotate(self, image: Image.Image) -> Image.Image:
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        return image.rotate(angle, expand=True, fillcolor='white')
    
    def _add_noise(self, image: Image.Image) -> Image.Image:
        np_image = np.array(image)
        noise = np.random.normal(0, self.noise_level * 255, np_image.shape)
        noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    
    def _color_jitter(self, image: Image.Image) -> Image.Image:
        brightness_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        contrast_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        saturation_factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation_factor)
        
        return image
    
    def _blur(self, image: Image.Image) -> Image.Image:
        blur_radius = random.uniform(0.5, 1.5)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def _perspective_transform(self, image: Image.Image) -> Image.Image:
        if not self.random_perspective:
            return image
        
        width, height = image.size
        margin = min(width, height) * 0.05
        
        original_points = [
            (0, 0), (width, 0), (width, height), (0, height)
        ]
        
        new_points = [
            (random.uniform(0, margin), random.uniform(0, margin)),
            (width - random.uniform(0, margin), random.uniform(0, margin)),
            (width - random.uniform(0, margin), height - random.uniform(0, margin)),
            (random.uniform(0, margin), height - random.uniform(0, margin))
        ]
        
        try:
            from PIL.Image import Transform
            return image.transform(
                (width, height),
                Transform.PERSPECTIVE,
                self._get_perspective_coeffs(original_points, new_points),
                fillcolor='white'
            )
        except:
            return image
    
    def _get_perspective_coeffs(self, original_points, new_points):
        matrix = []
        for p1, p2 in zip(original_points, new_points):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        
        A = np.array(matrix, dtype=np.float32)
        B = np.array([p[0] for p in new_points] + [p[1] for p in new_points], dtype=np.float32)
        
        try:
            coeffs = np.linalg.solve(A, B)
            return coeffs.tolist()
        except:
            return [1, 0, 0, 0, 1, 0, 0, 0]


class AdaptiveAugmentations(ImageAugmentations):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.epoch_factor = 1.0
        self.loss_factor = 1.0
    
    def update_factors(self, current_epoch: int, max_epochs: int, current_loss: float):
        self.epoch_factor = 1.0 - (current_epoch / max_epochs) * 0.5
        self.loss_factor = min(2.0, max(0.5, current_loss))
    
    def apply_augmentations(self, image: Image.Image) -> Image.Image:
        adaptive_prob = self.prob * self.epoch_factor * self.loss_factor
        
        if random.random() > adaptive_prob:
            return image
        
        return super().apply_augmentations(image)


class TrOCRAugmentator:
   def __init__(self, image_size: Tuple[int, int] = (384, 384)):
       self.image_size = image_size
       self.transform = transforms.Compose([
           transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
           transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
           transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.1),
           transforms.RandomRotation(degrees=1),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])
       logger.info("Setup TrOCR augmentations for training data")
   
   def apply(self, img: Image.Image):
       return self.transform(img)


class NoAugmentationTransform:
   def __init__(self, image_size: Tuple[int, int] = (384, 384)):
       self.image_size = image_size
       self.transform = transforms.Compose([
           transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])
   
   def apply(self, img: Image.Image):
       return self.transform(img)