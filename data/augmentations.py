import random
import logging
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

try:
   import cv2
except ImportError:
   cv2 = None
   logging.warning("OpenCV (cv2) not installed, some augmentations will be unavailable")

logger = logging.getLogger(__name__)


class ImageAugmentator:
   def __init__(self, augmentation_prob: float = 0.5, max_rotation: float = 5.0,
                brightness_range: Tuple[float, float] = (0.8, 1.2), contrast_range: Tuple[float, float] = (0.8, 1.2),
                blur_range: Tuple[int, int] = (0, 2), noise_level: float = 0.05,
                sharpness_range: Tuple[float, float] = (0.8, 1.5), enable_advanced: bool = True):
       self.augmentation_prob = augmentation_prob
       self.max_rotation = max_rotation
       self.brightness_range = brightness_range
       self.contrast_range = contrast_range
       self.blur_range = blur_range
       self.noise_level = noise_level
       self.sharpness_range = sharpness_range
       self.enable_advanced = enable_advanced
       self.augmentations = []
       
       self._setup_augmentations()
       
   def _setup_augmentations(self):
       self.augmentations = []
       
       self.augmentations.append(self._apply_rotation)
       self.augmentations.append(self._apply_brightness)
       self.augmentations.append(self._apply_contrast)
       self.augmentations.append(self._apply_blur)
       self.augmentations.append(self._apply_sharpness)
       self.augmentations.append(self._apply_salt_pepper_noise)
       
       if self.enable_advanced:
           self.augmentations.append(self._apply_perspective)
           if cv2 is not None:
               self.augmentations.append(self._apply_elastic_transform)
           self.augmentations.append(self._apply_color_jitter)
           self.augmentations.append(self._apply_cutout)
       
       logger.info(f"Setup {len(self.augmentations)} augmentation types")
   
   def _apply_rotation(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob:
           angle = random.uniform(-self.max_rotation, self.max_rotation)
           return TF.rotate(img, angle)
       return img
   
   def _apply_brightness(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob:
           brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
           return TF.adjust_brightness(img, brightness_factor)
       return img
   
   def _apply_contrast(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob:
           contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
           return TF.adjust_contrast(img, contrast_factor)
       return img
   
   def _apply_blur(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob and cv2 is not None:
           img_np = np.array(img)
           kernel_size = random.randint(self.blur_range[0], self.blur_range[1]) * 2 + 1
           if kernel_size > 1:
               img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
           return Image.fromarray(img_np)
       return img
   
   def _apply_sharpness(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob:
           sharpness_factor = random.uniform(self.sharpness_range[0], self.sharpness_range[1])
           return TF.adjust_sharpness(img, sharpness_factor)
       return img
   
   def _apply_salt_pepper_noise(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob:
           img_np = np.array(img)
           h, w, c = img_np.shape
           salt_mask = np.random.random((h, w)) < self.noise_level/2
           img_np[salt_mask] = 255
           pepper_mask = np.random.random((h, w)) < self.noise_level/2
           img_np[pepper_mask] = 0
           return Image.fromarray(img_np)
       return img
   
   def _apply_perspective(self, img: Image.Image) -> Image.Image:
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
   
   def _apply_elastic_transform(self, img: Image.Image) -> Image.Image:
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
           distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
           return Image.fromarray(distorted)
       return img
   
   def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
       if random.random() < self.augmentation_prob:
           hue_factor = random.uniform(-0.1, 0.1)
           saturation_factor = random.uniform(0.5, 1.5)
           
           img = TF.adjust_hue(img, hue_factor)
           img = TF.adjust_saturation(img, saturation_factor)
           return img
       return img
   
   def _apply_cutout(self, img: Image.Image) -> Image.Image:
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
   
   def apply(self, img: Image.Image) -> Image.Image:
       for augmentation_fn in self.augmentations:
           img = augmentation_fn(img)
       return img


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