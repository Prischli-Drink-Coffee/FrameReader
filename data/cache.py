"""
Image caching utilities for FrameReader OCR system.
"""

import logging
import io
from pathlib import Path
from typing import Dict, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)


class ImageCache:
   def __init__(self, enabled: bool = False):
       self.enabled = enabled
       self._cache: Dict[str, Image.Image] = {} if enabled else None
       logger.info(f"Image cache {'enabled' if enabled else 'disabled'}")
   
   def get(self, path: Union[str, Path]) -> Optional[Image.Image]:
       if not self.enabled or self._cache is None:
           return None
       
       path_str = str(path)
       if path_str in self._cache:
           return self._cache[path_str].copy()
       return None
   
   def set(self, path: Union[str, Path], image: Image.Image) -> None:
       if not self.enabled or self._cache is None:
           return
       
       path_str = str(path)
       self._cache[path_str] = image.copy()
   
   def clear(self) -> None:
       if self.enabled and self._cache is not None:
           self._cache.clear()
           logger.info("Image cache cleared")
   
   def size(self) -> int:
       if not self.enabled or self._cache is None:
           return 0
       return len(self._cache)
   
   def load_image_with_cache(self, image_path: Union[str, Path], image_size: Optional[tuple] = None) -> Image.Image:
       cached_image = self.get(image_path)
       if cached_image is not None:
           return cached_image
       
       try:
           with open(image_path, 'rb') as f:
               image_data = f.read()
           image = Image.open(io.BytesIO(image_data)).convert("RGB")
           
           if image_size:
               image.thumbnail(image_size[::-1], Image.LANCZOS)
           
           self.set(image_path, image)
           return image.copy()
           
       except Exception as e:
           logger.error(f"Error loading image {image_path}: {e}")
           size = image_size[::-1] if image_size else (640, 480)
           return Image.new('RGB', size, color='white')