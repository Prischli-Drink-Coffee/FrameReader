"""
Image caching utilities for FrameReader OCR system.
"""

import logging
import io
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)


class DataCache:
    def __init__(self, cache_dir: Union[str, Path], max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self._lock = threading.Lock()
    
    def _get_cache_key(self, data: Any) -> str:
        if isinstance(data, (str, bytes)):
            content = data.encode() if isinstance(data, str) else data
        else:
            content = str(data).encode()
        return hashlib.md5(content).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.memory_cache:
                self.cache_stats["hits"] += 1
                return self.memory_cache[key]
            
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    if len(self.memory_cache) < self.max_memory_items:
                        self.memory_cache[key] = data
                    
                    self.cache_stats["hits"] += 1
                    return data
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_path}: {e}")
            
            self.cache_stats["misses"] += 1
            return None
    
    def put(self, key: str, data: Any) -> None:
        with self._lock:
            if len(self.memory_cache) < self.max_memory_items:
                self.memory_cache[key] = data
            
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def get_or_compute(self, key: str, compute_fn: callable) -> Any:
        cached_data = self.get(key)
        if cached_data is not None:
            return cached_data
        
        data = compute_fn()
        self.put(key, data)
        return data
    
    def clear(self) -> None:
        with self._lock:
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        return self.cache_stats.copy()
    
    def get_cache_size(self) -> Dict[str, int]:
        return {
            "memory_items": len(self.memory_cache),
            "disk_files": len(list(self.cache_dir.glob("*.pkl")))
        }


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