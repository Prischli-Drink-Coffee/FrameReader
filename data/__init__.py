"""
FrameReader Data Module

This module contains dataset implementations, augmentations, and caching utilities
for the FrameReader OCR system.
"""

from .dataset import DonutDataset, TrOCRDataset
from .augmentations import ImageAugmentator
from .cache import ImageCache

__all__ = [
   "DonutDataset",
   "TrOCRDataset", 
   "ImageAugmentator",
   "ImageCache"
]