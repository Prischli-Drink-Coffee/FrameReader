"""
Core module for FrameReader OCR system.
Provides base classes and interfaces for modular OCR pipeline.
"""

from .base import BaseEncoder, BaseDecoder, BaseOCRModel
from .config import ModelConfig, TrainingConfig, DataConfig

__all__ = [
    'BaseEncoder',
    'BaseDecoder', 
    'BaseOCRModel',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig'
]