"""
Models module for FrameReader OCR system.
Contains concrete implementations of OCR models.
"""

from .donut import DonutEncoder, DonutDecoder, DonutOCRModel
from .trocr import TrOCREncoder, TrOCRDecoder, TrOCROCRModel

__all__ = [
    'DonutEncoder',
    'DonutDecoder', 
    'DonutOCRModel',
    'TrOCREncoder',
    'TrOCRDecoder',
    'TrOCROCRModel'
]