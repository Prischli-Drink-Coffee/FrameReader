"""
Models module for FrameReader OCR system.
Contains concrete implementations of OCR models.
"""

from .donut import SwinEncoder as DonutEncoder, BARTDecoder as DonutDecoder, DonutOCRModel
from .trocr import TrOCREncoder, TrOCRDecoder, TrOCROCRModel
from .vision_encoder_decoder import CustomVisionEncoderDecoderModel

__all__ = [
    'DonutEncoder',
    'DonutDecoder', 
    'DonutOCRModel',
    'TrOCREncoder',
    'TrOCRDecoder',
    'TrOCROCRModel',
    'CustomVisionEncoderDecoderModel'
]