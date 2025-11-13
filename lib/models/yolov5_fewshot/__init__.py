"""
YOLOv5 Few-Shot Detection Module

This module implements few-shot object detection by:
1. Extracting pretrained YOLOv5 backbone
2. Adding reference encoding mechanism
3. Using similarity-guided detection
"""

from .backbone_model import FewShotYOLOv5, build_fewshot_model
from .reference_encoder import ReferenceEncoder
from .similarity_module import SimilarityModule
from .detection_head import FewShotDetectionHead

__version__ = '1.0.0'

__all__ = [
    'FewShotYOLOv5',
    'build_fewshot_model',
    'ReferenceEncoder',
    'SimilarityModule',
    'FewShotDetectionHead'
]