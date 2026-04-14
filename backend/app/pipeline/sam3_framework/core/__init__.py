# MIT License - Copyright (c) 2025 SAM3 Project Contributors
"""
SAM3 core model components.

Provides the full model architecture: encoders, decoders, attention,
segmentation heads, vision transformers, and vision-language combiners.
"""

from .decoder import TransformerDecoder, TransformerDecoderLayer
from .encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .geometry_encoders import SequenceGeometryEncoder
from .maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .memory_attention import MemoryAttention, MemoryAttentionLayer
from .model_misc import DotProductScoring, LayerScale, TransformerWrapper
from .necks import Sam3DualViTDetNeck
from .sam import SAM2Model
from .sam3_image import SAM3SemanticModel
from .sam_blocks import PositionEmbeddingSine, RoPEAttention, SAM2TwoWayTransformer
from .sam_decoders import MaskDecoder, SAM2MaskDecoder
from .sam_encoders import ImageEncoderViT, PromptEncoder
from .sam_transformer import Attention, TwoWayAttentionBlock, TwoWayTransformer
from .text_encoder_ve import VETextEncoder
from .tiny_encoder import TinyViT
from .vitdet import ViT
from .vl_combiner import SAM3VLBackbone

__all__ = [
    "TransformerDecoder", "TransformerDecoderLayer",
    "TransformerEncoderFusion", "TransformerEncoderLayer",
    "SequenceGeometryEncoder",
    "PixelDecoder", "UniversalSegmentationHead",
    "MemoryAttention", "MemoryAttentionLayer",
    "DotProductScoring", "LayerScale", "TransformerWrapper",
    "Sam3DualViTDetNeck",
    "SAM2Model",
    "SAM3SemanticModel",
    "PositionEmbeddingSine", "RoPEAttention", "SAM2TwoWayTransformer",
    "MaskDecoder", "SAM2MaskDecoder",
    "ImageEncoderViT", "PromptEncoder",
    "Attention", "TwoWayAttentionBlock", "TwoWayTransformer",
    "VETextEncoder",
    "TinyViT",
    "ViT",
    "SAM3VLBackbone",
]
