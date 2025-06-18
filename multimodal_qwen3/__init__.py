"""
Multimodal LLM Extension for vLLM

This package extends base LLMs to handle multimodal inputs with arbitrary embeddings.
"""

from .model import MultimodalQwen3Model
from .projector import MultiModalProjector
from .processor import MultimodalProcessor
from .pipeline import MultimodalQwen3Pipeline

__version__ = "0.1.0"
__all__ = [
    "MultimodalQwen3Model",
    "MultiModalProjector", 
    "MultimodalProcessor",
    "MultimodalQwen3Pipeline",
] 