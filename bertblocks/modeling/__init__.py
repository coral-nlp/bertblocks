"""BertBlocks modeling components and architectures.

This module contains the core transformer model implementation including:
- Model configuration (BertBlocksConfig)
- Base transformer model (BertBlocksModel)
- Task-specific model variants for different NLP applications

The models support flexible architecture configurations including:
- Different attention mechanisms
- Various normalization schemes
- Configurable MLP architectures
- Efficient sequence packing for variable-length inputs

All models are designed to be compatible with HuggingFace transformers
while providing additional flexibility for research and experimentation.
"""

from .config import BertBlocksConfig
from .model import (
    BertBlocksForMaskedLM,
    BertBlocksForQuestionAnswering,
    BertBlocksForSequenceClassification,
    BertBlocksForTokenClassification,
    BertBlocksModel,
)

__all__ = [
    "BertBlocksConfig",
    "BertBlocksForMaskedLM",
    "BertBlocksForQuestionAnswering",
    "BertBlocksForSequenceClassification",
    "BertBlocksForTokenClassification",
    "BertBlocksModel",
]
