"""PolyBert modeling components and architectures.

This module contains the core transformer poly_model implementation including:
- Model configuration (PolyBertConfig)
- Base transformer poly_model (PolyBertModel)
- Task-specific poly_model variants for different NLP applications

The models support flexible architecture configurations including:
- Different attention mechanisms
- Various normalization schemes
- Configurable MLP architectures
- Efficient sequence packing for variable-length inputs

All models are designed to be compatible with HuggingFace transformers
while providing additional flexibility for research and experimentation.
"""

from .config import PolyBertConfig
from .model import (
    PolyBertForMaskedLM,
    PolyBertForQuestionAnswering,
    PolyBertForSequenceClassification,
    PolyBertForTokenClassification,
    PolyBertModel,
)

__all__ = [
    "PolyBertConfig",
    "PolyBertForMaskedLM",
    "PolyBertForQuestionAnswering",
    "PolyBertForSequenceClassification",
    "PolyBertForTokenClassification",
    "PolyBertModel",
]
