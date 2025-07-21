"""PolyBert: A Flexible Encoder Architecture for Research and Development.

PolyBert is a modular transformer encoder implementation designed for architecture search
and research purposes. It provides a flexible framework for experimenting with
different transformer configurations, attention mechanisms, and training strategies.

This package includes:
- Core model implementations (PolyBertModel, PolyBertConfig)
- Task-specific model heads for various NLP tasks
- Pretraining utilities for masked language modeling
- Flexible architecture components for research

The models are compatible with HuggingFace transformers ecosystem.
"""

from .modeling import (
    PolyBertConfig,
    PolyBertForMaskedLM,
    PolyBertForQuestionAnswering,
    PolyBertForSequenceClassification,
    PolyBertForTokenClassification,
    PolyBertModel,
)

__all__ = [
    "PolyBertConfig",
    "PolyBertModel",
    "PolyBertForMaskedLM",
    "PolyBertForTokenClassification",
    "PolyBertForSequenceClassification",
    "PolyBertForQuestionAnswering",
]
