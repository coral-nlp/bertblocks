"""PolyBert: A Flexible Encoder Architecture for Research and Development.

PolyBert is a modular transformer encoder implementation designed for architecture search
and research purposes. It provides a flexible framework for experimenting with
different transformer configurations, attention mechanisms, and training strategies.

This package includes:
- Core poly_model implementations (PolyBertModel, PolyBertConfig)
- Task-specific poly_model heads for various NLP tasks
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

# Register models with HuggingFace AutoModel system
try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    # Register configuration
    AutoConfig.register("polybert", PolyBertConfig)

    # Register models for automatic loading
    AutoModel.register(PolyBertConfig, PolyBertModel)
    AutoModelForMaskedLM.register(PolyBertConfig, PolyBertForMaskedLM)
    AutoModelForSequenceClassification.register(PolyBertConfig, PolyBertForSequenceClassification)
    AutoModelForTokenClassification.register(PolyBertConfig, PolyBertForTokenClassification)
    AutoModelForQuestionAnswering.register(PolyBertConfig, PolyBertForQuestionAnswering)
except ImportError:
    # transformers not available - skip registration
    pass

__all__ = [
    "PolyBertConfig",
    "PolyBertModel",
    "PolyBertForMaskedLM",
    "PolyBertForSequenceClassification",
    "PolyBertForTokenClassification",
    "PolyBertForQuestionAnswering",
]
