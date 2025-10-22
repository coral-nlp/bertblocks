"""BertBlocks is a modular transformer encoder implementation designed for architecture search and research purposes.

This package includes:

    - Core model implementations (BertBlocksModel, BertBlocksConfig)
    - Task-specific model heads for various NLP tasks
    - Pretraining utilities for masked language modeling
    - Wide range of architecture components for research

The models are compatible with HuggingFace transformers ecosystem.

"""

from bertblocks.config import BertBlocksConfig
from bertblocks.integration import from_huggingface
from bertblocks.modeling.model import (
    BertBlocksForMaskedLM,
    BertBlocksForQuestionAnswering,
    BertBlocksForSequenceClassification,
    BertBlocksForTokenClassification,
    BertBlocksModel,
)
from bertblocks.training.modules import (
    BertBlocksFinetuningDataModule,
    BertBlocksFinetuningModule,
    BertBlocksPretrainingDataModule,
    BertBlocksPretrainingModule,
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
    AutoConfig.register("bertblocks", BertBlocksConfig)

    # Register models for automatic loading
    AutoModel.register(BertBlocksConfig, BertBlocksModel)
    AutoModelForMaskedLM.register(BertBlocksConfig, BertBlocksForMaskedLM)
    AutoModelForSequenceClassification.register(BertBlocksConfig, BertBlocksForSequenceClassification)
    AutoModelForTokenClassification.register(BertBlocksConfig, BertBlocksForTokenClassification)
    AutoModelForQuestionAnswering.register(BertBlocksConfig, BertBlocksForQuestionAnswering)
except ImportError:
    # transformers not available - skip registration
    pass

__all__ = [
    "BertBlocksConfig",
    "BertBlocksModel",
    "BertBlocksForMaskedLM",
    "BertBlocksForSequenceClassification",
    "BertBlocksForTokenClassification",
    "BertBlocksForQuestionAnswering",
    "from_huggingface",
    "BertBlocksFinetuningDataModule",
    "BertBlocksDenoisingDataModule",
    "BertBlocksPretrainingDataModule",
    "BertBlocksFinetuningModule",
    "BertBlocksDenoisingModule",
    "BertBlocksPretrainingModule",
]
