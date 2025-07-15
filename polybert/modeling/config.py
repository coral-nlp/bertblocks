# Copyright 2025 onwards coral-nlp
# SPDX-License-Identifier: MIT

from typing import Literal

import pydantic
from pydantic import Field, model_validator
from transformers.modeling_utils import PretrainedConfig

# Type aliases for attention configuration
ScoreModType = Literal["none", "alibi", "relative", "rope", "sinusoidal"]
MaskType = Literal["none", "causal", "doc", "sliding", "dilated", "packed_doc"]


class Config:
    arbitrary_types_allowed = True


class ConfigMixin:
    """Mixin class that provides config access to modules."""
    def __init__(self, config: "PolyBertConfig"):
        super().__init__()
        self.config = config


@pydantic.dataclasses.dataclass(config=Config)
class PolyBertConfig(PretrainedConfig):
    model_type: str = Field(default="encoder")  # This is always the case
    vocab_size: int = Field(
        alias="vocabulary_size",
        default=30522,
        description="The vocabulary size of the model",
        gt=0,
    )
    hidden_size: int = Field(default=768, description="The number of hidden units in the model", gt=0)
    num_hidden_layers: int = Field(default=12, description="The number of hidden layers in the model", ge=1)
    num_attention_heads: int = Field(default=12, description="The number of attention heads in each layer", ge=1)
    intermediate_size: int = Field(default=3072, description="The size of the intermediate (feed-forward) layer", gt=0)
    hidden_dropout_prob: float = Field(
        default=0.1, description="The dropout probability for hidden layers", ge=0.0, le=1.0
    )
    attn_dropout_prob: float = Field(
        default=0.1, description="The dropout probability for attention layers", ge=0.0, le=1.0
    )
    initializer_range: float = Field(
        default=0.02, description="The standard deviation for initializing weights", gt=0.0
    )
    initializer_cutoff_factor: float = Field(
        default=3.0, description="The cutoff factor for weight initialization", gt=0.0
    )
    actv_fn: Literal["relu", "gelu"] = Field(
        default="gelu",
        description="The activation function used in models",
    )
    norm: Literal["group", "layer", "rms"] = Field(default="rms", description="The type of normalization to use")
    norm_kind: Literal["pre", "post", "both", "none"] = Field(default="pre", description="When to apply normalization")
    norm_eps: float = Field(default=1e-12, description="Epsilon value for normalization", gt=0.0)
    pad_token_id: int = Field(default=0, description="The token ID used for padding", ge=0)
    classifier_dropout: float = Field(
        default=0.1, description="The dropout probability for the classifier", ge=0.0, le=1.0
    )
    task: str = Field(default="regression", description="The type of downstream task")
    num_labels: int = Field(default=1, description="The number of labels for classification tasks", ge=1)
    attn_score_mod_type: ScoreModType = Field(default="none", description="Type of attention score modification")
    attn_mask_type: MaskType = Field(default="none", description="Type of attention mask to use")
    max_sequence_length: int = Field(default=512, description="Maximum packed sequence length", gt=0)
    enable_sequence_packing: bool = Field(default=False, description="Enable sequence packing for training efficiency")
    document_separator_token_id: int | None = Field(default=None, description="Token ID for document boundaries in packed sequences")

    @model_validator(mode="after")
    def check_head_dim(self):
        if self.hidden_size % self.num_attention_heads != 0 and not hasattr(self, "embedding_size"):
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )
        return self
