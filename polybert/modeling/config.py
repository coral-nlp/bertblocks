# Copyright 2025 onwards coral-nlp
# SPDX-License-Identifier: MIT

from typing import Literal

import pydantic
from pydantic import Field, model_validator
from transformers.modeling_utils import PretrainedConfig


class Config:
    """Pydantic configuration class for model validation.

    Attributes:
        arbitrary_types_allowed (bool): Allows arbitrary types in the model.
            This is needed for compatibility with Transformers PretrainedConfig.

    """

    arbitrary_types_allowed = True


class ConfigMixin:
    """Mixin class that provides config access to modules.

    This mixin should be inherited by modules that need access to the model
    configuration. It stores the config as an instance attribute for easy access.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the ConfigMixin.

        Args:
            config (PolyBertConfig): The model configuration object.

        """
        super().__init__()
        self.config = config


@pydantic.dataclasses.dataclass(config=Config)
class PolyBertConfig(PretrainedConfig):
    """Configuration class for PolyBert models.

    This configuration class stores all the configuration parameters for a PolyBert model.
    It extends Transformers' PretrainedConfig to ensure compatibility with the Transformers
    library while adding validation through Pydantic.

    The configuration supports various architectural choices including different attention
    mechanisms, normalization types, activation functions, and positional embeddings.

    Example:
        Basic usage:

        >>> config = PolyBertConfig(
        ...     hidden_size=768,
        ...     num_blocks=12,
        ...     num_attention_heads=12
        ... )

        Custom configuration:

        >>> config = PolyBertConfig(
        ...     vocab_size=50000,
        ...     hidden_size=1024,
        ...     num_blocks=24,
        ...     num_attention_heads=16,
        ...     intermediate_size=4096,
        ...     actv_fn="relu",
        ...     norm="layer",
        ...     pos_emb_type="rope"
        ... )

    Note:
        All field validation is handled by Pydantic. The hidden_size must be divisible
        by num_attention_heads unless an embedding_size is specified.

    """

    model_type: str = Field(default="encoder", description="The type of model architecture")
    """str: The type of model architecture. Always 'encoder' for PolyBert models."""
    vocab_size: int = Field(
        alias="vocabulary_size",
        default=30522,
        description="The vocabulary size of the model",
        gt=0,
    )
    """int: The size of the vocabulary.

    This determines the number of unique tokens the model can process.
    Common values: 30522 (BERT), 50257 (GPT-2), 32000 (T5).
    Must be greater than 0.
    """
    hidden_size: int = Field(default=768, description="The number of hidden units in the model", gt=0)
    """int: The dimensionality of the hidden layers.

    This is the primary dimension of the model and affects memory usage and
    computational requirements. Common values: 768 (BERT-base), 1024 (BERT-large).
    Must be divisible by num_attention_heads. Must be greater than 0.
    """
    num_blocks: int = Field(default=12, description="The number of encoder blocks in the model", ge=1)
    """int: The number of transformer layers in the model.

    More layers generally improve model capacity but increase computational cost.
    Common values: 12 (BERT-base), 24 (BERT-large). Must be at least 1.
    """
    num_attention_heads: int = Field(default=12, description="The number of attention heads in each block", ge=1)
    """int: The number of attention heads in the multi-head attention mechanism.

    Each head has dimension hidden_size // num_attention_heads. More heads can
    capture different types of relationships. Common values: 12 (BERT-base),
    16 (BERT-large). Must be at least 1 and hidden_size must be divisible by this value.
    """
    intermediate_size: int = Field(default=3072, description="The size of the intermediate (feed-forward) layer", gt=0)
    """int: The dimensionality of the feed-forward layers.

    This is typically 4x the hidden_size (e.g., 3072 for hidden_size=768).

    Must be greater than 0.
    """
    hidden_dropout_prob: float = Field(
        default=0.1, description="The dropout probability for hidden layers", ge=0.0, le=1.0
    )
    """float: Dropout probability applied to hidden layer outputs.

    Common values: 0.1 (default), 0.0 (no dropout).

    Must be between 0.0 and 1.0.
    """
    attn_dropout_prob: float = Field(
        default=0.1, description="The dropout probability for attention layers", ge=0.0, le=1.0
    )
    """float: Dropout probability applied to attention weights.

    Common values: 0.1 (default), 0.0 (no dropout).

    Must be between 0.0 and 1.0.
    """
    initializer_kind: Literal[
        "trunc_normal", "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform"
    ] = Field(default="trunc_normal", description="The initialization method for weights")
    """str: The initialization method for weights.

    Determines the type of distribution random weights are sampled from for initialization.

    Defaults to a truncated normal distribution.
    """
    initializer_range: float = Field(
        default=0.02, description="The standard deviation for initializing weights", gt=0.0
    )
    """float: Standard deviation for weight initialization.

    Used for truncated normal initialization of model weights.
    Smaller values lead to more conservative initialization.
    Common values: 0.02 (BERT), 0.01 (more conservative). Must be greater than 0.0.
    """
    initializer_cutoff_factor: float = Field(
        default=3.0, description="The cutoff factor for weight initialization", gt=0.0
    )
    """float: Cutoff factor for truncated normal initialization.

    Values beyond initializer_range * initializer_cutoff_factor are redrawn.
    This ensures no extremely large initial weights. Common values: 2.0-3.0.
    Must be greater than 0.0.
    """
    actv_fn: Literal["relu", "gelu"] = Field(
        default="gelu",
        description="The activation function used in models",
    )
    """str: The activation function used in feed-forward networks.

    Available options:
        - "gelu": Gaussian Error Linear Unit (default, used in BERT)
        - "relu": Rectified Linear Unit (faster but potentially less expressive)
    """
    norm: Literal["group", "layer", "rms"] = Field(default="rms", description="The type of normalization to use")
    """str: The type of normalization to apply.

    Available options:
        - "rms": Root Mean Square Layer Normalization (default, more efficient)
        - "layer": Standard Layer Normalization (as used in BERT)
        - "group": Group Normalization (useful for smaller batch sizes)
    """
    norm_kind: Literal["pre", "post", "both", "none"] = Field(default="pre", description="When to apply normalization")
    """str: When to apply normalization in the transformer layers.

    Available options:
        - "pre": Pre-normalization (normalize before attention/FFN, default, more stable)
        - "post": Post-normalization (normalize after attention/FFN, as in original Transformer)
        - "both": Apply normalization both before and after
        - "none": No normalization (not recommended)
    """
    norm_eps: float = Field(default=1e-12, description="Epsilon value for normalization", gt=0.0)
    """float: Small constant added to variance for numerical stability in normalization.

    Prevents division by zero in layer normalization. Common values: 1e-12 (BERT),
    1e-5 (standard), 1e-6 (more stable). Must be greater than 0.0.
    """
    pad_token_id: int = Field(default=0, description="The token ID used for padding", ge=0)
    """int: The token ID used for padding sequences to the same length.

    This token is ignored during attention computation. Common values: 0 (BERT),
    1 (RoBERTa). Must be non-negative and within the vocabulary range.
    """
    classifier_dropout: float = Field(
        default=0.1, description="The dropout probability for the classifier", ge=0.0, le=1.0
    )
    """float: Dropout probability for the classification head.

    Applied to the pooled representation before the final classification layer.
    Helps prevent overfitting in downstream tasks. Must be between 0.0 and 1.0.
    """
    task: str = Field(default="regression", description="The type of downstream task")
    """str: The type of downstream task the model is configured for.

    Common values: "regression", "classification", "token_classification",
    "question_answering". This affects the model head architecture.
    """
    num_labels: int = Field(default=1, description="The number of labels for classification tasks", ge=1)
    """int: The number of output labels for classification tasks.

    For regression tasks, typically 1. For binary classification, 2.
    For multi-class classification, the number of classes. Must be at least 1.
    """
    pos_emb_type: Literal["alibi", "sinusoidal", "rope"] = Field(
        default="none", description="Type of positional embedding."
    )
    """str: The type of positional embedding to use.

    Available options:
        - "none": No positional embeddings (default)
        - "alibi": Attention with Linear Biases (no learned parameters)
        - "sinusoidal": Fixed sinusoidal positional encodings
        - "rope": Rotary Position Embedding (relative positions)

    Note: "none" is unusual for transformer models and may hurt performance
    on tasks requiring positional understanding.
    """
    max_sequence_length: int = Field(default=512, description="Maximum sequence length", gt=0)
    """int: Maximum number of tokens the model can process in a single sequence.

    This affects memory usage and determines the size of positional embeddings
    (if used). Common values: 512 (BERT), 1024, 2048. Longer sequences require
    more memory. Must be greater than 0.
    """

    @model_validator(mode="after")
    def check_head_dim(self) -> "PolyBertConfig":
        """Validate that hidden_size is divisible by num_attention_heads.

        This validator ensures that the attention head dimension is an integer,
        which is required for the multi-head attention mechanism to work properly.
        The validation is skipped if an embedding_size attribute exists.

        Returns:
            PolyBertConfig: The validated configuration object.

        Raises:
            ValueError: If hidden_size is not divisible by num_attention_heads
                and no embedding_size is defined.

        """
        if self.hidden_size % self.num_attention_heads != 0 and not hasattr(self, "embedding_size"):
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )
        return self
