from typing import Any, Literal

from transformers.modeling_utils import PretrainedConfig


class PolyBertConfig(PretrainedConfig):
    """Configuration class for PolyBert models."""

    model_type: str = "polybert"

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_blocks: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        initializer_kind: Literal[
            "trunc_normal", "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform"
        ] = "trunc_normal",
        initializer_range: float = 0.02,
        initializer_cutoff_factor: float = 3.0,
        initializer_gain: float = 1.0,
        actv_fn: Literal["relu", "silu", "gelu", "leakyrelu", "selu", "logsigmoid", "sigmoid", "prelu"] = "silu",
        norm_kind: Literal["pre", "post", "both", "none"] = "pre",
        norm_fn: Literal["group", "layer", "rms", "deep", "dynamictanh"] = "rms",
        norm_eps: float = 1e-12,
        norm_params: dict | None = None,
        pad_token_id: int = 0,
        classifier_dropout: float = 0.1,
        problem_type: Literal["regression", "single_label_classification", "multi_label_classification"] = "regression",
        num_labels: int = 2,
        pos_emb_kind: Literal["alibi", "sinusoidal", "rope"] = "alibi",
        max_sequence_length: int = 512,
        mlp_type: Literal["mlp", "glu"] = "mlp",
        mlp_in_bias: bool = True,
        mlp_out_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PolyBertConfig.

        Args:
            vocab_size: The size of the vocabulary. This determines the number of unique tokens
                the model can process. Common values: 30522 (BERT), 50257 (GPT-2), 32000 (T5).
                Must be greater than 0.
            hidden_size: The dimensionality of the hidden layers. This is the primary dimension
                of the model and affects memory usage and computational requirements.
                Common values: 768 (BERT-base), 1024 (BERT-large). Must be divisible by
                num_attention_heads. Must be greater than 0.
            num_blocks: The number of transformer layers in the model. More layers generally
                improve model capacity but increase computational cost. Common values: 12 (BERT-base),
                24 (BERT-large). Must be at least 1.
            num_attention_heads: The number of attention heads in the multi-head attention mechanism.
                Each head has dimension hidden_size // num_attention_heads. More heads can capture
                different types of relationships. Common values: 12 (BERT-base), 16 (BERT-large).
                Must be at least 1 and hidden_size must be divisible by this value.
            intermediate_size: The dimensionality of the feed-forward layers. This is typically
                4x the hidden_size (e.g., 3072 for hidden_size=768). Must be greater than 0.
            hidden_dropout_prob: Dropout probability applied to hidden layer outputs.
                Common values: 0.1 (default), 0.0 (no dropout). Must be between 0.0 and 1.0.
            attn_dropout_prob: Dropout probability applied to attention weights.
                Common values: 0.1 (default), 0.0 (no dropout). Must be between 0.0 and 1.0.
            initializer_kind: The initialization method for weights. Determines the type of
                distribution random weights are sampled from for initialization.
                Defaults to a truncated normal distribution.
            initializer_range: Standard deviation for weight initialization. Smaller values lead
                to more conservative initialization. Common values: 0.02 (BERT). Must be greater than 0.0.
            initializer_cutoff_factor: Cutoff factor for truncated normal initialization.
                Values beyond initializer_range * initializer_cutoff_factor are redrawn.
                This ensures no extremely large initial weights. Common values: 2.0-3.0.
                Must be greater than 0.0.
            initializer_gain: Gain to scale initialized weights with, e.g., for DeepNorm.
                Must be greater than 0.0.
            actv_fn: The activation function used in feed-forward networks.
            norm_kind: When to apply normalization in the transformer layers. Available options:
                "pre" (Pre-normalization, normalize before attention/FFN, default, more stable),
                "post" (Post-normalization, normalize after attention/FFN, as in original Transformer),
                "both" (Apply normalization both before and after),
                "none" (No normalization, not recommended).
            norm_fn: The type of normalization to apply. Available options:
                "rms" (Root Mean Square Layer Normalization, default, more efficient),
                "layer" (Standard Layer Normalization as used in BERT),
                "group" (Group Normalization, useful for smaller batch sizes),
                "deep" (DeepNorm),
                "dynamictanh" (Dynamic Tanh Normalization).
            norm_eps: Small constant added to variance for numerical stability in normalization.
                Prevents division by zero in layer normalization. Common values: 1e-12 (BERT).
            norm_params: Additional parameters for custom normalization layers. This field allows
                passing custom parameters to normalization layers that require them. For example,
                for DeepNorm: {"alpha": 0.81} where alpha is the scaling factor.
            pad_token_id: The token ID used for padding sequences to the same length.
                This token is ignored during attention computation. Common values: 0 (BERT),
                1 (RoBERTa). Must be non-negative and within the vocabulary range.
            classifier_dropout: Dropout probability for the classification head. Applied to the
                pooled representation before the final classification layer. Helps prevent
                overfitting in downstream tasks. Must be between 0.0 and 1.0.
            problem_type: The problem type for automatic loss selection (HuggingFace standard).
                Automatically selects appropriate loss functions: "regression" (MSE loss for
                continuous targets), "single_label_classification" (CrossEntropy loss for
                single-label problems), "multi_label_classification" (BCEWithLogits loss for
                multi-label problems).
            num_labels: The number of output labels for classification tasks. For regression tasks,
                typically 1. For binary classification, 2. For multi-class classification, the
                number of classes. Must be at least 1. This is the standard HuggingFace field
                name for the number of classes.
            pos_emb_kind: The type of positional embedding to use. Available options:
                "alibi" (ALiBi positional encoding), "sinusoidal" (Sinusoidal positional encoding),
                "rope" (Rotary positional encoding).
            max_sequence_length: Maximum number of tokens the model can process in a single sequence.
                This affects memory usage and determines the size of positional embeddings (if used).
                Common values: 512 (BERT), 1024, 2048. Longer sequences require more memory.
                Must be greater than 0.
            mlp_type: The type of MLP (feed-forward) layer architecture. Available options:
                "mlp" (Standard two-layer feed-forward network),
                "glu" (Gated Linear Unit with learned gating mechanism, typically better performance).
            mlp_in_bias: Whether to include bias terms in the input projection of MLP layers.
                Setting to False can reduce parameters and sometimes improve performance.
                Common values: True (default), False (for efficiency).
            mlp_out_bias: Whether to include bias terms in the output projection of MLP layers.
                Setting to False can reduce parameters and sometimes improve performance.
                Common values: True (default), False (for efficiency).
            **kwargs: Additional keyword arguments passed to the parent PretrainedConfig class.

        """
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.initializer_kind = initializer_kind
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.initializer_gain = initializer_gain
        self.actv_fn = actv_fn
        self.norm_kind = norm_kind
        self.norm_fn = norm_fn
        self.norm_eps = norm_eps
        self.norm_params = norm_params or {}
        self.classifier_dropout = classifier_dropout
        self.problem_type = problem_type
        self.num_labels = num_labels
        self.pos_emb_kind = pos_emb_kind
        self.max_sequence_length = max_sequence_length
        self.mlp_type = mlp_type
        self.mlp_in_bias = mlp_in_bias
        self.mlp_out_bias = mlp_out_bias
