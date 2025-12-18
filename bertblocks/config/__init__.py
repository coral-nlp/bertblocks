from collections.abc import Sequence
from typing import Any, Literal

from transformers import AutoTokenizer
from transformers.modeling_utils import PretrainedConfig

NormalizationFunction = Literal["group", "layer", "rms", "deep", "dynamictanh"]
NormalizationPosition = Literal["pre", "post", "both", "none"]
ActivationFunction = Literal["relu", "silu", "gelu", "leakyrelu", "selu", "logsigmoid", "sigmoid", "prelu"]
Head = Literal["proj", "mlp", "glu"]
FeedForward = Literal["linear", "mlp", "glu"]
KeywordArgs = dict[str, Any]
Initializer = Literal["trunc_normal", "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform"]
AttentionBackend = Literal["flash_attention_2", "eager", "sdpa"]
PositionalEmbedding = Literal["alibi", "sinusoidal", "rope", "relative", "learned", "learned_alibi"]


class BertBlocksConfig(PretrainedConfig):
    """Configuration class for BertBlocks models.

    Attributes:
        model_type (str): model type name for Huggingface config resolution. Default: 'bertblocks'

    Args:
        vocab_size: The size of the vocabulary. This determines the number of unique tokens
            the model can process. Common values: 30522 (BERT), 50257 (GPT-2), 32000 (T5).
            Must be greater than 0.
        max_sequence_length: Maximum number of tokens the model can process in a single sequence.
            This affects memory usage and determines the size of positional embeddings (if used).
            Common values: 512 (BERT), 1024, 2048. Longer sequences require more memory.
            Must be greater than 0.
        pad_token_id: The token ID used for padding sequences to the same length.
            This token is ignored during attention computation. Common values: 0 (BERT),
            1 (RoBERTa). Must be non-negative and within the vocabulary range.
        mask_token_id: The token ID used for masking tokens. Must be non-negative and within the vocabulary range.
        hidden_size: The dimensionality of the hidden layers. This is the primary dimension
            of the model and affects memory usage and computational requirements.
            Common values: 768 (BERT-base), 1024 (BERT-large). Must be divisible by
            num_attention_heads. Must be greater than 0.
        intermediate_size: The dimensionality of the feed-forward layers. This is typically
            4x the hidden_size (e.g., 3072 for hidden_size=768). Must be greater than 0.
        num_blocks: The number of transformer layers in the model. More layers generally
            improve model capacity but increase computational cost. Common values: 12 (BERT-base),
            24 (BERT-large). Must be at least 1.
        num_attention_heads: The number of attention heads in the multi-head attention mechanism.
            Each head has dimension hidden_size // num_attention_heads. More heads can capture
            different types of relationships. Common values: 12 (BERT-base), 16 (BERT-large).
            Must be at least 2 and hidden_size must be divisible by this value.
        pos_emb_kind: The type of positional embedding to use. Available options:
            "alibi" (ALiBi positional encoding), "sinusoidal" (Sinusoidal positional encoding),
            "rope" (Rotary positional encoding), "relative" (Relative positional encoding),
            "learned" (Learned positional encoding), "learned_alibi" (ALiBi positional encoding with linear layer).
        pos_emb_kwargs: Additional keyword arguments to pass to the positional embedding class. Values dependent
            on chosen pos_emb_kind. All positional embeddings receive `dim` and `max_seq_len` automatically, these
            do not need to be specified.
        add_token_type_emb: Whether to add token type embeddings to the model.
        type_vocab_size: The size of the token_type vocabulary. Only used if add_token_type_emb is True.
        mlp_type: The type of MLP (feed-forward) layer architecture. Available options:
            "mlp" (Standard two-layer feed-forward network),
            "glu" (Gated Linear Unit with learned gating mechanism, typically better performance).
        head_type: The type of MLP (feed-forward) layer architecture for the final head. Available options:
            "proj" (Simple one-layer feed-forward network),
            "mlp" (Standard two-layer feed-forward network),
            "glu" (Gated Linear Unit with learned gating mechanism, typically better performance).
        mlp_in_bias: Whether to include bias terms in the input projection of MLP layers.
        mlp_out_bias: Whether to include bias terms in the output projection of MLP layers.
        attn_proj_bias: Whether to include bias terms in the qkv projection of attention layers.
        attn_out_bias: Whether to include bias terms in the output projection of attention layers.
        local_attention: Whether to include local attention mechanism. Default (-1, -1) means global attention.
        global_attention_every_n_layers: The layer step size for global attention.
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
        norm_bias: Whether to include bias terms in the output projection of normalization layers.
        norm_scaling: Whether norm scaling should be enabled. Defaults to False.
        norm_qk: Whether to apply query-key normalization.
        include_final_norm: Whether to apply a final normalization of the last hidden state.
        emb_dropout_prob: Dropout probability applied to the embedding layer output.
            Common values: 0.1 (default), 0.0 (no dropout). Must be between 0.0 and 1.0.
        hidden_dropout_prob: Dropout probability applied to hidden layer outputs.
            Common values: 0.1 (default), 0.0 (no dropout). Must be between 0.0 and 1.0.
        attn_dropout_prob: Dropout probability applied to attention weights.
            Common values: 0.1 (default), 0.0 (no dropout). Must be between 0.0 and 1.0.
        classifier_dropout_prob: Dropout probability for the classification head. Applied to the
            pooled representation before the final classification layer. Helps prevent
            overfitting in downstream tasks. Must be between 0.0 and 1.0.
        attn_implementation: Which backend implementation of attention to use; can be "flash_attention_2" for
            FlashAttention2, "sdpa" torch, or "eager" for manual implementation. Defaults to SDPA.
        problem_type: The problem type for automatic loss selection (HuggingFace standard).
            Automatically selects appropriate loss functions: "regression" (MSE loss for
            continuous targets), "single_label_classification" (CrossEntropy loss for
            single-label problems), "multi_label_classification" (BCEWithLogits loss for
            multi-label problems).
        num_classes: The number of output classes for classification tasks. For regression tasks,
            typically 1. For binary classification, 2. For multi-class classification, the
            number of classes. Must be at least 1.

        **kwargs: Additional keyword arguments passed to the parent PretrainedConfig class.

    """

    model_type: str = "bertblocks"

    def __init__(
        self,
        vocab_size: int = 30522,
        max_sequence_length: int = 512,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        num_blocks: int = 12,
        attn_implementation: AttentionBackend | None = None,
        local_attention: tuple[int, int] = (-1, -1),
        global_attention_every_n_layers: int = 0,
        initializer_kind: Initializer = "trunc_normal",
        initializer_range: float = 0.02,
        initializer_cutoff_factor: float = 3.0,
        initializer_gain: float = 1.0,
        add_token_type_emb: bool = False,
        type_vocab_size: int = 1,
        head_type: Head = "mlp",
        include_final_norm: bool = True,
        residual_first_layer: bool = False,
        emb_dropout_prob: float = 0.1,
        actv_fn: ActivationFunction = "silu",
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        pos_emb_kind: PositionalEmbedding = "alibi",
        pos_emb_kwargs: KeywordArgs | None = None,
        mlp_type: FeedForward = "mlp",
        mlp_in_bias: bool = True,
        mlp_out_bias: bool = True,
        attn_proj_bias: bool = True,
        attn_out_bias: bool = True,
        norm_kind: NormalizationPosition = "pre",
        norm_fn: NormalizationFunction = "rms",
        norm_eps: float = 1e-12,
        norm_bias: bool = True,
        norm_scaling: bool = False,
        norm_qk: bool = True,
        norm_params: KeywordArgs | None = None,
        hidden_dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        # Downstream task parameters
        classifier_dropout_prob: float = 0.1,
        problem_type: Literal["regression", "single_label_classification", "multi_label_classification"] = "regression",
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        self._validate(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_blocks=num_blocks,
            initializer_range=initializer_range,
            initializer_cutoff_factor=initializer_cutoff_factor,
            initializer_gain=initializer_gain,
            emb_dropout_prob=emb_dropout_prob,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            classifier_dropout_prob=classifier_dropout_prob,
            num_classes=num_classes,
        )

        if attn_implementation is None:
            attn_implementation = "sdpa"

        super().__init__(**kwargs)
        # Input parameters
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        # General architecture parameters
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.pos_emb_kind = pos_emb_kind
        self.pos_emb_kwargs = pos_emb_kwargs or {}
        self.add_token_type_emb = add_token_type_emb
        self.type_vocab_size = type_vocab_size
        self.residual_first_layer = residual_first_layer
        # MLP parameters
        self.mlp_type = mlp_type
        self.head_type = head_type
        self.mlp_in_bias = mlp_in_bias
        self.mlp_out_bias = mlp_out_bias
        # Attention parameters
        self.attn_proj_bias = attn_proj_bias
        self.attn_out_bias = attn_out_bias
        self.local_attention = local_attention
        self.global_attention_every_n_layers = global_attention_every_n_layers
        # Initialization parameters
        self.initializer_kind = initializer_kind
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.initializer_gain = initializer_gain
        # Activation functions
        self.actv_fn = actv_fn
        # Normalization parameters
        self.norm_kind = norm_kind
        self.norm_fn = norm_fn
        self.norm_eps = norm_eps
        self.norm_params = norm_params or {}
        self.norm_bias = norm_bias
        self.norm_scaling = norm_scaling
        self.include_final_norm = include_final_norm
        self.norm_qk = norm_qk
        # Dropout parameters
        self.emb_dropout_prob = emb_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.classifier_dropout = classifier_dropout_prob
        # Implementation details
        self._attn_implementation = attn_implementation
        # Downstream task parameters
        self.problem_type = problem_type
        self.num_classes = num_classes

        # Dependent parameters
        self._unpadding = self._attn_implementation == "flash_attention_2"
        self.pos_emb_kwargs.update(
            {"dim": self.hidden_size // self.num_attention_heads, "max_seq_len": self.max_sequence_length}
        )

    @staticmethod
    def _validate(
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        mask_token_id: int,
        hidden_size: int,
        intermediate_size: int,
        num_blocks: int,
        initializer_range: float,
        initializer_cutoff_factor: float,
        initializer_gain: float,
        emb_dropout_prob: float,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        classifier_dropout_prob: float,
        num_classes: int,
    ) -> None:
        """Validate BertBlocksConfig parameters using native Python."""
        # Scalar parameter validation
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be greater than 0, got {vocab_size}")

        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be greater than 0, got {max_sequence_length}")

        if num_blocks < 1:
            raise ValueError(f"num_blocks must be at least 1, got {num_blocks}")

        if initializer_range <= 0.0:
            raise ValueError(f"initializer_range must be greater than 0.0, got {initializer_range}")

        if initializer_cutoff_factor <= 0.0:
            raise ValueError(f"initializer_cutoff_factor must be greater than 0.0, got {initializer_cutoff_factor}")

        if initializer_gain <= 0.0:
            raise ValueError(f"initializer_gain must be greater than 0.0, got {initializer_gain}")

        if not 0.0 <= emb_dropout_prob <= 1.0:
            raise ValueError(f"emb_dropout_prob must be between 0.0 and 1.0, got {emb_dropout_prob}")

        if not 0.0 <= classifier_dropout_prob <= 1.0:
            raise ValueError(f"classifier_dropout_prob must be between 0.0 and 1.0, got {classifier_dropout_prob}")

        if num_classes < 1:
            raise ValueError(f"num_classes must be at least 1, got {num_classes}")

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be greater than 0, got {hidden_size}")

        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be greater than 0, got {intermediate_size}")

        if num_attention_heads <= 1:
            raise ValueError(f"num_attention_heads must be at least 2, got {num_attention_heads}")

        if not 0.0 <= hidden_dropout_prob <= 1.0:
            raise ValueError(f"hidden_dropout_prob must be between 0.0 and 1.0, got {hidden_dropout_prob}")

        if not 0.0 <= attn_dropout_prob <= 1.0:
            raise ValueError(f"attn_dropout_prob must be between 0.0 and 1.0, got {attn_dropout_prob}")

        if pad_token_id < 0:
            raise ValueError(f"pad_token_id must be non-negative, got {pad_token_id}")

        if pad_token_id >= vocab_size:
            raise ValueError(f"pad_token_id ({pad_token_id}) must be within vocabulary range (0 to {vocab_size - 1})")

        if mask_token_id < 0:
            raise ValueError(f"mask_token_id must be non-negative, got {mask_token_id}")

        if mask_token_id >= vocab_size:
            raise ValueError(f"mask_token_id ({mask_token_id}) must be within vocabulary range (0 to {vocab_size - 1})")

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even")


class BertConfig(BertBlocksConfig):
    """BertBlocksConfig with default arguments applied for Bert architecture."""

    model_type = "bert"

    def __init__(
        self,
        vocab_size: int = 28996,
        max_sequence_length: int = 512,
        pad_token_id: int = 0,
        mask_token_id: int = 103,
        hidden_size: int = 768,
        num_blocks: int = 12,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        pos_emb_kind: Literal["learned", "absolute", "relative"] = "absolute",
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        actv_fn: Literal["relu", "silu", "gelu", "leakyrelu", "selu", "logsigmoid", "sigmoid", "prelu"] = "gelu",
        norm_eps: float = 1e-12,
        emb_dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        classifier_dropout_prob: float = 0.1,
        attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = "flash_attention_2",
    ):
        super().__init__(
            # User-configurable args
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            pos_emb_kind="learned" if pos_emb_kind in ("absolute", "learned") else "relative",
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            actv_fn=actv_fn,
            norm_eps=norm_eps,
            emb_dropout_prob=emb_dropout_prob or 0.0,
            attn_dropout_prob=attn_dropout_prob or 0.0,
            hidden_dropout_prob=hidden_dropout_prob or 0.0,
            classifier_dropout_prob=classifier_dropout_prob or 0.0,
            attn_implementation=attn_implementation,
            # Hardcoded args for architecture compatibility
            residual_first_layer=False,
            add_token_type_emb=True,
            mlp_type="mlp",
            mlp_in_bias=True,
            mlp_out_bias=True,
            attn_proj_bias=True,
            attn_out_bias=True,
            initializer_kind="trunc_normal",
            initializer_cutoff_factor=4.0,
            initializer_gain=1.0,
            norm_kind="post",
            norm_fn="layer",
            norm_bias=True,
            norm_qk=False,
        )

    @classmethod
    def from_huggingface(
        cls,
        pretrained_model_name_or_path: str,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa",
    ) -> "BertConfig":
        """Instantiate an equivalent BertBlocks BertConfig from a pretrained HuggingFace config."""
        from transformers import BertConfig

        orig_config = BertConfig.from_pretrained(pretrained_model_name_or_path)

        if not hasattr(orig_config, "mask_token_id"):
            tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            mask_token_id = tok.mask_token_id
        else:
            mask_token_id = orig_config.mask_token_id

        return cls(
            vocab_size=orig_config.vocab_size,
            max_sequence_length=orig_config.max_position_embeddings,
            pad_token_id=orig_config.pad_token_id,
            mask_token_id=mask_token_id,
            hidden_size=orig_config.hidden_size,
            num_blocks=orig_config.num_hidden_layers,
            intermediate_size=orig_config.intermediate_size,
            num_attention_heads=orig_config.num_attention_heads,
            pos_emb_kind=orig_config.position_embedding_type,
            type_vocab_size=orig_config.type_vocab_size,
            initializer_range=orig_config.initializer_range,
            actv_fn=orig_config.hidden_act,
            norm_eps=orig_config.layer_norm_eps,
            emb_dropout_prob=orig_config.hidden_dropout_prob or 0.0,
            attn_dropout_prob=orig_config.attention_probs_dropout_prob or 0.0,
            hidden_dropout_prob=orig_config.hidden_dropout_prob or 0.0,
            classifier_dropout_prob=orig_config.classifier_dropout or 0.0,
            attn_implementation=attn_implementation,
        )


class ModernBertConfig(BertBlocksConfig):
    """BertBlocksConfig with default arguments applied for ModernBert architecture."""

    model_type = "modernbert"

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        mask_token_id: int,
        hidden_size: int,
        num_blocks: int,
        intermediate_size: int,
        num_attention_heads: int,
        pos_emb_kwargs: dict[str, Any],
        mlp_in_bias: bool,
        mlp_out_bias: bool,
        attn_proj_bias: bool,
        attn_out_bias: bool,
        local_attention: tuple[int, int],
        global_attention_every_n_layers: int,
        initializer_range: float,
        actv_fn: Literal["relu", "silu", "gelu", "leakyrelu", "selu", "logsigmoid", "sigmoid", "prelu"],
        norm_eps: float,
        norm_bias: bool,
        emb_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_dropout_prob: float,
        classifier_dropout_prob: float,
        attn_implementation: Literal["flash_attention_2", "eager", "sdpa"] = "flash_attention_2",
    ):
        super().__init__(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            pos_emb_kwargs=pos_emb_kwargs,
            mlp_in_bias=mlp_in_bias,
            mlp_out_bias=mlp_out_bias,
            attn_proj_bias=attn_proj_bias,
            attn_out_bias=attn_out_bias,
            local_attention=local_attention,
            global_attention_every_n_layers=global_attention_every_n_layers,
            initializer_range=initializer_range,
            actv_fn=actv_fn,
            norm_eps=norm_eps,
            norm_bias=norm_bias,
            emb_dropout_prob=emb_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            classifier_dropout_prob=classifier_dropout_prob,
            attn_implementation=attn_implementation,
            # Hard-coded values for architecture compatibility
            residual_first_layer=True,
            pos_emb_kind="rope",
            add_token_type_emb=False,
            mlp_type="glu",
            initializer_kind="trunc_normal",
            initializer_cutoff_factor=4.0,
            initializer_gain=1.0,
            norm_kind="pre",
            norm_fn="layer",
            include_final_norm=True,
        )

    @classmethod
    def from_huggingface(
        cls,
        pretrained_model_name_or_path: str,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
    ) -> "ModernBertConfig":
        """Instantiate an equivalent BertBlocks ModernBertConfig from a pretrained HuggingFace config."""
        from transformers import ModernBertConfig

        orig_config = ModernBertConfig.from_pretrained(pretrained_model_name_or_path)
        if not hasattr(orig_config, "mask_token_id"):
            tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            mask_token_id = tok.mask_token_id
        else:
            mask_token_id = orig_config.mask_token_id

        return cls(
            vocab_size=orig_config.vocab_size,
            max_sequence_length=orig_config.max_position_embeddings,
            pad_token_id=orig_config.pad_token_id,
            mask_token_id=mask_token_id,
            hidden_size=orig_config.hidden_size,
            num_blocks=orig_config.num_hidden_layers,
            intermediate_size=orig_config.intermediate_size,
            num_attention_heads=orig_config.num_attention_heads,
            pos_emb_kwargs={
                "base_global": orig_config.global_rope_theta,
                "base_local": orig_config.local_rope_theta,
                "scale": 1,
            },
            mlp_in_bias=orig_config.mlp_bias,
            mlp_out_bias=orig_config.mlp_bias,
            attn_proj_bias=orig_config.attention_bias,
            attn_out_bias=orig_config.attention_bias,
            local_attention=(
                (orig_config.local_attention // 2, orig_config.local_attention // 2)
                if orig_config.local_attention != -1
                else (-1, -1)
            ),
            global_attention_every_n_layers=orig_config.global_attn_every_n_layers,
            initializer_range=orig_config.initializer_range,
            actv_fn=orig_config.hidden_activation or "gelu",
            norm_eps=orig_config.layer_norm_eps,
            norm_bias=orig_config.norm_bias,
            emb_dropout_prob=orig_config.embedding_dropout or 0.0,
            attn_dropout_prob=orig_config.attention_dropout or 0.0,
            hidden_dropout_prob=orig_config.mlp_dropout or 0.0,
            classifier_dropout_prob=orig_config.classifier_dropout or 0.0,
            attn_implementation=attn_implementation,
        )


__all__ = ["BertBlocksConfig", "BertConfig", "ModernBertConfig"]
