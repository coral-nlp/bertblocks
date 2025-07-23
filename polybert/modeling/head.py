from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.activations import get_actv_fn
from polybert.modeling.norms import get_norm


class PolyBertPooler(nn.Module):
    """Pooling layer for PolyBert.

    Applies a linear layer and activation function to the first token of the last hidden state.
    """

    def __init__(self, config: "PolyBertConfig") -> None:
        super().__init__()
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size)
        self.actv = get_actv_fn(config)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the pooling layer."""
        x = self.ffwd(x[:, 0])
        x = self.actv(x)
        return x


class PolyBertGLUPredictionHead(nn.Module):
    """Prediction head for PolyBert poly_model with gated activation.

    This class implements a prediction head that uses a gated linear unit (GLU)
    architecture. It projects the hidden states to an expanded dimension, applies
    gating with an activation function, and includes optional pre-/post-normalization.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the prediction head.

        Args:
            config: PolyBert configuration object containing:
                - hidden_size: Dimensionality of the hidden states
                - actv_fn: Activation function type for gating
                - norm_kind: Normalization placement ("pre", "post", "both", or "none")
                - norm_type: Type of normalization layer
                - norm_eps: Epsilon for numerical stability in normalization

        """
        super().__init__()
        self.ffwd = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False)
        self.actv = get_actv_fn(config)
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the prediction head.

        Args:
            x: Input tensor. Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Transformed tensor after gated projection and normalization.
            Shape: (batch_size, sequence_length, hidden_size)

        """
        x = self.pre_norm(x)
        x, gate = self.ffwd(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.post_norm(x)
        return x


class PolyBertMLPPredictionHead(nn.Module):
    """MLP Prediction head for PolyBert poly_model.

    This class implements a traditional MLP prediction head. It projects the hidden states
    to an expanded dimension, and then projects it back down to the original dimension.
    Includes optional pre-/post-normalization.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the prediction head.

        Args:
            config: PolyBert configuration object containing:
                - hidden_size: Dimensionality of the hidden states
                - actv_fn: Activation function type for gating
                - norm_kind: Normalization placement ("pre", "post", "both", or "none")
                - norm_type: Type of normalization layer
                - norm_eps: Epsilon for numerical stability in normalization

        """
        super().__init__()
        # Upwards projection to intermediate size
        self.Uprj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        # Activation function
        self.actv = get_actv_fn(config)
        # Downwards projection to hidden size
        self.Dprj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)
        # Prenorm (if configured, nn.Identity otherwise)
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        # Postnorm (if configured, nn.Identity otherwise)
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the prediction head.

        Args:
            x: Input tensor. Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Transformed tensor after projection and normalization.
            Shape: (batch_size, sequence_length, hidden_size)

        """
        x = self.pre_norm(x)
        x = self.Uprj(x)
        x = self.actv(x)
        x = self.Dprj(x)
        x = self.post_norm(x)
        return x


def get_prediction_head(config: "PolyBertConfig") -> nn.Module:
    """Get the prediction head layer specified in the configuration.

    This factory function returns the appropriate prediction head architecture
    based on the configuration. Supports both standard MLP and GLU variants.

    Args:
        config (PolyBertConfig): Configuration object determining poly_model hyperparameters.

    Returns:
        An prediction head module (nn.Module) that can transform hidden states.

    Raises:
        ValueError: If the specified prediction head type is not supported.

    Supported prediction head types:
        - `mlp`: Standard two-layer feedforward network
        - `glu`: Gated Linear Unit

    """
    mlp_type = getattr(config, "mlp_type", "mlp")  # Default to mlp for backward compatibility

    if mlp_type == "mlp":
        return PolyBertMLPPredictionHead(config)
    elif mlp_type == "glu":
        return PolyBertGLUPredictionHead(config)
    else:
        supported_types = ["mlp", "glu"]
        raise ValueError(f"Unknown MLP type '{mlp_type}'. " f"Supported types: {', '.join(supported_types)}")
