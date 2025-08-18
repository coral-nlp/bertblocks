from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from bertblocks.modeling.config import BertBlocksConfig

from torch import nn

from bertblocks.modeling.activations import get_actv_fn
from bertblocks.modeling.norms import get_norm


class BertBlocksPooler(nn.Module):
    """Pooling layer for BertBlocks.

    Applies a linear layer and activation function to the first token of the last hidden state.
    """

    def __init__(self, config: "BertBlocksConfig") -> None:
        """Initialize the pooling layer.

        Args:
            config (BertBlocksConfig): Configuration object containing:
                - hidden_size: Dimensionality of hidden layers
                - actv_fn: Activation function used in feed-forward networks

        """
        super().__init__()
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size)
        self.actv = get_actv_fn(config)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the pooling layer.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): Input hidden states.

        Returns:
            torch.Tensor: Pooled representation of the first token. Shape [batch_size, hidden_size].

        """
        x = self.ffwd(x[:, 0])
        x = self.actv(x)
        return x


class BertBlocksGLUPredictionHead(nn.Module):
    """Prediction head for BertBlocks model with gated activation.

    This class implements a prediction head that uses a gated linear unit (GLU)
    architecture. It projects the hidden states to an expanded dimension, applies
    gating with an activation function, and includes optional pre-/post-normalization.
    """

    def __init__(self, config: "BertBlocksConfig"):
        """Initialize the prediction head.

        Args:
            config (BertBlocksConfig): Configuration object containing:
                - hidden_size: Dimensionality of hidden layers
                - actv_fn: Activation function used in feed-forward networks
                - norm_kind: When to apply normalization ("pre", "post", "both", "none")

        """
        super().__init__()
        self.ffwd = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False)
        self.actv = get_actv_fn(config)
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the prediction head.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after gated projection and normalization.
            Shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x, gate = self.ffwd(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.post_norm(x)
        return x


class BertBlocksMLPPredictionHead(nn.Module):
    """MLP Prediction head for BertBlocks model.

    This class implements a traditional MLP prediction head. It projects the hidden states
    to an expanded dimension, and then projects it back down to the original dimension.
    Includes optional pre-/post-normalization.
    """

    def __init__(self, config: "BertBlocksConfig"):
        """Initialize the prediction head.

        Args:
            config (BertBlocksConfig): Configuration object containing:
                - hidden_size: Dimensionality of hidden layers
                - intermediate_size: Dimensionality of feed-forward layers
                - actv_fn: Activation function used in feed-forward networks
                - mlp_in_bias: Whether to include bias in input projection of MLP layers
                - mlp_out_bias: Whether to include bias in output projection of MLP layers
                - norm_kind: When to apply normalization ("pre", "post", "both", "none")

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
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after projection and normalization.
            Shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x = self.Uprj(x)
        x = self.actv(x)
        x = self.Dprj(x)
        x = self.post_norm(x)
        return x


def get_prediction_head(config: "BertBlocksConfig") -> nn.Module:
    """Get the prediction head layer specified in the configuration.

    This factory function returns the appropriate prediction head architecture
    based on the configuration. Supports both standard MLP and GLU variants.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.

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
        return BertBlocksMLPPredictionHead(config)
    elif mlp_type == "glu":
        return BertBlocksGLUPredictionHead(config)
    else:
        supported_types = ["mlp", "glu"]
        raise ValueError(f"Unknown MLP type '{mlp_type}'. Supported types: {', '.join(supported_types)}")
