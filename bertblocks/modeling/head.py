from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from bertblocks.modeling.config import BertBlocksConfig

from torch import nn

from bertblocks.modeling.activations import get_actv_fn
from bertblocks.modeling.norms import get_norm


class Pooler(nn.Module):
    """Pooling layer.

    Applies a linear layer and activation function to the first token of the last hidden state.

    Attributes:
        ffwd: Feed-forward layer from hidden size to hidden size.
        actv: Activation function.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `hidden_size`: Dimensionality of hidden layers
                - `actv_fn`: Activation function used in feed-forward networks

    """

    def __init__(self, config: "BertBlocksConfig") -> None:
        super().__init__()
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size)
        self.actv = get_actv_fn(config)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the pooling layer.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): Padded input hidden states.

        Returns:
            torch.Tensor: Pooled representation of the first token. Shape [batch_size, hidden_size].

        """
        x = self.ffwd(x[:, 0])
        x = self.actv(x)
        return x


class GLUPredictionHead(nn.Module):
    """Prediction head with gated activation.

    Attributes:
        pre_norm (nn.Module): Pre-norm function. Falls back to `nn.Identity` if not configured.
        ffwd (nn.Linear): Feed-forward projection layer, from hidden size to 2 * hidden size.
        actv (nn.Module): Activation function.
        post_norm (nn.Module): Post-norm function. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `hidden_size`: Dimensionality of hidden layers
            - `actv_fn`: Activation function used in feed-forward networks
            - `norm_kind`: When to apply normalization ("pre", "post", "both", "none")

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.ffwd = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False)
        self.actv = get_actv_fn(config)
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the prediction head.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Padded input hidden state.

        Returns:
            torch.Tensor: Transformed hidden state, shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x, gate = self.ffwd(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.post_norm(x)
        return x


class MLPPredictionHead(nn.Module):
    """MLP Prediction head.

    Attributes:
        pre_norm (nn.Module): Pre-norm function. Falls back to `nn.Identity` if not configured.
        uprj (nn.Linear): MLP up projection layer, from hidden size to intermediate size.
        actv (nn.Module): Activation function.
        dprj (nn.Linear): MLP down projection layer, from intermediate size to hidden size.
        post_norm (nn.Module): Post-norm function. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `hidden_size`: Dimensionality of hidden layers
                - `intermediate_size`: Dimensionality of feed-forward layers
                - `actv_fn`: Activation function used in feed-forward networks
                - `mlp_in_bias`: Whether to include bias in input projection of MLP layers
                - `mlp_out_bias`: Whether to include bias in output projection of MLP layers
                - `norm_kind`: When to apply normalization ("pre", "post", "both", "none")

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.uprj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.actv = get_actv_fn(config)
        self.dprj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the prediction head.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Padded input hidden state.

        Returns:
            torch.Tensor: Transformed hidden state, shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x = self.uprj(x)
        x = self.actv(x)
        x = self.dprj(x)
        x = self.post_norm(x)
        return x


def get_prediction_head(config: "BertBlocksConfig") -> nn.Module:
    """Get the prediction head layer specified in the configuration.

    This factory function returns the appropriate prediction head architecture
    based on the configuration. Supports both standard MLP and GLU variants.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.

    Returns:
        An prediction head module that can transform hidden states.

    Raises:
        ValueError: If the specified prediction head type is not supported.

    Supported prediction head types:

        - `mlp`: Standard two-layer feedforward network
        - `glu`: Gated Linear Unit

    """
    mlp_type = getattr(config, "mlp_type", "mlp")  # Default to mlp for backward compatibility

    if mlp_type == "mlp":
        return MLPPredictionHead(config)
    elif mlp_type == "glu":
        return GLUPredictionHead(config)
    else:
        supported_types = ["mlp", "glu"]
        raise ValueError(f"Unknown MLP type '{mlp_type}'. Supported types: {', '.join(supported_types)}")


__all__ = ["get_prediction_head", "MLPPredictionHead", "GLUPredictionHead", "Pooler"]
