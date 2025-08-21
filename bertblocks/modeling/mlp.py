from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from bertblocks.modeling.config import BertBlocksConfig

from torch import nn

from bertblocks.modeling.activations import get_actv_fn


class GLU(nn.Module):
    """Gated Linear Unit (GLU) implementation for BertBlocks.

    This class implements a GLU-style MLP layer that uses gating to control information flow.

    Attributes:
        uprj (nn.Linear): up projection layer, from hidden size to 2 * intermediate size.
        actv (nn.Module): Activation function.
        dprj (nn.Linear): down projection layer, from intermediate size to hidden size.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `hidden_size`: Dimensionality of hidden layers (input/output dimension)
                - `intermediate_size`: Dimensionality of feed-forward layers
                - `mlp_in_bias`: Whether to include bias in the input projection layer
                - `mlp_out_bias`: Whether to include bias in the output projection layer
                - `actv_fn`: Activation function used in feed-forward networks

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.uprj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_in_bias)
        self.actv = get_actv_fn(config)
        self.dprj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the GLU layer.

        Implements the gated linear unit computation: value * activation(gate)
        where both value and gate are linear projections of the input.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after gated projection, down-projection, and dropout,
                shape [batch_size, sequence_length, hidden_size].

        """
        x, gate = self.uprj(x).chunk(2, axis=-1)
        x = gate * self.actv(x)
        x = self.dprj(x)
        return x


class MLP(nn.Module):
    """Standard Multi-Layer Perceptron for BertBlocks.

    This class implements a standard two-layer MLP (feedforward network).

    Attributes:
        uprj (nn.Linear): up projection layer, from hidden size to intermediate size.
        actv (nn.Module): Activation function.
        dprj (nn.Linear): down projection layer, from intermediate size to hidden size.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `hidden_size`: Dimensionality of hidden layers (input/output dimension)
                - `intermediate_size`: Dimensionality of feed-forward layers
                - `mlp_in_bias`: Whether to include bias in the input projection layer
                - `mlp_out_bias`: Whether to include bias in the output projection layer
                - `actv_fn`: Activation function used in feed-forward networks

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.uprj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.actv = get_actv_fn(config)
        self.dprj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the MLP layer.

        Applies standard feedforward transformation: activation(W1*x + b1)*W2 + b2
        where biases are optional based on configuration.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after two linear projections, activation, and dropout,
                shape [batch_size, sequence_length, hidden_size].

        """
        x = self.uprj(x)
        x = self.actv(x)
        x = self.dprj(x)
        return x


def get_mlp(config: "BertBlocksConfig") -> "nn.Module":
    """Get the MLP layer specified in the configuration.

    This factory function returns the appropriate MLP architecture based on
    the configuration. Supports both standard MLP and GLU variants.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.

    Returns:
        An MLP module (nn.Module) that can transform hidden states.

    Raises:
        ValueError: If the specified MLP type is not supported.

    Supported MLP types:

        - `mlp`: Standard two-layer feedforward network
        - `glu`: Gated Linear Unit with learned gating mechanism

    """
    mlp_type = getattr(config, "mlp_type", "mlp")  # Default to mlp for backward compatibility

    if mlp_type == "mlp":
        return MLP(config)
    elif mlp_type == "glu":
        return GLU(config)
    else:
        supported_types = ["mlp", "glu"]
        raise ValueError(f"Unknown MLP type '{mlp_type}'. Supported types: {', '.join(supported_types)}")


__all__ = ["get_mlp", "GLU", "MLP"]
