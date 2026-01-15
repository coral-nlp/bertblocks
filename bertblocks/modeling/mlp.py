from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from bertblocks.config import BertBlocksConfig

from torch import nn

from bertblocks.modeling.activations import get_actv_fn


class Linear(nn.Module):
    """Linear layer wrapper implementation for BertBlocks.

    Attributes:
        ffwd (nn.Linear): linear feed-forward layer.
        actv (nn.Module): activation function.

    Args:
        hidden_size (int): Dimensionality of hidden layers (input/output dimension).
        actv_fn (str): Activation function.
        bias (bool): Whether to include bias in the layer. Defaults to True.
    """

    def __init__(self, hidden_size: int, actv_fn: str, bias: bool = True):
        super().__init__()
        self.ffwd = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.actv = get_actv_fn(actv_fn)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.actv(self.ffwd(x))


class GLU(nn.Module):
    """Gated Linear Unit (GLU) implementation for BertBlocks.

    This class implements a GLU-style MLP layer that uses gating to control information flow.

    Attributes:
        uprj (nn.Linear): up projection layer, from hidden size to 2 * intermediate size.
        actv (nn.Module): Activation function.
        dprj (nn.Linear): down projection layer, from intermediate size to hidden size.

    Args:
        hidden_size (int): Dimensionality of hidden layers (input/output dimension).
        intermediate_size (int): Dimensionality of feed-forward layers.
        actv_fn (str): Activation function used in feed-forward networks.
        in_bias (bool): Whether to include bias in the input projection layer. Defaults to True.
        out_bias (bool): Whether to include bias in the output projection layer. Defaults to True.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, actv_fn: str, in_bias: bool = True, out_bias: bool = True
    ):
        super().__init__()
        self.uprj = nn.Linear(hidden_size, intermediate_size * 2, bias=in_bias)
        self.actv = get_actv_fn(actv_fn)
        self.dprj = nn.Linear(intermediate_size, hidden_size, bias=out_bias)

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
        hidden_size (int): Dimensionality of hidden layers (input/output dimension).
        intermediate_size (int): Dimensionality of feed-forward layers.
        actv_fn (str): Activation function used in feed-forward networks.
        in_bias (bool): Whether to include bias in the input projection layer. Defaults to True.
        out_bias (bool): Whether to include bias in the output projection layer. Defaults to True.

    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, actv_fn: str, in_bias: bool = True, out_bias: bool = True
    ):
        super().__init__()
        self.uprj = nn.Linear(hidden_size, intermediate_size, bias=in_bias)
        self.actv = get_actv_fn(actv_fn)
        self.dprj = nn.Linear(intermediate_size, hidden_size, bias=out_bias)

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
        - `linear`: Standard single feed-forward layer.
        - `mlp`: Standard two-layer feedforward network
        - `glu`: Gated Linear Unit with learned gating mechanism

    """
    match config.mlp_type:
        case "linear":
            return Linear(config.hidden_size, config.actv_fn, bias=config.mlp_in_bias)
        case "mlp":
            return MLP(
                config.hidden_size, config.intermediate_size, config.actv_fn, config.mlp_in_bias, config.mlp_out_bias
            )
        case "glu":
            return GLU(
                config.hidden_size, config.intermediate_size, config.actv_fn, config.mlp_in_bias, config.mlp_out_bias
            )
        case _:
            supported_types = ["mlp", "glu"]
            raise ValueError(f"Unknown MLP type '{config.mlp_type}'. Supported types: {', '.join(supported_types)}")


__all__ = ["GLU", "MLP", "get_mlp"]
