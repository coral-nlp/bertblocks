from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.activations import get_actv_fn


class PolyBertGLU(nn.Module):
    """Gated Linear Unit (GLU) implementation for PolyBert.

    This class implements a GLU-style MLP layer that uses gating to control information flow.

    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the GLU layer.

        Args:
            config: PolyBert configuration object containing:
                - hidden_size: Input/output dimension size
                - intermediate_size: Intermediate layer dimension size
                - actv_fn: Activation function type for gating
                - hidden_dropout_prob: Dropout probability (0 means no dropout)

        """
        super().__init__()
        # Upwards projection to intermediate size & gate
        self.Uprj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        # Activation function for gate
        self.actv = get_actv_fn(config)
        # Downwards projection to hidden size
        self.Dprj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Optional dropout
        self.drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the GLU layer.

        Implements the gated linear unit computation: value * activation(gate)
        where both value and gate are linear projections of the input.

        Args:
            x: Input tensor. Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Transformed tensor after gated projection, down-projection, and dropout.
            Shape: (batch_size, sequence_length, hidden_size)

        """
        x, gate = self.Uprj(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.Dprj(x)
        x = self.drop(x)
        return x


class PolyBertMLP(nn.Module):
    """Standard Multi-Layer Perceptron for PolyBert.

    This class implements a standard two-layer MLP (feedforward network).

    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the MLP layer.

        Args:
            config: PolyBert configuration object containing:
                - hidden_size: Input/output dimension size
                - intermediate_size: Intermediate layer dimension size
                - mlp_in_bias: Whether to use bias in the input projection
                - mlp_out_bias: Whether to use bias in the output projection
                - actv_fn: Activation function type
                - hidden_dropout_prob: Dropout probability (0 means no dropout)

        """
        super().__init__()
        # Upwards projection to intermediate size
        self.Uprj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        # Activation function
        self.actv = get_actv_fn(config)
        # Downwards projection to hidden size
        self.Dprj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)
        # Optional dropout
        self.drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0.0 else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the MLP layer.

        Applies standard feedforward transformation: activation(W1*x + b1)*W2 + b2
        where biases are optional based on configuration.

        Args:
            x: Input tensor. Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Transformed tensor after two linear projections, activation, and dropout.
            Shape: (batch_size, sequence_length, hidden_size)

        """
        x = self.Uprj(x)
        x = self.actv(x)
        x = self.Dprj(x)
        x = self.drop(x)
        return x


def get_mlp(config: "PolyBertConfig") -> "nn.Module":
    """Get the MLP layer specified in the configuration.

    This factory function returns the appropriate MLP architecture based on
    the configuration. Supports both standard MLP and GLU variants.

    Args:
        config: PolyBert configuration object containing the MLP type
                specification in config.mlp_type

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
        return PolyBertMLP(config)
    elif mlp_type == "glu":
        return PolyBertGLU(config)
    else:
        supported_types = ["mlp", "glu"]
        raise ValueError(f"Unknown MLP type '{mlp_type}'. " f"Supported types: {', '.join(supported_types)}")
