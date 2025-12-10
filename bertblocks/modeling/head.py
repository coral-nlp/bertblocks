import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from bertblocks.config import BertBlocksConfig

from torch import nn

from bertblocks.modeling.mlp import get_mlp
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
        config.mlp_type = "linear"  # Overwrite to get desired behaviour from getter functions
        self.ffwd = get_mlp(config)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the pooling layer.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): Padded input hidden states.

        Returns:
            torch.Tensor: Pooled representation of the first token. Shape [batch_size, hidden_size].

        """
        x = self.ffwd(x[:, 0])
        return x


class ProjectionPredictionHead(nn.Module):
    """Prediction head with linear projection.

    Attributes:
        pre_norm (nn.Module): Pre-norm function. Falls back to `nn.Identity` if not configured.
        ffwd (nn.Linear): Feed-forward projection layer.
        post_norm (nn.Module): Post-norm function. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `norm_kind`: When to apply normalization ("pre", "post", "both", "none")

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        config_copy = copy.deepcopy(config)
        config_copy.mlp_type = "linear"  # Overwrite to get desired behaviour from getter functions
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.ffwd = get_mlp(config_copy)
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the prediction head.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Padded input hidden state.

        Returns:
            torch.Tensor: Transformed hidden state, shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x = self.ffwd(x)
        x = self.post_norm(x)
        return x


class GLUPredictionHead(nn.Module):
    """Prediction head with gated activation.

    Attributes:
        pre_norm (nn.Module): Pre-norm function. Falls back to `nn.Identity` if not configured.
        ffwd (nn.Module): Feed-forward projection layer.
        post_norm (nn.Module): Post-norm function. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `norm_kind`: When to apply normalization ("pre", "post", "both", "none")

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        config_copy = copy.deepcopy(config)
        config_copy.mlp_type = "glu"  # Overwrite to get desired behaviour from getter functions
        # Overwrite to get desired behaviour from getter functions
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.ffwd = get_mlp(config_copy)
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the prediction head.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Padded input hidden state.

        Returns:
            torch.Tensor: Transformed hidden state, shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x = self.ffwd(x)
        x = self.post_norm(x)
        return x


class MLPPredictionHead(nn.Module):
    """MLP Prediction head.

    Attributes:
        pre_norm (nn.Module): Pre-norm function. Falls back to `nn.Identity` if not configured.
        ffwd (nn.Module): Feed-forward projection layer.
        post_norm (nn.Module): Post-norm function. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `norm_kind`: When to apply normalization ("pre", "post", "both", "none")

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        config_copy = copy.deepcopy(config)
        config_copy.mlp_type = "mlp"  # Overwrite to get desired behaviour from getter functions
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.ffwd = get_mlp(config_copy)
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass of the prediction head.

        Args:
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Padded input hidden state.

        Returns:
            torch.Tensor: Transformed hidden state, shape [batch_size, sequence_length, hidden_size].

        """
        x = self.pre_norm(x)
        x = self.ffwd(x)
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

        - `proj`: Projection prediction head.
        - `mlp`: Standard two-layer feedforward network
        - `glu`: Gated Linear Unit

    """
    match config.head_type:
        case "proj":
            return ProjectionPredictionHead(config)
        case "mlp":
            return MLPPredictionHead(config)
        case "glu":
            return GLUPredictionHead(config)
        case _:
            supported_types = ["proj", "mlp", "glu"]
            raise ValueError(f"Unknown head type '{config.head_type}'. Supported types: {', '.join(supported_types)}")


__all__ = ["GLUPredictionHead", "MLPPredictionHead", "Pooler", "ProjectionPredictionHead", "get_prediction_head"]
