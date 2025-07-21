from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.activations import get_actv_fn
from polybert.modeling.norms import get_norm


class PolyBertPredictionHead(nn.Module):
    """Prediction head for PolyBert model with gated activation.

    This class implements a prediction head that uses a gated linear unit (GLU)
    architecture. It projects the hidden states to an expanded dimension, applies
    gating with an activation function, and includes optional pre-/post-normalization.

    The head uses a two-stage approach:
    1. Project to 2x hidden_size and split into value and gate
    2. Apply gating: value * activation(gate)
    3. Apply normalization based on configuration
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

        Applies gated linear transformation with optional normalization.
        The gating mechanism helps control information flow by learning
        which features to emphasize or suppress.

        Args:
            x: Input tensor. Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Transformed tensor after gated projection and normalization.
            Shape: (batch_size, sequence_length, hidden_size)

        Note:
            The forward pass follows this sequence:
            1. Apply pre-normalization (if configured)
            2. Project to 2x hidden_size and split into value and gate
            3. Apply gating: value * activation(gate)
            4. Apply post-normalization (if configured)

        """
        x = self.pre_norm(x)
        x, gate = self.ffwd(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.post_norm(x)
        return x
