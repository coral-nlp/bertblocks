from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import torch
from torch import nn


class LayerScaler(nn.Module):
    """Scales an input inversely to the layer depth.

    Attributes:
        scaling_factor (torch.Tensor): scaling factor.

    Args:
        layer_id (int): layer position in the encoder stack (0-indexed).

    References:
        - The Curse of Depth in Large Language Models (https://arxiv.org/pdf/2502.05795)
    """

    def __init__(self, layer_id: int):
        super().__init__()
        self.register_buffer("scale", 1.0 / torch.sqrt(torch.tensor(layer_id + 1)))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Apply layer scaling.

        Args:
            x (torch.Tensor): Input tensor to scale.

        Returns:
            torch.Tensor: Scaled tensor.

        """
        return x * self.scale
