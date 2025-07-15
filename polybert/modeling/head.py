from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.activations import get_actv_fn
from polybert.modeling.norms import get_norm


class PolyBertPredictionHead(nn.Module):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__()
        self.ffwd = nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=False)
        self.actv = get_actv_fn(config)
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.pre_norm(x)
        x, gate = self.ffwd(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.post_norm(x)
        return x
