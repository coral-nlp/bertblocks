from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.activations import get_actv_fn


class PolyBertGLU(nn.Module):
    def __init__(self, config: "PolyBertConfig"):
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
        x, gate = self.Uprj(x).chunk(2, axis=-1)
        x = x * self.actv(gate)
        x = self.Dprj(x)
        x = self.drop(x)
        return x


class PolyBertMLP(nn.Module):
    def __init__(self, config: "PolyBertConfig"):
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
        x = self.Uprj(x)
        x = self.actv(x)
        x = self.Dprj(x)
        x = self.drop(x)
        return x


def get_mlp(config: "PolyBertConfig") -> nn.Module:
    match config.mlp_type:
        case "mlp":
            return PolyBertMLP(config)
        case "glu":
            return PolyBertGLU(config)
        case _:
            raise ValueError(f"Unknown mlp type: {config.mlp_type}")
