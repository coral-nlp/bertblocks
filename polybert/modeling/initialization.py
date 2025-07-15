import math
from typing import Literal

from torch import nn


class InitMixin:
    def __init__(self, config: "PolyBertConfig"):
        super().__init__()
        self.config = config

    def _init_weights(self, reset_params: bool = False):
        if hasattr(self, "attn"):
            self.attn._init_weights(reset_params)
        if hasattr(self, "mlp"):
            self.mlp._init_weights(reset_params)


def init_weights(config: "PolyBertConfig"):
    def __inner__(module: nn.Module) -> None:
        def init_weight(
            module: nn.Module,
            std: float,
            init_fn: Literal["trunc_normal"] = "trunc_normal",
        ):
            match init_fn:
                case "trunc_normal":
                    nn.init.trunc_normal_(
                        module.weight,
                        mean=0.0,
                        std=std,
                        a=-config.initializer_cutoff_factor * std,
                        b=config.initializer_cutoff_factor * std,
                    )
                case "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight)
                case "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight)
                case "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                case "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                case _:
                    raise ValueError(f"Unknown initialization function {init_fn}")

            if isinstance(module, nn.Linear):  # noqa: SIM102
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": config.initializer_range,
            "out": config.initializer_range / math.sqrt(2.0 * config.num_hidden_layers),
            "embedding": config.initializer_range,
            "final_out": config.hidden_size**-0.5,
        }

        if isinstance(module, PolyBertEmbeddings):
            init_weight(module.embd, stds["embedding"])
        elif isinstance(module, PolyBertGLU):
            init_weight(module.ffwd, stds["in"])
            init_weight(module.proj, stds["out"])
        elif isinstance(module, PolyBertAttention):
            init_weight(module.proj, stds["in"])
            init_weight(module.ffwd, stds["out"])
        elif isinstance(module, PolyBertPredictionHead):
            init_weight(module.ffwd, stds["out"])
        elif hasattr(module, "decoder") and module.__class__.__name__ == "PolyBertForMaskedLM":
            init_weight(module.decoder, stds["out"])
        elif hasattr(module, "classifier") and module.__class__.__name__ in (
            "PolyBertForSequenceClassification",
            "PolyBertForTokenClassification",
            "PolyBertForQuestionAnswering",
        ):
            init_weight(module.classifier, stds["final_out"])

    return __inner__
