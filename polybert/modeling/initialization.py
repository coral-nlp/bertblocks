"""Model weight initialization utilities for PolyBert.

This module provides comprehensive weight initialization strategies for transformer
models, including support for various initialization schemes (truncated normal,
Kaiming, Xavier) and layer-specific initialization patterns.

The module implements a mixin class and initialization functions that can be
applied to PolyBert models to ensure proper weight initialization for training
stability and performance.
"""

import functools
import math
from collections.abc import Callable
from typing import Literal

from torch import nn

from polybert.modeling.config import PolyBertConfig


class InitMixin:
    """Mixin class for initialization strategies."""

    def __init__(self, config: PolyBertConfig):
        self.intializer_kind = config.initializer_kind
        self.initializer_cutoff_factor = config.initializer_cutoff_factor
        self.initializer_range = config.initializer_range

        self.std = {
            "in": self.initializer_range,
            "out": self.initializer_range / math.sqrt(2.0 * config.num_hidden_layers),
            "embedding": self.initializer_range,
            "final_out": config.hidden_size**-0.5,
        }

    def _init_module_weights(self, module: nn.Module, std_kind: Literal["in", "out", "embedding", "final_out"]) -> None:
        """Apply to a module a specific initialization function based on given distribution parameters.

        Args:
            module: The module to initialize.
            std_kind: Standard deviation for weight initialization, differs by layer.

        Raises:
            ValueError: If unsupported initialization options are used.

        """
        # Different standard deviations depending on layer type
        if std_kind not in ["in", "out", "embedding", "final_out"]:
            raise ValueError(
                f"Unknown standard deviation type {std_kind}, supported types: 'in', 'out', 'embedding', 'final_out'"
            )

        std = self.std[std_kind]

        def _get_init_fn() -> Callable[["nn.Module"], None]:
            match self.intializer_kind:
                case "trunc_normal":
                    return functools.partial(
                        nn.init.trunc_normal_,
                        mean=0.0,
                        std=std,
                        a=-self.initializer_cutoff_factor * std,
                        b=self.initializer_cutoff_factor * std,
                    )
                case "kaiming_normal":
                    return functools.partial(
                        nn.init.kaiming_normal_,
                    )
                case "kaiming_uniform":
                    return functools.partial(
                        nn.init.kaiming_uniform_,
                    )
                case "xavier_normal":
                    return functools.partial(
                        nn.init.xavier_normal_,
                    )
                case "xavier_uniform":
                    return functools.partial(
                        nn.init.xavier_uniform_,
                    )
                case _:
                    raise ValueError(
                        f"Unknown initialization function {self.intializer_kind}, supported functions: "
                        f"'trunc_normal', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform'"
                    )

        # Apply initialization function to module
        module.apply(_get_init_fn())

        # Initialize bias terms to zero for linear layers
        if isinstance(module, nn.Linear):  # noqa: SIM102
            if module.bias is not None:
                nn.init.zeros_(module.bias)
