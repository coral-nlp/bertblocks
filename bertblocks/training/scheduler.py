import math
import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR, LRScheduler

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, ExponentialLR, LambdaLR, LinearLR


class InverseSqrtScheduler(LambdaLR):
    """A scheduler that applies inverse sqrt scaling.

     Scaling is a function of the steps ("1-sqrt cooldown"), where learning rate is scaled (down)
     `1 - sqrt(current_step/total_steps)`. This scheduler is intended for cooldown phases. It has been reported to be
     superior to linear cooldown and is presented as an alternative to cosine decay.

    Args:
        optimizer: The optimizer to use.
        cooldown_steps: The number of cooldown steps. If this number is exceeded, the scheduler will transition
            to a constant learning rate of 0.0.
        last_epoch: The index of the last epoch. Defaults to -1.

    References:
        - Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
          (https://arxiv.org/abs/2405.18392)
    """

    def __init__(self, optimizer: "torch.optim.Optimizer", cooldown_steps: int, last_epoch: int = -1) -> None:
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

        self.cooldown_steps = cooldown_steps

    def lr_lambda(self, current_step: int) -> float:
        """Adjust the learning rate based on cooldown steps."""
        if current_step == 0:
            return 1.0

        if self.last_epoch > self.cooldown_steps:
            return 0.0

        return 1 - math.sqrt(float(current_step) / self.cooldown_steps)


def get_scheduler(
    optimizer: "Optimizer",
    warmup_kind: Literal["linear", "constant"] = "linear",
    warmup_steps: int = 0,
    warmup_decay: float = 0.0,
    training_kind: Literal["constant", "linear", "cosine", "exponential"] = "constant",
    training_steps: int = -1,
    training_decay: float = 1.0,
    cooldown_kind: Literal["linear", "inverse-sqrt", "exponential"] = "linear",
    cooldown_steps: int = 0,
    cooldown_decay: float = 0.0,
) -> "LRScheduler":
    """Construct a sequential learning rate schedule with three phases warmup, training, and cooldown.

    Args:
        optimizer: The optimizer to schedule the learning rate for.
        warmup_kind: Kind of scheduler for warmup phase. Defaults to 'linear'.
        warmup_steps: Duration in steps for warmup phase. Defaults to 0 (no warmup).
        warmup_decay: Decay value for warmup phase; has different effect depending on `warmup_kind`. Defaults to 0.0.
        training_kind: Kind of scheduler for training phase.
        training_steps: Duration in steps for training phase.
        training_decay: Decay value for training phase; has different effect depending on `cooldown_kind`. Defaults
            to 1.0 (constant learning rate).
        cooldown_kind: Kind of scheduler for cooldown phase. Defaults to 'linear'.
        cooldown_steps: Duration in steps for cooldown phase. Defaults to 0 (no cooldown).
        cooldown_decay: Decay value for warmup phase; has different effect depending on `cooldown_kind`.  Defaults
            to 0.0.

    Returns:
        Learning rate scheduler with sequential phases as defined.
    """
    if cooldown_steps > 0 >= training_steps:
        raise ValueError("If cooldown is specified, an `training_steps` must also be explicitly specified.")

    schedulers = []
    milestones = []
    if warmup_steps > 0:
        schedulers.append(
            get_single_scheduler(optimizer, warmup_kind, warmup_steps, warmup_decay, direction="increase")
        )
        milestones.append(warmup_steps)

    schedulers.append(
        get_single_scheduler(optimizer, training_kind, training_steps, training_decay, direction="decrease")
    )

    if cooldown_steps > 0:
        milestones.append(warmup_steps + training_steps)
        schedulers.append(
            get_single_scheduler(optimizer, cooldown_kind, cooldown_steps, cooldown_decay, direction="decrease")
        )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )


def get_single_scheduler(
    optimizer: "Optimizer",
    kind: Literal["constant", "linear", "inverse-sqrt", "exponential", "cosine"] = "constant",
    num_steps: int = 0,
    decay: float = 0.0,
    direction: Literal["increase", "decrease"] = "increase",
) -> "LRScheduler | None":
    """Return the corresponding instantiated scheduler for configuration provided.

    Args:
        optimizer: Optimizer to schedule the learning rate for.
        kind: Kind of scheduler. One of 'constant', 'linear', 'cosine', 'exponential'. Defaults to 'constant'.
        num_steps: Number of steps to schedule. Defaults to 0.
        decay: Decay value, depending on kind of scheduler. If 'constant', is applied as factor; if 'linear', is
            applied as start_factor; if 'exponential', is applied as gamma; if 'cosine', is applied as eta_min.
        direction: Direction of scheduler. Defaults to 'increase'.

    Returns:
        The specified scheduler.
    """
    match kind:
        case "constant":
            return ConstantLR(optimizer, factor=decay, total_iters=num_steps)
        case "linear":
            return LinearLR(
                optimizer,
                start_factor=decay if direction == "increase" else 1.0,
                end_factor=decay if direction == "decrease" else 1.0,
                total_iters=num_steps,
            )
        case "exponential":
            if direction == "increase":
                warnings.warn("InverseSqrtScheduler is intended for cooldown only, but used for warmup!", stacklevel=2)
            return ExponentialLR(optimizer, gamma=decay)
        case "cosine":
            if direction == "increase":
                warnings.warn("CosineAnnealingLR is intended for cooldown only, but used for warmup!", stacklevel=2)
            return CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=decay)
        case "inverse-sqrt":
            if direction == "increase":
                warnings.warn("InverseSqrtScheduler is intended for cooldown only, but used for warmup!", stacklevel=2)
            return InverseSqrtScheduler(optimizer, num_steps)
        case "_":
            raise ValueError(
                f"Unknown scheduler kind, got {kind}, expected one of 'constant', 'linear', 'exponential', 'cosine'."
            )
