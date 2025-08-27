from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

import torch
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, ExponentialLR, LinearLR


def get_scheduler(
    optimizer: "Optimizer",
    warmup_kind: Literal["linear", "constant"] = "linear",
    warmup_steps: int = 0,
    warmup_decay: float = 0.0,
    training_kind: Literal["constant", "linear", "cosine", "exponential"] = "constant",
    training_steps: int = -1,
    training_decay: float = 1.0,
    cooldown_kind: Literal["linear", "exponential"] = "linear",
    cooldown_steps: int = 0,
    cooldown_decay: float = 0.0,
) -> LRScheduler:
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
        schedulers.append(get_single_scheduler(optimizer, warmup_kind, warmup_steps, warmup_decay))
        milestones.append(warmup_steps)

    schedulers.append(get_single_scheduler(optimizer, training_kind, training_steps, training_decay))

    if cooldown_steps > 0:
        schedulers.append(get_single_scheduler(optimizer, cooldown_kind, cooldown_steps, cooldown_decay))
        milestones.append(training_steps)

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )


def get_single_scheduler(
    optimizer: "Optimizer",
    kind: Literal["constant", "linear", "exponential", "cosine"] = "constant",
    num_steps: int = 0,
    decay: float = 0.0,
) -> "LRScheduler | None":
    """Return the corresponding instantiated scheduler for configuration provided.

    Args:
        optimizer: Optimizer to schedule the learning rate for.
        kind: Kind of scheduler. One of 'constant', 'linear', 'cosine', 'exponential'. Defaults to 'constant'.
        num_steps: Number of steps to schedule. Defaults to 0.
        decay: Decay value, depending on kind of scheduler. If 'constant', is applied as factor; if 'linear', is
            applied as start_factor; if 'exponential', is applied as gamma; if 'cosine', is applied as eta_min.

    Returns:
        The specified scheduler.
    """
    match kind:
        case "constant":
            return ConstantLR(optimizer, factor=decay, total_iters=num_steps)
        case "linear":
            return LinearLR(optimizer, start_factor=decay, total_iters=num_steps)
        case "exponential":
            return ExponentialLR(optimizer, gamma=decay)
        case "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=decay)
        case "_":
            raise ValueError(
                f"Unknown scheduler kind, got {kind}, expected one of 'constant', 'linear', 'exponential', 'cosine'."
            )
