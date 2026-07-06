"""Custom Lightning loggers for BertBlocks training."""

from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.loggers.wandb import WandbLogger as _WandbLogger

__all__ = ["WandbLogger"]


# NOTE remove once https://github.com/Lightning-AI/pytorch-lightning/pull/21462 is merged.
# The stock ``WandbLogger.save_dir`` returns the constructor ``save_dir`` argument rather than the
# directory wandb actually writes the run into (``experiment.dir``, i.e. ``<save_dir>/wandb/run-*/files``).
# Lightning derives ``trainer.log_dir`` — and hence the ``ModelCheckpoint`` directory and the HuggingFace
# export in ``BertBlocksPretrainingModule.on_save_checkpoint`` — from ``save_dir``, so with the stock logger
# checkpoints land next to the ephemeral default instead of inside the persistent wandb run directory. On
# Modal that means they never reach the mounted volume. Returning ``experiment.dir`` fixes the resolution.
class WandbLogger(_WandbLogger):
    """WandbLogger whose ``save_dir`` is the actual wandb run directory (``experiment.dir``)."""

    @property
    def save_dir(self) -> str | None:
        """Return the wandb run directory, or ``None`` before an experiment exists.

        Returns:
            str | None: ``experiment.dir`` once the wandb run is initialized, else ``None`` (e.g. when the
                experiment is a ``DummyExperiment`` on non-zero ranks).
        """
        if isinstance(self.experiment, DummyExperiment):
            return None
        return self.experiment.dir
