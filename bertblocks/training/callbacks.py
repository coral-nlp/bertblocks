"""Custom Lightning callbacks for BertBlocks training."""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import ModelCheckpoint

if TYPE_CHECKING:
    import lightning.pytorch as pl

__all__ = ["HuggingFaceModelCheckpoint"]


class HuggingFaceModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that also exports a HuggingFace checkpoint next to every ``.ckpt``.

    Each HuggingFace export shares the lifecycle of the Lightning checkpoint it accompanies: it is written
    whenever that ``.ckpt`` is written and deleted whenever that ``.ckpt`` is pruned. The retention strategy
    (``save_top_k``, ``save_last``, ``every_n_train_steps``, ...) therefore applies identically to both — e.g.
    ``save_top_k: 3`` keeps exactly three HuggingFace checkpoints, matching the three retained ``.ckpt`` files,
    and every other export is pruned.

    Binding the export to the checkpoint lifecycle (rather than the ``LightningModule.on_save_checkpoint``
    hook) is what makes pruning work: only the callback knows which ``.ckpt`` files it removes.
    """

    #: Subdirectory (next to the ``.ckpt`` files) that holds the per-checkpoint HuggingFace exports.
    HF_SUBDIR = "huggingface"

    def _hf_dir(self, filepath: str) -> Path:
        """Map a checkpoint ``.ckpt`` filepath to its sibling HuggingFace export directory.

        ``.../checkpoints/step=10000.ckpt`` -> ``.../checkpoints/huggingface/step=10000``.
        """
        path = Path(filepath)
        return path.parent / self.HF_SUBDIR / path.stem

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Write the Lightning checkpoint, then export a HuggingFace checkpoint alongside it (rank 0)."""
        super()._save_checkpoint(trainer, filepath)
        if not trainer.is_global_zero:
            return
        model = getattr(trainer.lightning_module, "model", None)
        if model is None or not hasattr(model, "save_pretrained"):
            return
        # safe_serialization=False: the MLM/BOW heads are weight-tied to the input embeddings and
        # safetensors rejects the resulting shared storage.
        model.save_pretrained(self._hf_dir(filepath), safe_serialization=False)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Remove the Lightning checkpoint, then delete its sibling HuggingFace export (rank 0)."""
        super()._remove_checkpoint(trainer, filepath)
        if not trainer.is_global_zero:
            return
        hf_dir = self._hf_dir(filepath)
        if hf_dir.is_dir():
            shutil.rmtree(hf_dir, ignore_errors=True)
