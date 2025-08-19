"""Masked Language Modeling (MLM) pretraining implementation for BertBlocks.

This module provides a complete MLM pretraining setup using PyTorch Lightning,
including data loading, model configuration, optimization, and training logic.
It supports streaming datasets, flexible model compilation, and various
optimization strategies.

The implementation includes:
- MaskedLanguageModelingCollator for dynamic masking
- BertBlocksPretrainingDataModule for data loading
- BertBlocksPretrainingModule for training logic
- Support for model compilation and advanced optimization
"""

import os.path
from pathlib import Path
from typing import Any

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import get_parameter_names

from bertblocks.modeling import BertBlocksConfig, BertBlocksForMaskedLM
from bertblocks.pretraining.objectives import MaskedLanguageModelingCollator
from bertblocks.pretraining.optimizer import get_optimizer


class BertBlocksPretrainingDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for BertBlocks MLM pretraining.

    This DataModule handles all aspects of data loading for pretraining,
    including dataset preparation, tokenization, and batch creation.
    Currently configured to use the TinyStories dataset but can be easily
    adapted for other datasets.

    The module supports streaming datasets for large-scale pretraining
    and includes configurable batch sizes and data loading parameters.
    """

    def __init__(
        self,
        pretrained_tokenizer_name_or_path: str,
        max_sequence_length: "int | None" = 256,
        dataset_name_or_path: "str | list[str] | None" = None,
        mlm_probability: "float | None" = 0.3,
        binarize_labels: "bool | None" = False,
        train_batch_size: "int | None" = 32,
        val_batch_size: "int | None" = 32,
        pretokenized: "bool | None" = False,
        num_workers: "int | None" = 0,
    ) -> None:
        """Initialize the pretraining data module.

        Args:
            pretrained_tokenizer_name_or_path: Path or name of HuggingFace tokenizer
                to use for text processing.
            max_sequence_length: Maximum sequence length for tokenization.
                Longer sequences will be truncated. Defaults to 256.
            dataset_name_or_path: Dataset name or path (currently unused,
                TinyStories is hardcoded). Defaults to None.
            mlm_probability: Probability of masking tokens. Defaults to 0.3.
            binarize_labels: Whether to binarize labels (currently unused).
                Defaults to False.
            train_batch_size: Batch size for training. Defaults to 32.
            val_batch_size: Batch size for validation. Defaults to 32.
            pretokenized: Whether input is pre-tokenized. Defaults to False.
            num_workers: Number of workers for data loading. Defaults to 0.

        """
        super().__init__()
        self.save_hyperparameters()
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
        self.mlm_collation_fn = MaskedLanguageModelingCollator(
            tokenizer=tokenizer,
            max_sequence_length=self.hparams.max_sequence_length,
            text_column="text",
            pretokenized=self.hparams.pretokenized,
        )
        # Dummy object for dataset
        self.dataset = None

    def prepare_data(self) -> None:
        """Prepare the dataset for training. Called once per node."""
        if os.path.isdir(self.hparams.dataset_name_or_path):
            # If local path, load from disk (future: not only support JSON?)
            self.dataset = load_dataset(
                "json", data_dir=self.hparams.dataset_name_or_path, split="train", streaming=True
            )
        else:
            # If not local path, try HF
            self.dataset = load_dataset(self.hparams.dataset_name_or_path, split="train", streaming=True)

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            DataLoader: PyTorch DataLoader configured for MLM training
                with the specified batch size, collation function, and
                data loading parameters.

        """
        return DataLoader(
            self.dataset,
            collate_fn=self.mlm_collation_fn,
            shuffle=False,  # It's streamed so we can't shuffle it
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


class BertBlocksPretrainingModule(L.LightningModule):
    """PyTorch Lightning module for BertBlocks MLM pretraining.

    This module encapsulates the complete training logic for BertBlocks pretraining,
    including model initialization, optimization setup, and training step implementation.
    It supports advanced features like model compilation, sophisticated learning rate
    scheduling, and automatic checkpoint saving.

    The module automatically configures the BertBlocks model based on the provided
    hyperparameters and handles all aspects of the training loop.
    """

    def __init__(
        self,
        learning_rate: float | None = 1e-7,
        weight_decay: float | None = 1e-6,
        warmup_steps: int | None = 1_000,
        warmup_decay: float | None = 0.1,
        learning_rate_decay: float | None = 0.99999,
        compile_model: bool | None = True,
        pretrained_tokenizer_name_or_path: str | None = None,
        optimizer_class: str = "adamw",
        optimizer_kwargs: dict | None = None,
        model_config_kwargs: "dict[str, Any] | None" = None,
    ):
        """Initialize the BertBlocks pretraining module.

        Args:
            learning_rate: Peak learning rate for optimization. Defaults to 1e-7.
            weight_decay: Weight decay coefficient for AdamW. Defaults to 1e-6.
            warmup_steps: Number of warmup steps for learning rate schedule.
                Defaults to 1,000.
            warmup_decay: Factor to scale learning rate during warmup.
                Defaults to 0.1 (starts at 10% of peak LR).
            learning_rate_decay: Exponential decay factor after warmup.
                Defaults to 0.99999 (very gradual decay).
            compile_model: Whether to compile the model with torch.compile.
                Defaults to True for better performance.
            pretrained_tokenizer_name_or_path: str
                Tokenizer name; if provided, will overwrite the model vocab size using the given tokenizer.
            optimizer_class: Optimizer class name. Defaults to "adamw".
            optimizer_kwargs: Optional arguments to pass to torch.optim.optimizer.
            model_config_kwargs: dict[str, Any] or None
                Optional dictionary of model configuration options passed to BertBlocksConfig for instantiation.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        if model_config_kwargs is None:
            model_config_kwargs = {}
        self.model_config = BertBlocksConfig(**model_config_kwargs)
        # Patch model config with tokenizer vocab size if given
        if self.hparams.pretrained_tokenizer_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
            self.model_config.vocab_size = tokenizer.vocab_size
            del tokenizer
        self.model = BertBlocksForMaskedLM(self.model_config)
        if self.hparams.compile_model:
            torch.set_float32_matmul_precision("medium")
            self.model = torch.compile(self.model, dynamic=True)

    def configure_optimizers(self) -> tuple[list["torch.optim.Optimizer"], list[dict[str, Any]]]:
        """Configure optimizers and learning rate schedulers.

        Sets up AdamW optimizer with weight decay only applied to non-bias parameters
        (excluding RMSNorm parameters). Uses a sequential learning rate schedule
        with linear warmup followed by exponential decay.

        Returns:
            tuple: Contains optimizer list and scheduler configuration list.
                The scheduler is configured to update every step during training.

        """
        decay_parameters = get_parameter_names(self.model, [torch.nn.RMSNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer = get_optimizer(
            self.hparams.optimizer_class, optimizer_grouped_parameters, self.hparams.optimizer_kwargs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=self.hparams.warmup_decay, total_iters=self.hparams.warmup_steps
                ),
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.learning_rate_decay),
            ],
            milestones=[self.hparams.warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch: Batch dictionary containing 'input_ids', 'attention_mask',
                and 'labels' tensors from the MLM collator.
            batch_idx: Index of the current batch (unused but required by Lightning).

        Returns:
            torch.Tensor: MLM loss for backpropagation.

        """
        if self.hparams.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("loss/train", output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self.model(batch, labels=batch)
        self.log("loss/valid", output.loss, prog_bar=True)
        return output.loss

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict({f'gradnorm/{k}': v for k, v in norms.items()})

    def on_save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Save model checkpoint in HuggingFace format.

        This method is called whenever Lightning saves a checkpoint and
        additionally saves the model in HuggingFace format for easy loading
        and deployment. Only saves on the main process in distributed training.

        Args:
            *args: Variable arguments (unused).
            **kwargs: Keyword arguments (unused).

        """
        if self.trainer is not None and self.trainer.log_dir is not None:
            if self.trainer.global_rank != 0:
                return
            log_dir = Path(self.trainer.log_dir)
            save_path = log_dir / "huggingface_checkpoint"
            self.model.save_pretrained(save_path, safe_serialization=False)
