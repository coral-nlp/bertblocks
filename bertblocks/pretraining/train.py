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

import functools
import os.path
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import get_parameter_names

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.norms import DeepNorm, DynamicTanhNorm, GroupNorm, LayerNorm, RMSNorm
from bertblocks.pretraining.objectives import get_collator, get_model
from bertblocks.pretraining.optimizer import get_optimizer
from bertblocks.pretraining.scheduler import get_scheduler
from bertblocks.pretraining.utils import chunk_examples


class BertBlocksPretrainingDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for BertBlocks MLM pretraining.

    This DataModule handles all aspects of data loading for pretraining,
    including dataset preparation, tokenization, and batch creation.
    Currently configured to use the TinyStories dataset but can be easily
    adapted for other datasets.

    The module supports streaming datasets for large-scale pretraining
    and includes configurable batch sizes and data loading parameters.
    """

    dataset: torch.utils.data.Dataset

    def __init__(
        self,
        pretrained_tokenizer_name_or_path: str,
        objective: Literal["mlm", "enhanced_mlm"] = "mlm",
        max_sequence_length: int | None = 512,
        dataset_name_or_path: str | list[str] | None = None,
        file_format: str | None = None,
        data_split: str | None = None,
        text_column: str | None = "text",
        split_char: str | None = None,
        split_len: int | None = None,
        shuffle: bool | None = False,
        mlm_probability: float | None = 0.3,
        train_batch_size: int | None = 32,
        val_batch_size: int | None = 32,
        pretokenized: bool | None = False,
        num_workers: int | None = 0,
        collator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the pretraining data module.

        Args:
            pretrained_tokenizer_name_or_path (str): Path or name of HuggingFace tokenizer
                to use for text processing.
            objective (Literal["mlm", "enhanced_mlm"]): The training objective. Available options:
                "mlm", "enhanced_mlm".
            max_sequence_length (int | None): Maximum sequence length for tokenization.
                Longer sequences will be truncated. Defaults to 512.
            dataset_name_or_path (str | list[str] | None): Dataset name or path. Defaults to None.
            data_split (str, optional): Dataset split to use for pretraining. Defaults to 'train'.
            text_column (str, optional): Text column name pretrain with. Defaults to 'text'.
            split_char (str, optional): Character to split examples at. Only one of `split_char` and `split_len`
                should be specified. Defaults to None.
            split_len (int, optional): Number of characters to split examples at. Only one of `split_char` and
                `split_len` should be specified. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the dataset before pretraining. Defaults to False.
            train_batch_size (int | None): Batch size for training. Defaults to 32.
            val_batch_size (int | None): Batch size for validation. Defaults to 32.
            pretokenized (bool | None): Whether input is pre-tokenized. Defaults to False.
            num_workers (int | None): Number of workers for data loading. Defaults to 0.
            collator_kwargs (dict[str, Any] | None): Additional keyword arguments for the data collator.
                For example, the mlm_probability.

        """
        super().__init__()
        self.save_hyperparameters()
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
        self.collator = get_collator(objective)(
            tokenizer=tokenizer,
            max_sequence_length=self.hparams.max_sequence_length,
            text_column=self.hparams.text_column or "text",
            pretokenized=self.hparams.pretokenized,
            **(self.hparams.collator_kwargs or {}),
        )

    def prepare_data(self) -> None:
        """Prepare the dataset for training. Called once per node."""
        if os.path.isdir(self.hparams.dataset_name_or_path):
            # If local path, load from disk
            self.dataset = load_dataset(
                self.hparams.data_format or "json",
                data_dir=self.hparams.dataset_name_or_path,
                split=self.hparams.data_split or "train",
                streaming=not self.hparams.shuffle,  # We can't stream if we're shuffling
            )
        else:
            # If not local path, try HF
            self.dataset = load_dataset(
                self.hparams.dataset_name_or_path,
                split=self.hparams.data_split or "train",
                streaming=not self.hparams.shuffle,  # We can't stream if we're shuffling
            )

        if self.hparams.split_char or self.hparams.split_len:
            self.dataset.map(
                functools.partial(
                    chunk_examples,
                    column=self.hparams.column_text,
                    split_char=self.hparams.split_char,
                    split_len=self.hparams.split_len,
                )
            )

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            DataLoader: PyTorch DataLoader configured for MLM training
                with the specified batch size, collation function, and
                data loading parameters.

        """
        return DataLoader(
            self.dataset,
            collate_fn=self.collator,
            shuffle=self.hparams.shuffle,
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
        learning_rate: float | None = 1e-5,
        weight_decay: float | None = 0.001,
        compile_model: bool | None = True,
        pretrained_tokenizer_name_or_path: str | None = None,
        optimizer_class: str | None = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_warmup_kind: Literal["constant", "linear", "cosine", "exponential"] | None = "linear",
        scheduler_warmup_steps: int | None = 1000,
        scheduler_warmup_decay: float = 0.1,
        scheduler_training_kind: Literal["constant", "linear", "cosine", "exponential"] = "constant",
        scheduler_training_steps: int = -1,
        scheduler_training_decay: float = 1.0,
        scheduler_cooldown_kind: Literal["constant", "linear", "cosine", "exponential"] = "linear",
        scheduler_cooldown_steps: int = 0,
        scheduler_cooldown_decay: float = 0.0,
        objective: Literal["mlm", "enhanced_mlm"] = "mlm",
        model_config_kwargs: "dict[str, Any] | None" = None,
        model_kwargs: "dict[str, Any] | None" = None,
    ):
        """Initialize the BertBlocks pretraining module.

        Args:
            learning_rate (float, optional): Peak learning rate for optimization. Defaults to 1e-7.
            weight_decay (float, optional): Weight decay coefficient for AdamW. Defaults to 1e-6.
            compile_model (bool, optional): Whether to compile the model with torch.compile.
                Defaults to True for better performance.
            pretrained_tokenizer_name_or_path (str, optional): Path to pretrained tokenizer; if provided, will
                overwrite the model vocab size using the given tokenizer. Defaults to None.
            optimizer_class (str, optional): Optimizer class name. Defaults to "adamw".
            optimizer_kwargs (dict[str, Any], optional): Optional arguments to pass to torch.optim.optimizer.
            scheduler_warmup_kind (Literal["constant", "linear", "exponential", "cosine"], optional): scheduler kind
                for warmup phase. Defaults to "linear".
            scheduler_warmup_steps (int, optional): Number of steps in warmup phase. Defaults to 1000.
            scheduler_warmup_decay (float, optional): Decay value for phase. Usage depends on scheduler kind chosen for
                warmup phase. Defaults to 0.1.
            scheduler_training_kind (Literal["constant", "linear", "exponential", "cosine"], optional): scheduler kind
                for the training phase. Defaults to "constant".
            scheduler_training_steps (int, optional): Number of steps in training phase. Defaults to -1 (remains in this
                phase forever).
            scheduler_training_decay (float, optional): Decay value for phase. Usage depends on scheduler kind chosen
                for training phase. Defaults to 1 (no decay with constant kind).
            scheduler_cooldown_kind (Literal["constant", "linear", "exponential", "cosine"], optional): scheduler kind
                for the cooldown phase. Defaults to "constant".
            scheduler_cooldown_steps (int, optional): Number of steps in cooldown phase. Defaults to 0 (no cooldown).
            scheduler_cooldown_decay (float, optional): Decay value for phase. Usage depends on scheduler kind chosen
                for cooldown phase. Defaults to 0.0.
            objective: The training objective. Available options:
                "mlm", "enhanced_mlm".
            model_config_kwargs (dict[str, Any], optional): Optional dictionary of model configuration options passed
                to BertBlocksConfig for instantiation.
            model_kwargs (dict[str, Any], optional): Optional dictionary of model-specific and objective-specific
                arguments.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model_config = BertBlocksConfig(**(model_config_kwargs or {}))
        # Patch model config with tokenizer vocab size if given
        if self.hparams.pretrained_tokenizer_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
            self.model_config.vocab_size = tokenizer.vocab_size
            del tokenizer
        self.model = get_model(objective)(self.model_config, **(model_kwargs or {}))
        if self.hparams.compile_model:
            torch.set_float32_matmul_precision("high")
            self.model = torch.compile(self.model, dynamic=True)

    def configure_optimizers(self) -> tuple[list["torch.optim.Optimizer"], list[dict[str, Any]]]:
        """Configure optimizers and learning rate schedulers.

        Sets up optimizer with weight decay only applied to non-bias parameters
        (excluding norm parameters). Uses a sequential learning rate schedule
        with linear warmup followed by exponential decay.

        Returns:
            tuple: Contains optimizer list and scheduler configuration list.
                The scheduler is configured to update every step during training.

        """
        norm_cls = [RMSNorm, LayerNorm, GroupNorm, DeepNorm, DynamicTanhNorm]
        decay_parameters = get_parameter_names(self.model, norm_cls)
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
        optimizer_kwargs = self.hparams.optimizer_kwargs or {}
        optimizer_kwargs.update({"lr": self.hparams.learning_rate})
        optimizer = get_optimizer(self.hparams.optimizer_class, optimizer_grouped_parameters, optimizer_kwargs)
        scheduler = get_scheduler(
            optimizer,
            self.hparams.scheduler_warmup_kind,
            self.hparams.scheduler_warmup_steps,
            self.hparams.scheduler_warmup_decay,
            self.hparams.scheduler_training_kind,
            self.hparams.scheduler_training_steps,
            self.hparams.scheduler_training_decay,
            self.hparams.scheduler_cooldown_kind,
            self.hparams.scheduler_cooldown_steps,
            self.hparams.scheduler_cooldown_decay,
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
        output = self.model(**batch)
        self.log("loss/train", output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single validation step."""
        output = self.model(**batch)
        self.log("loss/validation", output.loss, prog_bar=True)
        return output.loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log grad norms at each optimizer step."""
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict({f"gradnorm/{k}": v for k, v in norms.items()})

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


__all__ = ["BertBlocksPretrainingDataModule", "BertBlocksPretrainingModule"]
