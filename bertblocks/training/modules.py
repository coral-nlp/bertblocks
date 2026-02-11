import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torchmetrics import MetricCollection

from pathlib import Path

import lightning as L
import torch
import torchmetrics
from datasets import load_dataset
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.trainer_pt_utils import get_parameter_names

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.model import get_model_cls
from bertblocks.modeling.norms import DeepNorm, DynamicTanhNorm, GroupNorm, LayerNorm, RMSNorm
from bertblocks.training.data import EmptyDataset, _cache_paths, _load_dataset, _tokenize_batch, _truncated_add_length
from bertblocks.training.metrics import get_metrics_for_task
from bertblocks.training.objectives import get_collator_cls
from bertblocks.training.optimizer import get_optimizer
from bertblocks.training.packing import PackingBatchSampler, PackingCollatorWrapper, PackingIterableDataset
from bertblocks.training.scheduler import get_scheduler


class BertBlocksPretrainingDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for BertBlocks MLM pretraining.

    This DataModule handles all aspects of data loading for pretraining,
    including dataset preparation, tokenization, and batch creation.
    Currently configured to use the TinyStories dataset but can be easily
    adapted for other datasets.

    The module supports streaming datasets for large-scale pretraining
    and includes configurable batch sizes and data loading parameters.
    """

    train_dataset: torch.utils.data.Dataset
    val_dataset: torch.utils.data.Dataset | None

    def __init__(
        self,
        train_dataset_name_or_path: str | list[str],
        pretrained_tokenizer_name_or_path: str,
        objective: Literal["mlm", "enhanced_mlm", "diffusion"] = "mlm",
        max_sequence_length: int = 512,
        val_dataset_name_or_path: str | list[str] | None = None,
        train_split: str | None = None,
        val_split: str | None = None,
        file_format: str | None = None,
        text_column: str = "text",
        streaming: bool = False,
        shuffle: bool = False,
        num_shards: int | None = None,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        pretokenized_train: bool = False,
        pretokenized_val: bool = False,
        num_workers: int = 0,
        collator_kwargs: dict[str, Any] | None = None,
        packing: bool = False,
        packing_pad_to_budget: bool = False,
        cache_dir: str | None = None,
        data_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the pretraining data module.

        Args:
            train_dataset_name_or_path (str | list[str]): Dataset name or path to load via huggingface
                datasets to use for training.
            pretrained_tokenizer_name_or_path (str): Path or name of HuggingFace tokenizer
                to use for text processing.
            objective (Literal["mlm", "enhanced_mlm", "diffusion"]): The training objective. Available options:
                "mlm", "enhanced_mlm", "diffusion".
            max_sequence_length (int, optional): Maximum sequence length for tokenization.
                Longer sequences will be truncated. Defaults to 512.
            val_dataset_name_or_path (str | list[str], optional): Dataset name or path to load via huggingface
                datasets to use for validation. Defaults to None.
            train_split (str, optional): Dataset split to use for training. Defaults to 'train'.
            val_split (str, optional): Dataset split to use for training. Defaults to 'validation'.
            file_format (str, optional): File format to use if loading data from disk. Defaults to 'json'.
            text_column (str, optional): Text column name pretrain with. Defaults to 'text'.
            streaming (bool, optional): Whether to use streaming data. Defaults to False.
            shuffle (bool, optional): Whether to shuffle the dataset before pretraining. Shuffling is only supported
                for non-streaming datasets and only when not using sequence packing. Defaults to False.
            num_shards (int, optional): Number of shards to partition the dataset into. Defaults to None (no sharding).
            train_batch_size (int, optional): Batch size for training. Defaults to 32.
            val_batch_size (int, optional): Batch size for validation. Defaults to 32.
            pretokenized_train (bool, optional): Whether train dataset is pre-tokenized. Defaults to False.
            pretokenized_val (bool, optional): Whether validation dataset is pre-tokenized. Defaults to False.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            collator_kwargs (dict[str, Any], optional): Additional keyword arguments for the data collator.
                For example, the mlm_probability.
            packing (bool, optional): Enable sequence packing. Defaults to False.
            packing_pad_to_budget (bool, optional): When True, packed batches are padded to token_budget for
                fixed shapes (useful for torch.compile). When False, uses dynamic shapes. Defaults to False.
            cache_dir (str, optional): Directory to cache data transformations. Defaults to None.
            data_kwargs (dict[str, Any], optional): Additional keyword arguments passed to huggingface loader functions.
        """
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
        self.collator = get_collator_cls(objective)(
            pad_token_id=self.tokenizer.pad_token_id or 0,
            mask_token_id=self.tokenizer.mask_token_id or 1,
            vocab_size=self.tokenizer.vocab_size,
            max_sequence_length=self.hparams.max_sequence_length,
            text_column=self.hparams.text_column or "text",
            **(self.hparams.collator_kwargs or {}),
        )

    def prepare_data(self) -> None:
        """Download and tokenize data. Called only on LOCAL_RANK=0 per node (Lightning handles this automatically)."""
        # Construct cache file paths if cache_dir is specified
        train_cache, val_cache, _ = _cache_paths(self.hparams.cache_dir)

        # Tokenize training dataset to populate HuggingFace's cache
        # Other ranks will load from cache in setup(); skip if already local path
        if not os.path.isdir(self.hparams.train_dataset_name_or_path):
            if not self.hparams.pretokenized_train:
                kwargs: dict[str, Any] = (
                    {"batched": True} if self.hparams.streaming else {"batched": True, "cache_file_name": train_cache}
                )
                _ = _load_dataset(
                    dataset_name_or_path=self.hparams.train_dataset_name_or_path,
                    split=self.hparams.train_split or "train",
                    add_index=self.hparams.packing,
                    file_format=self.hparams.file_format,
                    streaming=self.hparams.streaming,
                    **self.hparams.data_kwargs or {},
                ).map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    **kwargs,
                )
            else:
                kwargs = (
                    {"batched": False} if self.hparams.streaming else {"batched": False, "cache_file_name": train_cache}
                )
                _ = _load_dataset(
                    dataset_name_or_path=self.hparams.train_dataset_name_or_path,
                    split=self.hparams.train_split or "train",
                    add_index=self.hparams.packing,
                    file_format=self.hparams.file_format,
                    streaming=self.hparams.streaming,
                    **self.hparams.data_kwargs or {},
                ).map(lambda x: _truncated_add_length(x, self.hparams.max_sequence_length), **kwargs)

        # Optionally prepare validation dataset
        if (
            self.hparams.val_dataset_name_or_path is not None
            and not self.hparams.streaming
            and not os.path.isdir(self.hparams.val_dataset_name_or_path)
        ):
            if not self.hparams.pretokenized_val:
                kwargs = (
                    {"batched": True} if self.hparams.streaming else {"batched": True, "cache_file_name": val_cache}
                )
                _ = _load_dataset(
                    dataset_name_or_path=self.hparams.val_dataset_name_or_path,
                    split=self.hparams.val_split or "validation",
                    add_index=self.hparams.packing,
                    file_format=self.hparams.file_format,
                    streaming=self.hparams.streaming,
                    **self.hparams.data_kwargs or {},
                ).map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    **kwargs,
                )
            else:
                kwargs = (
                    {"batched": False} if self.hparams.streaming else {"batched": False, "cache_file_name": val_cache}
                )
                _ = _load_dataset(
                    dataset_name_or_path=self.hparams.val_dataset_name_or_path,
                    split=self.hparams.val_split or "validation",
                    add_index=self.hparams.packing,
                    file_format=self.hparams.file_format,
                    streaming=self.hparams.streaming,
                    **self.hparams.data_kwargs or {},
                ).map(lambda x: _truncated_add_length(x, self.hparams.max_sequence_length), **kwargs)

    def setup(self, stage: str | None = None) -> None:
        """Load the dataset for training. Called on every process - loads from cache populated by prepare_data()."""
        if stage == "fit" or stage is None:
            train_cache, val_cache, _ = _cache_paths(self.hparams.cache_dir)
            self.train_dataset = _load_dataset(
                dataset_name_or_path=self.hparams.train_dataset_name_or_path,
                split=self.hparams.train_split or "train",
                add_index=self.hparams.packing,
                file_format=self.hparams.file_format,
                streaming=self.hparams.streaming,
                **self.hparams.data_kwargs or {},
            )
            if self.hparams.val_dataset_name_or_path is not None:
                self.val_dataset = _load_dataset(
                    dataset_name_or_path=self.hparams.val_dataset_name_or_path,
                    split=self.hparams.val_split or "validation",
                    add_index=self.hparams.packing,
                    file_format=self.hparams.file_format,
                    streaming=self.hparams.streaming,
                    **self.hparams.data_kwargs or {},
                )
            else:
                self.val_dataset = None

            if not self.hparams.pretokenized_train:
                kwargs: dict[str, Any] = (
                    {"batched": True} if self.hparams.streaming else {"batched": True, "cache_file_name": train_cache}
                )
                self.train_dataset = self.train_dataset.map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    **kwargs,
                )
                #  When packing, ensure deterministic order across all ranks after map by sorting by original index
                if self.hparams.packing and not self.hparams.streaming:
                    self.train_dataset = self.train_dataset.sort("_idx")
                    self.train_dataset = self.train_dataset.remove_columns(["_idx"])

            else:
                kwargs = (
                    {"batched": False} if self.hparams.streaming else {"batched": False, "cache_file_name": train_cache}
                )
                self.train_dataset = self.train_dataset.map(
                    lambda x: _truncated_add_length(x, self.hparams.max_sequence_length), **kwargs
                )
                # When packing, ensure deterministic order across all ranks after map by sorting by original index
                if self.hparams.packing and not self.hparams.streaming:
                    self.train_dataset = self.train_dataset.sort("_idx")
                    self.train_dataset = self.train_dataset.remove_columns(["_idx"])

            if self.hparams.val_dataset_name_or_path is not None and not self.hparams.pretokenized_val:
                kwargs = (
                    {"batched": True} if self.hparams.streaming else {"batched": True, "cache_file_name": val_cache}
                )
                self.val_dataset = self.val_dataset.map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    **kwargs,
                )
            elif self.val_dataset is not None:
                kwargs = (
                    {"batched": False} if self.hparams.streaming else {"batched": False, "cache_file_name": val_cache}
                )
                self.val_dataset = self.val_dataset.map(
                    lambda x: _truncated_add_length(x, self.hparams.max_sequence_length), **kwargs
                )

            # For non-packing mode, shard the dataset across ranks
            if not self.hparams.streaming and self.trainer.world_size > 1 and not self.hparams.packing:
                self.train_dataset = self.train_dataset.shard(
                    num_shards=self.trainer.world_size, index=self.trainer.global_rank, contiguous=True
                )

            if self.hparams.streaming and self.hparams.shuffle:
                warnings.warn(
                    "Shuffling is not supported for streamed datasets. Falling back to unshuffled "
                    "mode. You may need to pre-shuffle your data to get correct results.",
                    stacklevel=2,
                )
                self.hparams.shuffle = False

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            DataLoader configured for pretraining, optionally with packed batches.
        """
        if self.hparams.packing:
            token_budget = self.hparams.max_sequence_length * self.hparams.train_batch_size
            if self.hparams.streaming:
                # Use batch_size=None since dataset yields batches internally
                dataloader = DataLoader(
                    PackingIterableDataset(
                        dataset=self.train_dataset,
                        token_budget=token_budget,
                        length_column="length",
                        drop_last=False,
                    ),
                    batch_size=None,
                    collate_fn=PackingCollatorWrapper(
                        base_collator=self.collator,
                        token_budget=token_budget if self.hparams.packing_pad_to_budget else 0,
                    ),
                    num_workers=self.hparams.num_workers,
                )
            else:
                dataloader = DataLoader(
                    self.train_dataset,
                    batch_sampler=PackingBatchSampler(
                        lengths=list(self.train_dataset["length"]),
                        token_budget=token_budget,
                        world_size=self.trainer.world_size,
                        rank=self.trainer.global_rank,
                        shuffle=self.hparams.shuffle,
                        drop_last=True,
                    ),
                    collate_fn=PackingCollatorWrapper(
                        base_collator=self.collator,
                        token_budget=token_budget if self.hparams.packing_pad_to_budget else 0,
                    ),
                    num_workers=self.hparams.num_workers,
                )
        else:
            dataloader = DataLoader(
                self.train_dataset,
                collate_fn=self.collator,
                shuffle=self.hparams.shuffle,
                batch_size=self.hparams.train_batch_size,
                num_workers=self.hparams.num_workers,
            )

        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.

        Returns:
            DataLoader configured for validation or dummy dataloader if no validation dataset is set.
        """
        if self.val_dataset is None:
            return DataLoader(EmptyDataset(), batch_size=self.hparams.val_batch_size, shuffle=False)

        return DataLoader(
            self.val_dataset,
            collate_fn=self.collator,
            shuffle=False,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
        )


class BertBlocksFinetuningDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for finetuning tasks.

    Supports classification, token classification, and question answering tasks
    with flexible dataset loading from HuggingFace Hub or local files.
    """

    def __init__(
        self,
        task: Literal["classification", "question_answering", "token_classification"],
        pretrained_tokenizer_name_or_path: str,
        dataset_name_or_path: str | None = None,
        dataset_config_name: str | None = None,
        max_sequence_length: int = 512,
        train_split: str = "train",
        val_split: str = "validation",
        test_split: str = "test",
        text_column: str = "text",
        label_column: str = "label",
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
        collator_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the finetuning data module.

        Args:
            task: The task type (classification, token_classification, question_answering).
            pretrained_tokenizer_name_or_path: HuggingFace tokenizer name or path.
            dataset_name_or_path: Dataset name (HF Hub) or local path. If None, datasets must be provided separately.
            dataset_config_name: Dataset configuration name for HF datasets.
            max_sequence_length: Maximum sequence length for tokenization.
            train_split: Name of training split.
            val_split: Name of validation split.
            test_split: Name of test split.
            text_column: Name of text column in dataset.
            label_column: Name of label column in dataset.
            train_batch_size: Training batch size.
            val_batch_size: Validation batch size.
            test_batch_size: Test batch size.
            num_workers: Number of workers for data loading.
            shuffle_train: Whether to shuffle training data.
            collator_kwargs: Additional arguments for the data collator.
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize tokenizer and collator
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        collator_cls = get_collator_cls(task)
        self.collator = collator_cls(
            pad_token_id=self.tokenizer.pad_token_id or 0,
            mask_token_id=self.tokenizer.mask_token_id or 1,
            vocab_size=self.tokenizer.vocab_size,
            max_sequence_length=max_sequence_length,
            text_column=text_column,
            label_column=label_column,
            **(collator_kwargs or {}),  # type: ignore
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Download datasets if needed. Called once per node."""
        if self.hparams.dataset_name_or_path is not None:
            _ = _load_dataset(self.hparams.dataset_name_or_path, name=self.hparams.dataset_config_name, split=None)

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each process. Called on every process.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        if self.hparams.dataset_name_or_path is None:
            return  # Datasets should be set manually
        dataset = load_dataset(self.hparams.dataset_name_or_path, name=self.hparams.dataset_config_name)

        if stage == "fit" or stage is None:
            if self.hparams.train_split in dataset:
                self.train_dataset = dataset[self.hparams.train_split].map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    batched=True,
                )
            else:
                raise ValueError(f"Train split {self.hparams.train_split} not found, got: {dataset.keys()}")

            if self.hparams.val_split is not None and self.hparams.val_split in dataset:
                self.val_dataset = dataset[self.hparams.val_split].map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    batched=True,
                )
            else:
                raise ValueError(f"Validation split {self.hparams.val_split} not found, got: {dataset.keys()}")

        if stage == "test" or stage is None:
            if self.hparams.test_split is not None and self.hparams.test_split in dataset:
                self.test_dataset = dataset[self.hparams.test_split].map(
                    lambda x: _tokenize_batch(
                        x,
                        tokenizer=self.tokenizer,
                        text_column=self.hparams.text_column,
                        max_sequence_length=self.hparams.max_sequence_length,
                    ),
                    batched=True,
                )
            else:
                raise ValueError(f"Test split {self.hparams.val_split} not found, got: {dataset.keys()}")

    def set_datasets(
        self,
        train: "torch.utils.data.Dataset | None" = None,
        val: "torch.utils.data.Dataset | None" = None,
        test: "torch.utils.data.Dataset | None" = None,
    ) -> None:
        """Manually set datasets for custom data loading.

        Args:
            train: Training dataset.
            val: Validation dataset.
            test: Test dataset.
        """
        if train is not None:
            self.train_dataset = train
        if val is not None:
            self.val_dataset = val
        if test is not None:
            self.test_dataset = test

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            DataLoader configured for training or None if no training dataset.
        """
        if self.train_dataset is None:
            return DataLoader(EmptyDataset(), batch_size=16, shuffle=False)

        return DataLoader(
            self.train_dataset,
            collate_fn=self.collator,
            shuffle=self.hparams.shuffle_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.

        Returns:
            DataLoader configured for validation or None if no validation dataset.
        """
        if self.val_dataset is None:
            return DataLoader(EmptyDataset(), batch_size=16, shuffle=False)

        return DataLoader(
            self.val_dataset,
            collate_fn=self.collator,
            shuffle=False,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader.

        Returns:
            DataLoader configured for testing or None if no test dataset.
        """
        if self.test_dataset is None:
            return DataLoader(EmptyDataset(), batch_size=16, shuffle=False)

        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            shuffle=False,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
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
        optimizer_quantized: bool = False,
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_warmup_kind: Literal["constant", "linear", "cosine", "exponential"] | None = "linear",
        scheduler_warmup_steps: int | None = 1000,
        scheduler_warmup_decay: float = 0.1,
        scheduler_training_kind: Literal["constant", "linear", "cosine", "exponential"] = "constant",
        scheduler_training_steps: int = -1,
        scheduler_training_decay: float = 1.0,
        scheduler_cooldown_kind: Literal["constant", "linear", "inverse-sqrt", "cosine", "exponential"] = "linear",
        scheduler_cooldown_steps: int = 0,
        scheduler_cooldown_decay: float = 0.0,
        objective: Literal["mlm", "enhanced_mlm", "diffusion"] = "mlm",
        gradient_checkpointing: bool = False,
        use_ddp_join: bool = False,
        ddp_join_divide_by_initial_world_size: bool = True,
        ddp_join_throw_on_early_termination: bool = False,
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
            optimizer_quantized (bool, optional): Whether to use quantized optimization. Defaults to False.
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
            scheduler_cooldown_kind (Literal["constant", "linear", "inverse-sqrt", "exponential", "cosine"], optional):
                scheduler kind for the cooldown phase. Defaults to "constant".
            scheduler_cooldown_steps (int, optional): Number of steps in cooldown phase. Defaults to 0 (no cooldown).
            scheduler_cooldown_decay (float, optional): Decay value for phase. Usage depends on scheduler kind chosen
                for cooldown phase. Defaults to 0.0.
            objective: The training objective. Available options:
                "mlm", "diffusion", "enhanced_mlm".
            gradient_checkpointing (bool, optional): Whether to enable gradient checkpointing to reduce
                activation memory at the cost of additional compute. Defaults to False.
            model_config_kwargs (dict[str, Any], optional): Optional dictionary of model configuration options passed
                to BertBlocksConfig for instantiation.
            model_kwargs (dict[str, Any], optional): Optional dictionary of model-specific and objective-specific
                arguments.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])
        self.model_config = BertBlocksConfig(**(model_config_kwargs or {}))
        # Patch model config with tokenizer info if given
        if self.hparams.pretrained_tokenizer_name_or_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
            self.model_config.vocab_size = tokenizer._tokenizer.get_vocab_size(
                with_added_tokens=True
            )  # tokenizer.vocab_size
            self.pad_token_id = tokenizer.pad_token_id
            self.model_config.pad_token_id = tokenizer.pad_token_id
            self.mask_token_id = tokenizer.mask_token_id
            del tokenizer
        else:
            self.pad_token_id = 0
            self.model_config.pad_token_id = 0
            self.mask_token_id = 0
        self.model = get_model_cls(objective)(self.model_config, **(model_kwargs or {}))
        if self.hparams.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        torch.set_float32_matmul_precision("high")

    def configure_model(self) -> None:
        """Compile the model core after DDP setup so each rank has its own CUDA context.

        Only the inner compute path (embed -> encode -> norm -> scale) is compiled.
        Unpadding and output assembly remain outside the compiler scope to avoid
        dynamic shape issues from varying batch dimensions.
        """
        if self.hparams.compile_model:
            if self.trainer.world_size > 1:
                torch._dynamo.config.optimize_ddp = True
            self.model.model._forward = torch.compile(self.model.model._forward, dynamic=True)  # type: ignore

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
        optimizer = get_optimizer(
            self.hparams.optimizer_class,
            optimizer_grouped_parameters,
            optimizer_kwargs,
            quantized=self.hparams.optimizer_quantized,
        )
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
        self.log("packing/tokens_per_batch", (batch["attention_mask"] != -1).float().sum(), sync_dist=False)
        self.log(
            "packing/efficiency",
            (batch["attention_mask"] != -1).float().sum() / batch["attention_mask"].numel(),
            sync_dist=False,
        )
        self.log(
            "packing/sequences_per_batch",
            float(batch["attention_mask"].max()) + 1  # Packed sequences
            if (batch["attention_mask"] == -1).any()
            else float(batch["attention_mask"].shape[0]),  # Unpacked sequences
            sync_dist=False,
        )
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


class BertBlocksFinetuningModule(L.LightningModule):
    """PyTorch Lightning module for BertBlocks finetuning.

    This module handles finetuning of pretrained BertBlocks models on downstream tasks
    including classification, token classification, and question answering.
    """

    def __init__(
        self,
        task: Literal["classification", "token_classification", "question_answering"],
        pretrained_model_name_or_path: str,
        num_labels: int | None = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        compile_model: bool = True,
        optimizer_class: str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        optimizer_quantized: bool = False,
        scheduler_type: Literal["linear", "cosine", "constant", "polynomial"] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        """Initialize the BertBlocks finetuning module.

        Args:
            task: The finetuning task type.
            pretrained_model_name_or_path: Path to pretrained model.
            num_labels: Number of labels for classification tasks. Auto-detected if None.
            learning_rate: Peak learning rate for optimization.
            weight_decay: Weight decay coefficient.
            compile_model: Whether to compile the model with torch.compile.
            optimizer_class: Optimizer class name.
            optimizer_kwargs: Additional optimizer arguments.
            scheduler_type: Type of learning rate scheduler.
            scheduler_kwargs: Additional scheduler arguments.
            warmup_steps: Number of warmup steps (overrides warmup_ratio).
            warmup_ratio: Ratio of total steps to use for warmup.
            gradient_checkpointing: Whether to enable gradient checkpointing to reduce
                activation memory at the cost of additional compute.
        """
        super().__init__()
        self.save_hyperparameters()

        if task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path, num_labels=num_labels if num_labels is not None else 2
            )
        elif task == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path, num_labels=num_labels if num_labels is not None else 2
            )
        elif task == "question_answering":
            self.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path)
        else:
            raise ValueError(f"Unknown task: {task}")

        metric_dict = get_metrics_for_task(task, num_labels if num_labels is not None else 2)
        self.val_metrics = torchmetrics.MetricCollection(metric_dict, prefix="val/")
        self.test_metrics = self.val_metrics.clone(prefix="test/")

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if compile_model:
            torch.set_float32_matmul_precision("high")
            self.model = torch.compile(self.model, dynamic=True)

    def configure_optimizers(self) -> "torch.optim.Optimizer | dict[str, Any]":
        """Configure optimizers and learning rate schedulers."""
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
        optimizer = get_optimizer(
            self.hparams.optimizer_class,
            optimizer_grouped_parameters,
            optimizer_kwargs,
            self.hparams.optimizer_quantized,
        )

        if self.hparams.scheduler_type is None:
            return optimizer

        if self.hparams.warmup_steps > 0:
            warmup_steps = self.hparams.warmup_steps
        elif self.hparams.warmup_ratio > 0:
            # Estimate total steps (this is approximate)
            warmup_steps = int(self.trainer.estimated_stepping_batches * self.hparams.warmup_ratio)
        else:
            warmup_steps = 0

        # Create scheduler
        from transformers import get_scheduler as get_hf_scheduler

        scheduler = get_hf_scheduler(
            name=self.hparams.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            **(self.hparams.scheduler_kwargs or {}),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform training step."""
        torch.compiler.cudagraph_mark_step_begin()
        output = self.model(**batch)
        self.log("train/loss", output.loss, prog_bar=True)
        return output.loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform validation step."""
        torch.compiler.cudagraph_mark_step_begin()
        output = self.model(**batch)
        self.log("val/loss", output.loss, prog_bar=True)
        self._update_metrics(output, batch, self.val_metrics)
        self.log_dict(self.val_metrics, on_epoch=True)
        return output.loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform test step."""
        output = self.model(**batch)
        self.log("test/loss", output.loss)
        self._update_metrics(output, batch, self.test_metrics)
        self.log_dict(self.test_metrics, on_epoch=True)
        return output.loss

    def _update_metrics(
        self,
        output: "SequenceClassifierOutput | TokenClassifierOutput | QuestionAnsweringModelOutput",
        batch: "dict[str, torch.Tensor]",
        metrics: "MetricCollection",
    ) -> None:
        """Update metrics based on model output type."""
        labels = batch.get("labels")
        if labels is None:
            return

        if isinstance(output, SequenceClassifierOutput):
            predictions = output.logits
            metrics(predictions, labels)
        elif isinstance(output, TokenClassifierOutput):
            predictions = output.logits.view(-1, output.logits.shape[-1])
            labels_flat = labels.view(-1)
            mask = labels_flat != -100
            if torch.any(mask):
                metrics(predictions[mask], labels_flat[mask])
        elif isinstance(output, QuestionAnsweringModelOutput):
            start_logits = output.start_logits
            # end_logits = output.end_logits
            start_positions = batch.get("start_positions")
            end_positions = batch.get("end_positions")

            if start_positions is not None and end_positions is not None:
                # For now, just use start position accuracy as a proxy
                # More sophisticated QA metrics would require the actual text
                start_preds = start_logits.argmax(dim=-1)
                metrics(start_preds, start_positions)


__all__ = [
    "BertBlocksFinetuningDataModule",
    "BertBlocksFinetuningModule",
    "BertBlocksPretrainingDataModule",
    "BertBlocksPretrainingModule",
]
