"""Masked Language Modeling (MLM) pretraining implementation for PolyBert.

This module provides a complete MLM pretraining setup using PyTorch Lightning,
including data loading, model configuration, optimization, and training logic.
It supports streaming datasets, flexible model compilation, and various
optimization strategies.

The implementation includes:
- MaskedLanguageModelingCollator for dynamic masking
- PolyBertPretrainingDataModule for data loading
- PolyBertPretrainingModule for training logic
- Support for model compilation and advanced optimization
"""

from pathlib import Path
from typing import Any

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizer
from transformers.trainer_pt_utils import get_parameter_names

from polybert.modeling import PolyBertConfig, PolyBertForMaskedLM

# Some needed compile settings
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = False
torch._functorch.config.enable_autograd_cache = False
torch._inductor.config.triton.cudagraph_trees = False  # Bug with cudagraph trees in this case


class MaskedLanguageModelingCollator:
    """Data collator for masked language modeling pretraining.

    This collator handles tokenization (if needed) and applies dynamic masking
    to create MLM training examples. It uses HuggingFace's DataCollatorForLanguageModeling
    under the hood but adds support for custom tokenization and preprocessing.

    The collator can work with both pre-tokenized and raw text data, and ensures
    consistent padding and masking across batches.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        max_sequence_length: int | None = 256,
        mlm_probability: float | None = 0.3,
        pretokenized: bool | None = False,
    ):
        """Initialize the MLM collator.

        Args:
            tokenizer: HuggingFace tokenizer for text processing.
            text_column: Name of the column containing text data in the dataset.
            max_sequence_length: Maximum sequence length after tokenization.
                Defaults to 256.
            mlm_probability: Probability of masking tokens for MLM.
                Defaults to 0.3 (30% of tokens will be masked).
            pretokenized: Whether the input data is already tokenized.
                Defaults to False.

        """
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=64,
            return_tensors="pt",
        )
        self.max_seq_len = max_sequence_length
        self.pretokenized = pretokenized

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a batch of examples for MLM training.

        Args:
            batch: List of examples, where each example is a dictionary
                containing the text data in the specified text_column.

        Returns:
            dict[str, Any]: batch dictionary containing:
                - input_ids: Token IDs with masked tokens (shape: [batch_size, seq_len])
                - attention_mask: Attention mask (shape: [batch_size, seq_len])
                - labels: Original token IDs for computing MLM loss (shape: [batch_size, seq_len])
                    -100 for non-masked tokens, original token IDs for masked tokens

        """
        num_samples = len(batch)
        if not self.pretokenized:
            batch_tokenized = self.tokenizer(
                [item[self.text_column] for item in batch],
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
        else:
            batch_tokenized = batch
        batch_tokenized = [{k: v[i] for k, v in batch_tokenized.items()} for i in range(num_samples)]
        batch_masked = self.mlm_collator(batch_tokenized)
        batch_out = {k: v for k, v in batch_masked.items() if k in ["input_ids", "attention_mask", "labels"]}
        return batch_out


class PolyBertPretrainingDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for PolyBert MLM pretraining.

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
        """Prepare the dataset for training.

        This method loads the TinyStories dataset in streaming mode
        for efficient large-scale pretraining. Called once per node.
        """
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

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


class PolyBertPretrainingModule(L.LightningModule):
    """PyTorch Lightning module for PolyBert MLM pretraining.

    This module encapsulates the complete training logic for PolyBert pretraining,
    including model initialization, optimization setup, and training step implementation.
    It supports advanced features like model compilation, sophisticated learning rate
    scheduling, and automatic checkpoint saving.

    The module automatically configures the PolyBert model based on the provided
    hyperparameters and handles all aspects of the training loop.
    """

    def __init__(
        self,
        pretrained_tokenizer_name_or_path: str,
        learning_rate: float | None = 1e-7,
        weight_decay: float | None = 1e-6,
        warmup_steps: int | None = 1_000,
        warmup_decay: float | None = 0.1,
        learning_rate_decay: float | None = 0.99999,
        compile_model: bool | None = True,
        hidden_size: int | None = 768,
        num_hidden_layers: int | None = 12,
        num_attention_heads: int | None = 12,
        intermediate_size: int | None = 3072,
        hidden_dropout_prob: float | None = 0.1,
        attention_probs_dropout_prob: float | None = 0.1,
        initializer_range: float | None = 0.02,
        initializer_cutoff_factor: float | None = 3.0,
        norm_eps: float | None = 1e-12,
    ):
        """Initialize the PolyBert pretraining module.

        Args:
            pretrained_tokenizer_name_or_path: Path or name of tokenizer for vocab size.
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
            hidden_size: Hidden dimension size. Defaults to 768.
            num_hidden_layers: Number of transformer layers. Defaults to 12.
            num_attention_heads: Number of attention heads. Defaults to 12.
            intermediate_size: Feed-forward intermediate size. Defaults to 3072.
            hidden_dropout_prob: Dropout probability for hidden layers.
                Defaults to 0.1.
            attention_probs_dropout_prob: Dropout probability for attention.
                Defaults to 0.1.
            initializer_range: Standard deviation for weight initialization.
                Defaults to 0.02.
            initializer_cutoff_factor: Cutoff factor for truncated normal init.
                Defaults to 3.0.
            norm_eps: Epsilon for layer normalization. Defaults to 1e-12.

        """
        super().__init__()
        self.save_hyperparameters()
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
        self.config = PolyBertConfig(
            hidden_size=hidden_size,
            num_blocks=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            initializer_cutoff_factor=initializer_cutoff_factor,
            norm_eps=norm_eps,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=tokenizer.vocab_size,
        )
        self.model = PolyBertForMaskedLM(self.config)
        self.model = torch.compile(
            self.model, options={"shape_padding": True, "trace.enabled": True, "trace.graph_diagram": True}
        )

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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
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
        torch.compiler.cudagraph_mark_step_begin()
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("loss/train", output.loss, prog_bar=True)
        return output.loss

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
