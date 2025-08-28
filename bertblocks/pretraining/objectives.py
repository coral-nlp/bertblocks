from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

from ..modeling.model import BertBlocksForEnhancedMaskedLM, BertBlocksForMaskedLM, BertBlocksPreTrainedModel


class Collator(ABC):
    """Abstract data collator class for pretraining tasks.

    A data collator is responsible for processing raw input data into a format
    suitable for model training. This typically involves tokenization, padding,
    and applying any necessary transformations such as masking for language
    modeling tasks.

    Inherited classes must implement the `compute_labels` method.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str = "text",
        max_sequence_length: int | None = 256,
        pretokenized: bool | None = False,
    ) -> None:
        """Initialize the data collator.

        Args:
            tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to use for
                text processing.
            text_column (str): Name of the column containing text data in the dataset.
                Defaults to "text".
            max_sequence_length (int | None): Maximum sequence length after tokenization.
                Defaults to 256.
            pretokenized (bool | None): Whether the input data is already tokenized.
                Defaults to False.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_sequence_length = max_sequence_length
        self.pretokenized = pretokenized
        self.special_tokens = torch.tensor(self.tokenizer.all_special_ids)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a batch of examples.

        Args:
            batch (list[dict[str, Any]]): List of examples, where each example is a dictionary
                containing the raw input data.

        Returns:
            dict[str, Any]: Processed batch dictionary suitable for model input.
        """
        if not self.pretokenized:
            tokenized = self.tokenizer(
                [item[self.text_column] for item in batch],
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
        else:
            tokenized = batch
            tokenized = {
                "input_ids": torch.stack([item["input_ids"] for item in tokenized]),
                "attention_mask": torch.stack([item["attention_mask"] for item in tokenized]),
            }
        tokenized_with_labels = self.compute_labels(tokenized)
        tokenized_with_labels = {
            k: v for k, v in tokenized_with_labels.items() if k in ["input_ids", "attention_mask", "labels"]
        }
        return tokenized_with_labels

    @abstractmethod
    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute the labels for the given batch of tokenized inputs.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed labels for the batch.
        """
        pass


class MaskedLanguageModelingCollator(Collator):
    """Data collator for masked language modeling pretraining.

    This collator handles tokenization (if needed) and applies dynamic masking
    to create MLM training examples. It uses HuggingFace's DataCollatorForLanguageModeling
    under the hood but adds support for custom tokenization and preprocessing.

    The collator can work with both pre-tokenized and raw text data, and ensures
    consistent padding and masking across batches.

    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        text_column: str = "text",
        max_sequence_length: int | None = 256,
        pretokenized: bool | None = False,
        mlm_probability: float | None = 0.3,
    ):
        """Initialize the MLM collator.

        Args:
            tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to use for
                text processing.
            text_column (str): Name of the column containing text data in the dataset.
                Defaults to "text".
            max_sequence_length (int | None): Maximum sequence length after tokenization.
                Defaults to 256.
            pretokenized (bool | None): Whether the input data is already tokenized.
                Defaults to False.
            mlm_probability (float | None): Probability of masking tokens for MLM.
                Defaults to 0.3 (30% of tokens will be masked).
        """
        super().__init__(
            tokenizer=tokenizer,
            text_column=text_column,
            max_sequence_length=max_sequence_length,
            pretokenized=pretokenized,
        )
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=64,
            return_tensors="pt",
        )

    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute the MLM labels for the given batch of tokenized inputs.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed MLM labels for the batch.
        """
        return self.mlm_collator(
            [{k: v[i] for k, v in tokenized.items()} for i in range(tokenized["input_ids"].shape[0])]
        )


class EnhancedMaskedLanguageModelingCollator(Collator):
    """Data collator for enhanced masked language modeling pretraining.

    The collator is the same as the MaskedLanguageModelingCollator but turns
    off masking in the input sequence, because masking is handled by the model.
    """

    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute labels for enhanced masked language modeling.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed labels for the batch.
        """
        labels = torch.where(
            tokenized["attention_mask"].bool() & torch.isin(tokenized["input_ids"], self.special_tokens, invert=True),
            tokenized["input_ids"],
            -100,
        )
        tokenized["labels"] = labels
        return tokenized


def get_collator(objective: str) -> type[Collator]:
    """Get the appropriate data collator for the given objective.

    Args:
        objective (str): The training objective. Available options:
            "mlm", "enhanced_mlm".

    Raises:
        ValueError: If the objective is unknown.

    Returns:
        type[Collator]: The corresponding data collator class.
    """
    match objective:
        case "mlm":
            return MaskedLanguageModelingCollator
        case "enhanced_mlm":
            return EnhancedMaskedLanguageModelingCollator
        case _:
            raise ValueError(f"Unknown objective: {objective}")


def get_model(objective: str) -> type[BertBlocksPreTrainedModel]:
    """Get the appropriate model class for the given objective.

    Args:
        objective (str): The training objective. Available options:
            "mlm", "enhanced_mlm".

    Raises:
        ValueError: If the objective is unknown.

    Returns:
        type[BertBlocksPreTrainedModel]: The corresponding model class.
    """
    match objective:
        case "mlm":
            return BertBlocksForMaskedLM
        case "enhanced_mlm":
            return BertBlocksForEnhancedMaskedLM
        case _:
            raise ValueError(f"Unknown objective: {objective}")
