from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


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
        label_column: str = None,
        max_sequence_length: int | None = 256,
        pretokenized: bool | None = False,
    ) -> None:
        """Initialize the data collator.

        Args:
            tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to use for text processing.
            text_column (str): Name of the column containing text data in the dataset. Defaults to "text".
            label_column (str): Name of the column containing label data in the dataset. Defaults to "label".
            max_sequence_length (int | None): Maximum sequence length after tokenization. Defaults to 256.
            pretokenized (bool | None): Whether the input data is already tokenized. Defaults to False.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
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
        if self.label_column:
            tokenized.update({"labels": [item[self.label_column] for item in batch]})
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

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        text_column: str = "text",
        max_sequence_length: int | None = 256,
        pretokenized: bool | None = False,
        mlm_probability: float | None = 0.3,
    ):
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

    def compute_labels(self, tokenized: dict[str, Any]) -> Any:
        """Compute the MLM labels for the given batch of tokenized inputs.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed MLM labels for the batch.
        """
        return self.mlm_collator(
            [{k: v[i] for k, v in tokenized.items()} for i in range(tokenized["input_ids"].shape[0])]
        )  # type: ignore


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


class TokenClassificationCollator(Collator):
    """Data collator for token classification tasks.

    This collator handles tokenization and formatting for token classification tasks
    like NER, POS tagging, etc. It tokenizes the input text and properly aligns
    the token-level labels with the tokenized sequence.

    Args:
        tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to use for text processing.
        text_column (str): Name of the column containing text data in the dataset. Defaults to "text".
        label_column (str): Name of the column containing label data in the dataset. Defaults to "labels".
        max_sequence_length (int | None): Maximum sequence length after tokenization. Defaults to 512.
        pretokenized (bool | None): Whether the input data is already tokenized. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        text_column: str = "text",
        label_column: str = "labels",
        max_sequence_length: int | None = 512,
        pretokenized: bool | None = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=label_column,
            max_sequence_length=max_sequence_length,
            pretokenized=pretokenized,
        )

    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute token classification labels for the given batch.

        For token classification, we need to ensure labels align with tokenized input.
        The labels should be padded to match the sequence length and use -100 for
        special tokens and padding.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The tokenized inputs with properly formatted labels.
        """
        if "labels" in tokenized:
            labels = tokenized["labels"]

            # Convert labels to tensor if they aren't already
            if not isinstance(labels[0], torch.Tensor):
                # Pad labels to match input sequence length
                max_length = tokenized["input_ids"].shape[1]
                padded_labels = []

                for label_seq in labels:
                    if isinstance(label_seq, list | tuple):
                        # Pad or truncate to max_length
                        if len(label_seq) > max_length:
                            padded_seq = label_seq[:max_length]
                        else:
                            padded_seq = list(label_seq) + [-100] * (max_length - len(label_seq))
                        padded_labels.append(padded_seq)
                    else:
                        # Single label, repeat for whole sequence (shouldn't happen for token classification)
                        padded_labels.append([label_seq] * max_length)

                tokenized["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            else:
                # Labels are already tensors, just stack them
                tokenized["labels"] = torch.stack(labels)

        return tokenized


class SequenceClassificationCollator(Collator):
    """Data collator for sequence classification tasks.

    This collator handles tokenization and formatting for sequence classification tasks
    like sentiment analysis, text classification, etc. It tokenizes the input text
    and preserves the labels for classification.

    Args:
        tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to use for text processing.
        text_column (str): Name of the column containing text data in the dataset. Defaults to "text".
        label_column (str): Name of the column containing label data in the dataset. Defaults to "label".
        max_sequence_length (int | None): Maximum sequence length after tokenization. Defaults to 512.
        pretokenized (bool | None): Whether the input data is already tokenized. Defaults to False.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        text_column: str = "text",
        label_column: str = "label",
        max_sequence_length: int | None = 512,
        pretokenized: bool | None = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=label_column,
            max_sequence_length=max_sequence_length,
            pretokenized=pretokenized,
        )

    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute sequence classification labels for the given batch.

        For sequence classification, we just need to preserve the original labels
        as they apply to the entire sequence.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The tokenized inputs with labels preserved.
        """
        # For sequence classification, labels are already in the correct format
        # Just ensure they're tensors
        if "labels" in tokenized:
            tokenized["labels"] = torch.tensor(tokenized["labels"])
        return tokenized


class QuestionAnsweringCollator(Collator):
    """Data collator for question answering tasks.

    This collator handles tokenization and formatting for question answering tasks
    like SQuAD. It processes question-context pairs and preserves start/end positions
    for answer span prediction.

    Args:
        tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to use for text processing.
        question_column (str): Name of the column containing question data. Defaults to "question".
        context_column (str): Name of the column containing context data. Defaults to "context".
        answer_column (str): Name of the column containing answer data. Defaults to "answers".
        max_sequence_length (int | None): Maximum sequence length after tokenization. Defaults to 512.
        pretokenized (bool | None): Whether the input data is already tokenized. Defaults to False.
        doc_stride (int): Stride for sliding window when context is too long. Defaults to 128.
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        text_column: str = "question",  # Override default for QA
        label_column: str = "answers",  # Override default for QA
        context_column: str = "context",
        max_sequence_length: int | None = 512,
        pretokenized: bool | None = False,
        doc_stride: int = 128,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_column=text_column,  # question column
            label_column=label_column,  # answers column
            max_sequence_length=max_sequence_length,
            pretokenized=pretokenized,
        )
        self.context_column = context_column
        self.doc_stride = doc_stride

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a batch of QA examples.

        For QA, we need to handle question-context pairs differently than the base class.
        """
        if not self.pretokenized:
            questions = [item[self.text_column] for item in batch]
            contexts = [item[self.context_column] for item in batch]

            # Tokenize question-context pairs
            tokenized = self.tokenizer(
                questions,
                contexts,
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
                return_offsets_mapping=True,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
            )
        else:
            # Handle pre-tokenized case
            tokenized = {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            }

        # Add answers/start_positions/end_positions if available
        if self.label_column and all(self.label_column in item for item in batch):
            tokenized.update({"answers": [item[self.label_column] for item in batch]})

        tokenized_with_labels = self.compute_labels(tokenized)

        # Keep QA-specific fields
        qa_fields = ["input_ids", "attention_mask", "start_positions", "end_positions"]
        tokenized_with_labels = {k: v for k, v in tokenized_with_labels.items() if k in qa_fields}
        return tokenized_with_labels

    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute question answering labels for the given batch.

        For QA, we need to find the start and end positions of the answer spans
        within the tokenized context.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The tokenized inputs with start/end positions.
        """
        if "answers" not in tokenized:
            # No answers provided, return with dummy positions for inference
            batch_size = tokenized["input_ids"].shape[0]
            tokenized["start_positions"] = torch.zeros(batch_size, dtype=torch.long)
            tokenized["end_positions"] = torch.zeros(batch_size, dtype=torch.long)
            return tokenized

        answers = tokenized["answers"]
        start_positions = []
        end_positions = []

        # For each example, find answer span positions
        for i, answer_data in enumerate(answers):
            if isinstance(answer_data, dict) and "answer_start" in answer_data:
                # SQuAD format: {"text": ["answer"], "answer_start": [42]}
                answer_start = answer_data["answer_start"][0] if answer_data["answer_start"] else 0
                answer_text = answer_data["text"][0] if answer_data["text"] else ""
            elif isinstance(answer_data, str):
                # Simple string answer - find in context (approximate)
                answer_start = 0  # Would need more sophisticated matching
                answer_text = answer_data
            else:
                # Default case
                answer_start = 0
                answer_text = ""

            # Convert character positions to token positions
            # This is simplified - proper implementation would use offset mappings
            start_pos = 1  # Default to position after [CLS]
            end_pos = 1  # Default to same position

            if hasattr(tokenized, "offset_mapping") and tokenized["offset_mapping"] is not None:
                # Use offset mapping for precise position finding
                offsets = tokenized["offset_mapping"][i]
                for token_idx, (start_char, end_char) in enumerate(offsets):
                    if start_char <= answer_start < end_char:
                        start_pos = token_idx
                    if start_char < answer_start + len(answer_text) <= end_char:
                        end_pos = token_idx
                        break

            start_positions.append(start_pos)
            end_positions.append(end_pos)

        tokenized["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
        tokenized["end_positions"] = torch.tensor(end_positions, dtype=torch.long)

        return tokenized


def get_collator_cls(
    objective: Literal["mlm", "enhanced_mlm", "classification", "token_classification", "question_answering"],
) -> type[Collator]:
    """Get the appropriate data collator for the given objective.

    Args:
        objective (str): The training objective. Available options:
            "mlm", "enhanced_mlm", "classification", "token_classification", "question_answering".

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
        case "classification":
            return SequenceClassificationCollator
        case "token_classification":
            return TokenClassificationCollator
        case "question_answering":
            return QuestionAnsweringCollator
        case _:
            raise ValueError(f"Unknown objective: {objective}")
