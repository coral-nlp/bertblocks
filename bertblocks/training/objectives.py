from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import Tensor

from bertblocks.modeling.utils import LogLinearNoise
from bertblocks.training.packing import is_packed_batch


class Collator(ABC):
    """Abstract data collator class for pretraining tasks.

    A data collator is responsible for processing pretokenized input data into a format
    suitable for model training. This includes padding and applying any necessary
    transformations such as masking for language modeling tasks.

    Inherited classes must implement the `compute_labels` method.
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        text_column: str = "text",
        label_column: str | None = None,
        max_sequence_length: int = 1024,
    ) -> None:
        """Initialize the data collator.

        Args:
            pad_token_id (int): Token ID used for padding sequences.
            mask_token_id (id): Token ID used for masking sequences.
            vocab_size (int): Size of the tokenizer vocabulary.
            text_column (str): Name of the column containing text data. Defaults to "text".
            label_column (str): Name of the column containing label data. Defaults to "label".
            max_sequence_length (int): Maximum sequence length for padding. Defaults to 1024.
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.text_column = text_column
        self.label_column = label_column
        self.max_sequence_length = max_sequence_length
        self.batch_keys = ["input_ids", "attention_mask", "labels"]

    def _get_valid_token_mask(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # Create valid token mask
        if is_packed_batch(attention_mask):
            # Packed format: valid tokens have attention_mask >= 0
            valid_mask = (attention_mask >= 0) & torch.isin(
                input_ids, [self.pad_token_id, self.mask_token_id], invert=True
            )
        else:
            # Standard format: valid tokens have attention_mask == 1
            valid_mask = attention_mask.bool() & torch.isin(
                input_ids, [self.pad_token_id, self.mask_token_id], invert=True
            )
        return valid_mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a batch of pretokenized examples.

        Args:
            batch (list[dict[str, Any]]): List of pretokenized examples, where each example
                contains "input_ids" and "attention_mask" keys.

        Returns:
            dict[str, Any]: Processed batch dictionary with padded tensors suitable for model input.
        """
        pad_id = self.pad_token_id
        seq_lens = [len(item["input_ids"]) for item in batch]

        # Convert to tensor first, then truncate to ensure all sequences are properly limited
        input_ids = [
            (
                item["input_ids"]
                if isinstance(item["input_ids"], torch.Tensor)
                else torch.tensor(item["input_ids"], dtype=torch.long)
            )[: self.max_sequence_length]
            for item in batch
        ]
        attention_mask = [
            (
                item["attention_mask"]
                if isinstance(item["attention_mask"], torch.Tensor)
                else torch.tensor(item["attention_mask"], dtype=torch.long)
            )[: self.max_sequence_length]
            for item in batch
        ]

        # Pad to max_sequence_length
        tokenized = {
            "input_ids": torch.stack(
                [
                    torch.nn.functional.pad(ids, (0, self.max_sequence_length - len(ids)), value=pad_id)
                    for ids in input_ids
                ]
            ),
            "attention_mask": torch.stack(
                [
                    torch.nn.functional.pad(mask, (0, self.max_sequence_length - len(mask)), value=0)
                    for mask in attention_mask
                ]
            ),
        }

        # Handle labels if present
        if self.label_column:
            pad_id = self.pad_token_id
            # Inspect first element to figure out if we have sequence-level or token-level labels
            first_label = batch[0][self.label_column]
            if isinstance(first_label, torch.Tensor):
                is_token_level = first_label.dim() > 1 and first_label.shape[1] == seq_lens[0]
            else:
                # For lists/arrays, check if it's 2D and matches sequence length
                is_token_level = (
                    isinstance(first_label, list | tuple)
                    and len(first_label) > 0
                    and isinstance(first_label[0], list | tuple)
                )

            if is_token_level:
                # Token-level labels: convert to tensor first, then truncate
                labels = [
                    (
                        item[self.label_column]
                        if isinstance(item[self.label_column], torch.Tensor)
                        else torch.tensor(item[self.label_column], dtype=torch.long)
                    )[: self.max_sequence_length]
                    for item in batch
                ]
                labels = torch.stack(
                    [
                        torch.nn.functional.pad(lb, (0, self.max_sequence_length - len(lb)), value=pad_id)
                        for lb in labels
                    ]
                )
            else:
                # Sequence-level labels: just convert to tensor
                labels = torch.tensor([item[self.label_column] for item in batch], dtype=torch.long)
            tokenized.update({"labels": labels})

        tokenized_with_labels = self.compute_labels(tokenized)
        tokenized_with_labels = {k: v for k, v in tokenized_with_labels.items() if k in self.batch_keys}
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

    Applies masking to create MLM training examples.

    Args:
         pad_token_id (int): Token ID used for padding sequences.
        mask_token_id (int): Token ID used for masking sequences.
        vocab_size (int): Size of the tokenizer vocabulary.
        text_column (str): Name of the column containing text data in the dataset.
            Defaults to "text".
        max_sequence_length (int | None): Maximum sequence length after tokenization.
            Defaults to 256.
        mlm_probability (float | None): Probability of masking tokens for MLM.
            Defaults to 0.3 (30% of tokens will be masked).
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        text_column: str = "text",
        max_sequence_length: int = 1024,
        mlm_probability: float = 0.3,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            text_column=text_column,
            max_sequence_length=max_sequence_length,
        )
        self.mlm_probability = mlm_probability

    def compute_labels(self, tokenized: dict[str, Any]) -> Any:
        """Compute the MLM labels for the given batch of tokenized inputs.

        Applies masking following the BERT MLM strategy for tokens sampled with MLM probability:
        - 80% of the time: replace token with [MASK]
        - 10% of the time: replace token with random token
        - 10% of the time: keep token unchanged

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed MLM labels for the batch.
        """
        input_ids = tokenized["input_ids"]
        # Create labels (clone of original input_ids)
        labels = input_ids.clone()

        # Create mask for tokens that can be masked, excluding special tokens and padding tokens
        maskable = self._get_valid_token_mask(input_ids, tokenized["attention_mask"])

        # Sample which tokens to mask using mlm_probability
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool() & maskable

        # Set labels for non-masked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long, device=input_ids.device)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time, keep unchanged (already in input_ids)
        tokenized["input_ids"] = input_ids
        tokenized["labels"] = labels
        return tokenized


class EnhancedMaskedLanguageModelingCollator(Collator):
    """Data collator for enhanced masked language modeling pretraining.

    Prepares pretokenized sequences for enhanced MLM. Unlike standard MLM, masking
    is not applied in the collator but is handled by the model itself.

    Expects input to be already tokenized (handled in the data module).
    """

    def compute_labels(self, tokenized: dict[str, Any]) -> dict[str, Any]:
        """Compute labels for enhanced masked language modeling.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed labels for the batch.
        """
        valid_mask = self._get_valid_token_mask(tokenized["input_ids"], tokenized["attention_mask"])
        labels = torch.where(valid_mask, tokenized["input_ids"], -100)
        tokenized["labels"] = labels
        return tokenized


class TokenClassificationCollator(Collator):
    """Data collator for token classification tasks.

    Handles formatting for token classification tasks like NER, POS tagging, etc.
    Pads pretokenized sequences and aligns token-level labels.

    Expects input to be already tokenized (handled in the data module).

    Args:
        pad_token_id (int): Token ID used for padding sequences.
        mask_token_id (int): Token ID used for masking sequences.
        vocab_size (int): Size of the tokenizer vocabulary.
        text_column (str): Name of the column containing text data. Defaults to "text".
        label_column (str): Name of the column containing label data. Defaults to "labels".
        max_sequence_length (int): Maximum sequence length for padding. Defaults to 1024.
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        text_column: str = "text",
        label_column: str = "labels",
        max_sequence_length: int = 1024,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            text_column=text_column,
            label_column=label_column,
            max_sequence_length=max_sequence_length,
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

    Handles formatting for sequence classification tasks like sentiment analysis,
    text classification, etc. Pads pretokenized sequences and preserves sequence-level labels.

    Args:
        pad_token_id (int): Token ID used for padding sequences.
        mask_token_id (int): Token ID used for masking sequences.
        vocab_size (int): Size of the tokenizer vocabulary.
        text_column (str): Name of the column containing text data. Defaults to "text".
        label_column (str): Name of the column containing label data. Defaults to "label".
        max_sequence_length (int): Maximum sequence length for padding. Defaults to 1024.
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        text_column: str = "text",
        label_column: str = "label",
        max_sequence_length: int = 1024,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            text_column=text_column,
            label_column=label_column,
            max_sequence_length=max_sequence_length,
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

    Handles formatting for question answering tasks like SQuAD. Pads pretokenized
    question-context pairs and preserves start/end positions for answer span prediction.

    Args:
        pad_token_id (int): Token ID used for padding sequences.
        mask_token_id (int): Token ID used for masking sequences.
        vocab_size (int): Size of the tokenizer vocabulary.
        text_column (str): Name of the column containing question data. Defaults to "question".
        label_column (str): Name of the column containing answer data. Defaults to "answers".
        context_column (str): Name of the column containing context data. Defaults to "context".
        max_sequence_length (int): Maximum sequence length for padding. Defaults to 1024.
        doc_stride (int): Stride for sliding window when context is too long. Defaults to 128.
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        text_column: str = "question",  # Override default for QA
        label_column: str = "answers",  # Override default for QA
        context_column: str = "context",
        max_sequence_length: int = 1024,
        doc_stride: int = 128,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            text_column=text_column,  # question column
            label_column=label_column,  # answers column
            max_sequence_length=max_sequence_length,
        )
        self.context_column = context_column
        self.doc_stride = doc_stride

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Process a batch of pretokenized QA examples.

        Expects input to already be tokenized with question-context pairs.
        """
        # Use base class to handle pretokenized input and padding
        tokenized = super().__call__(batch)

        # Add answers if available
        if self.label_column and all(self.label_column in item for item in batch):
            tokenized.update({"answers": [item[self.label_column] for item in batch]})

        # Compute start/end positions
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


class MaskedDiffusionCollator(Collator):
    """Diffusion-based MLM collator that samples random masking probabilities per batch.

    Randomly masks tokens using a sampled masking probability from a diffusion schedule.
    Expects input to be already tokenized (handled in the data module).

    Parameters
    ----------
    pad_token_id: int
        Token ID used for padding sequences.
    mask_token_id: int
        Token ID to use for masking.
    vocab_size: int
        Size of the tokenizer vocabulary.
    text_column: str
        The name of the column containing text data.
    max_sequence_length: int
        The maximum length for padding.
    num_steps: int
        Number of diffusion steps.
    sampling_eps: float
        Minimum timestep for sampling, controls minimum masking probability.
    noise_eps: float
        Epsilon for the noise schedule, controls maximum masking probability (1 - noise_eps).
    min_masked: int | None
        Minimum number of masked tokens per sequence. If None, not enforced.
    max_masked: int | None
        Maximum number of masked tokens per sequence. If None, not enforced.
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
        text_column: str = "text",
        max_sequence_length: int = 1024,
        num_steps: int = 1000,
        sampling_eps: float = 0.1,
        noise_eps: float = 1e-3,
        min_masked: int | None = None,
        max_masked: int | None = None,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            text_column=text_column,
            max_sequence_length=max_sequence_length,
        )
        self.noise = LogLinearNoise(eps=noise_eps)
        self.num_steps = num_steps
        self.sampling_eps = sampling_eps
        self.min_masked = min_masked
        self.max_masked = max_masked

    def _sample_t(self, batch_size: int, sampling_eps: float = 1e-3) -> torch.FloatTensor:
        """Sample timesteps for diffusion training with antithetic sampling.

        Args:
            batch_size: Number of timesteps to sample for this batch

        Returns:
            timesteps: Sampled timesteps in [sampling_eps, 1] with shape [batch_size]
        """
        # Antithetic sampling: stratify [0,1] into n equal bins, sample one point per bin
        random_offsets = torch.rand(batch_size)
        grid_positions = torch.arange(batch_size) / batch_size
        uniform_samples = (random_offsets / batch_size + grid_positions) % 1
        # Scale to timestep range
        timesteps = (1 - sampling_eps) * uniform_samples + sampling_eps
        return timesteps

    def _apply_noise(
        self,
        input_ids: "torch.Tensor",
        noise_prob: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Apply noise to a given input ID tensor.

        Args:
          input_ids (torch.Tensor, shape [batch_size, seq_len]): input IDs to apply noise to.
          noise_prob (torch.Tensor, shape [batch_size, 1]): Level of noise to apply to each sequence.
          attention_mask (torch.Tensor | None, shape [batch_size, seq_len]): Attention mask indicating
              valid tokens. Can be binary (0/1) or packed (sequence indices with -1 for padding).
              If None, all positions are considered valid.

        Returns:
            torch.Tensor, shape [batch_size, seq_len]: Input IDs changed to mask tokens at random positions with
                probability given by noise_level.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Handle different attention mask formats
        if attention_mask is None:
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            # Detect packed format
            valid_mask = attention_mask >= 0 if is_packed_batch(attention_mask) else attention_mask.bool()

        # Initial probabilistic masking (only on valid positions)
        rand_vals = torch.rand(batch_size, seq_len, device=device)
        mask = (rand_vals < noise_prob) & valid_mask

        # Mask additional tokens if needed
        if self.min_masked is not None:
            masked_count = mask.sum(dim=1, keepdim=True)
            need = (self.min_masked - masked_count).clamp(min=0)
            max_need = need.max().item()

            if max_need > 0:
                # Assign random priorities to unmasked valid positions (masked/padding get -inf)
                priority = torch.rand(batch_size, seq_len, device=device)
                priority = priority.masked_fill(mask | ~valid_mask, -float("inf"))

                # Select top k positions per sequence by priority
                _, top_indices = priority.topk(max_need, dim=1)

                # Only mask the first k positions for each sequence
                col_idx = torch.arange(max_need, device=device).unsqueeze(0)
                additional_mask = col_idx < need
                mask.scatter_(1, top_indices, additional_mask)

        # Unmask excess tokens if needed
        if self.max_masked is not None:
            masked_count = mask.sum(dim=1, keepdim=True)
            excess = (masked_count - self.max_masked).clamp(min=0)
            max_excess = excess.max().item()

            if max_excess > 0:
                # Assign random priorities to masked positions (unmasked get -inf)
                priority = torch.rand(batch_size, seq_len, device=device)
                priority = priority.masked_fill(~mask, -float("inf"))

                # Select top k masked positions per sequence by priority
                _, top_indices = priority.topk(max_excess, dim=1)

                # Only unmask the first k positions for each sequence
                col_idx = torch.arange(max_excess, device=device).unsqueeze(0)
                to_unmask = col_idx < excess
                unmask_targets = torch.zeros_like(mask)
                unmask_targets.scatter_(1, top_indices, to_unmask)
                mask = mask & ~unmask_targets

        return torch.where(mask, self.mask_token_id, input_ids)

    def compute_labels(self, tokenized: dict[str, Any]) -> Any:
        """Compute the denoising MLM labels for the given batch of tokenized inputs.

        Args:
            tokenized (dict[str, Any]): The tokenized inputs for the batch.

        Returns:
            dict[str, Any]: The computed MLM labels for the batch.
        """
        tokenized["labels"] = tokenized["input_ids"].clone()
        # At training time, we apply random timestep modification
        t = self._sample_t(tokenized["input_ids"].shape[0], sampling_eps=self.sampling_eps)
        dalpha_t, alpha_t = self.noise(t)
        # Masking probability = 1 - alpha_t (fraction of tokens that should be MASK)
        noise_prob = (1 - alpha_t).unsqueeze(-1)
        attention_mask = tokenized.get("attention_mask")
        noised_input_ids = self._apply_noise(tokenized["input_ids"], noise_prob, attention_mask)
        tokenized["input_ids"] = noised_input_ids
        return tokenized


def get_collator_cls(
    objective: Literal[
        "mlm", "enhanced_mlm", "classification", "token_classification", "question_answering", "diffusion"
    ],
) -> type[Collator]:
    """Get the appropriate data collator for the given objective.

    Args:
        objective (str): The training objective. Available options:
            "mlm", "enhanced_mlm", "classification", "token_classification", "question_answering", "diffusion".

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
        case "diffusion":
            return MaskedDiffusionCollator
        case "classification":
            return SequenceClassificationCollator
        case "token_classification":
            return TokenClassificationCollator
        case "question_answering":
            return QuestionAnsweringCollator
        case _:
            raise ValueError(f"Unknown objective: {objective}")
