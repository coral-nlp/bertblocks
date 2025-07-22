from typing import Any

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer


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
        tokenizer: "PreTrainedTokenizer",
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
