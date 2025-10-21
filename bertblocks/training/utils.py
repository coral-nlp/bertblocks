import warnings
from typing import Any

import torch


def chunk_examples(
    examples: dict[str, list[Any]],
    column: str | None = None,
    split_char: str | None = None,
    split_len: int | None = None,
) -> dict[str, list[Any]]:
    """Chunk an input example text into paragraphs.

    Args:
        examples (dict[str, list[Any]]): Input batch.
        column (str, optional): Text column name to chunk. Defaults to 'text'.
        split_char (str, optional): Character to split examples at. Only one of `split_char` and `split_len`
            should be specified. Defaults to None.
        split_len (int, optional): Number of characters to split examples at. Only one of `split_char` and
            `split_len` should be specified. Defaults to None.

    Returns:
        dict[str, list[Any]]: Chunked input batch, reduced to `column` only.
    """
    if split_char is not None and split_len is not None:
        warnings.warn("Both `split_char` and `split_len` are given, falling back to use `split_len`.", stacklevel=2)
    if split_char is None and split_len is None:
        raise ValueError("Either `split_char` or `split_len` are required.")

    column = column if column is not None else "text"

    chunks = []
    if split_len is not None:
        for sentence in examples[column]:
            chunks += [sentence[i : i + split_len] for i in range(0, len(sentence), split_len)]
    else:
        for sentence in examples[column or "text"]:
            chunks += sentence.split(split_char)
    return {column: chunks}


def top_k_top_p_filtering(
    logits: "torch.Tensor",
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> "torch.Tensor":
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits (torch.Tensor, shape [batch size, vocabulary size]): Token logits distribution.
        top_k (int): Number of tokens with highest probability to keep.
            Defaults to 0.
        top_p (float): Tokens with cumulative probability >= top_p to keep (nucleus filtering).
            Defaults to 1.0.
        filter_value (float): Value to insert as logits for filtered tokens.
            Defaults to negative infinity (0 after softmax).
        min_tokens_to_keep (int): Minimum number of tokens to retain per batch example in the output.
            Defaults to 1.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


__all__ = ["chunk_examples", "top_k_top_p_filtering"]
