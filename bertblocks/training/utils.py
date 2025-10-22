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


def predict_denoising(
    model: "torch.nn.Module",
    batch: dict[str, "torch.Tensor"],
    num_steps: int = 10,
    top_k: int = 0,
    top_p: float = 1.0,
    pad_token_id: int = 0,
    mask_token_id: int = 0,
    sep_token_id: int | None = None,
) -> "torch.Tensor":
    """Predicts text for a batch using a denoising objective.

    Args:
        model (torch.nn.Module): Model to use. Must return logits.
        batch (dict[str, Any]): Input batch, must contain keys `input_ids`, `attention_mask`
        num_steps (int): Number of denoising steps to perform.
        top_k (int): Number of top-k tokens to sample per mask during denoising. Defaults to 0 (uses top_p only).
        top_p (float): Tokens with cumulative probability >= top_p to keep per mask during denoising. Defaults to 1.0.
        pad_token_id (int, optional): Padding token id. Defaults to 0.
        mask_token_id (int, optional): Mask token id. Defaults to 0.
        sep_token_id (int, optional): Separator token id. Defaults to None.

    Returns:
        torch.Tensor: Predicted text for a batch using denoising.
    """
    B, S = batch["input_ids"].shape
    device = batch["input_ids"].device

    for mask_prob in torch.linspace(0.001, 1.0, num_steps):
        # Forward pass: predict masked tokens
        with torch.no_grad():
            predictions = model(**batch).logits  # shape: [B, S, V]

        # Find all masked positions across the batch
        prefix_length = batch["attention_mask"].sum(dim=-1) - 1
        # Set padding to masked to allow prediction beyond prompt
        batch["input_ids"] = torch.where(batch["input_ids"] == pad_token_id, mask_token_id, batch["input_ids"])
        if sep_token_id is not None:
            # If using a sep token, move it to the end of the context window.
            batch["input_ids"] = torch.where(batch["input_ids"] == sep_token_id, mask_token_id, batch["input_ids"])
            batch["input_ids"][:, -1] = sep_token_id

        mask_positions = (batch["input_ids"] == mask_token_id) & (
            torch.arange(S, device=device).repeat(B, 1) >= prefix_length.unsqueeze(-1)
        )

        if mask_positions.any():
            # Get logits for all masked positions
            masked_logits = predictions[mask_positions]  # shape: [num_masked, V]

            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(masked_logits, top_k=top_k, top_p=top_p)
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # Sample tokens for all masked positions at once
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)  # shape: [num_masked,]

            # Update the batch with sampled tokens
            batch["input_ids"][mask_positions] = sampled_tokens

        # Re-mask a portion of non-prefix tokens for next iteration (except last step)
        if mask_prob > 0:
            # Create random mask for non-prefix positions
            non_prefix_mask = torch.rand((B, S), device=device)
            non_prefix_mask[~mask_positions] = 0
            non_prefix_mask = non_prefix_mask >= mask_prob
            # Apply mask
            batch["input_ids"][non_prefix_mask.bool()] = mask_token_id

    return batch["input_ids"]


__all__ = ["chunk_examples", "top_k_top_p_filtering", "predict_denoising"]
