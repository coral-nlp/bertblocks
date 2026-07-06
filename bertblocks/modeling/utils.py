from typing import Any

import torch
from torch import nn


class LogLinearNoise(nn.Module):
    """Log Linear noise schedule.

    Returns alpha_t = 1 - (1-eps)*t and its derivative dalpha_t/dt = -(1-eps).
    The model internally converts alpha_t to sigma_t = -log(alpha_t) when needed.

    Implementation uses t' = (1 - eps) * t to avoid division/log by 0 as t approaches 1.

    Args:
        eps (float): Small value to avoid numerical issues at t=1. Defaults to 1e-3.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, t: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
        """Compute alpha_t and dalpha_t/dt for the given timestep.

        Args:
            t: Timestep in [0, 1]

        Returns:
            dalpha_t: Derivative of alpha with respect to t (constant = -(1-eps))
            alpha_t: Alpha value at timestep t (= 1 - (1-eps)*t)
        """
        alpha_t = 1 - (1 - self.eps) * t
        dalpha_t = -(1 - self.eps) * torch.ones_like(t)
        return dalpha_t, alpha_t


def flatten_and_segment(
    output: "Any", attention_mask: "torch.Tensor | None"
) -> "tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]":
    """Derive per-document segment structure from an encoder output.

    Produces a segment id for every *valid* token so that per-sequence pooling and bag-of-word
    targets can be computed identically for both the unpadded/packed representation (where the
    hidden states are already a flat ``[total_tokens, hidden]`` tensor delimited by ``cu_seqlens``)
    and the padded representation (``[batch, seq_len, hidden]`` delimited by ``attention_mask``).

    Args:
        output: Encoder output exposing ``cu_seqlens`` (unpadded/packed) or ``None`` (padded). When
            ``cu_seqlens`` is set it delimits per-document segments in the flat token dimension, even
            when multiple documents are packed into a row.
        attention_mask (torch.Tensor, shape [batch, seq_len], optional): Required for the padded
            path; ``1`` marks valid tokens.

    Returns:
        tuple:
            - ``segment_ids`` (torch.Tensor, shape [num_valid_tokens]): Document index of each valid
              token, aligned to the flat/selected token order.
            - ``lengths`` (torch.Tensor, shape [num_segments]): Number of valid tokens per document.
            - ``num_segments`` (int): Number of documents in the batch.
            - ``valid_mask`` (torch.Tensor | None): For the padded path, the ``[batch, seq_len]``
              boolean mask used to select flat tokens; ``None`` when tokens are already flat.
    """
    cu_seqlens = getattr(output, "cu_seqlens", None)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.long()
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        num_segments = int(lengths.numel())
        segment_ids = torch.repeat_interleave(torch.arange(num_segments, device=lengths.device), lengths)
        return segment_ids, lengths, num_segments, None

    if attention_mask is None:
        raise ValueError("attention_mask is required to derive segments for a padded batch")
    valid_mask = attention_mask.bool()
    num_segments = valid_mask.shape[0]
    lengths = valid_mask.sum(dim=1)
    segment_ids = torch.arange(num_segments, device=valid_mask.device).unsqueeze(1).expand_as(valid_mask)[valid_mask]
    return segment_ids, lengths, num_segments, valid_mask


def segment_mean(
    flat_hidden: "torch.Tensor", segment_ids: "torch.Tensor", lengths: "torch.Tensor", num_segments: int
) -> "torch.Tensor":
    """Mean-pool flat token embeddings into one vector per document.

    Args:
        flat_hidden (torch.Tensor, shape [num_valid_tokens, hidden]): Valid token embeddings in
            segment order.
        segment_ids (torch.Tensor, shape [num_valid_tokens]): Document index of each token.
        lengths (torch.Tensor, shape [num_segments]): Token count per document.
        num_segments (int): Number of documents.

    Returns:
        torch.Tensor, shape [num_segments, hidden]: Mean-pooled sequence embeddings.
    """
    sums = torch.zeros(num_segments, flat_hidden.shape[-1], dtype=flat_hidden.dtype, device=flat_hidden.device)
    sums = sums.index_add(0, segment_ids, flat_hidden)
    counts = lengths.clamp(min=1).unsqueeze(1).to(sums.dtype)
    return sums / counts


def top_k_top_p_filtering(
    logits: "torch.Tensor",
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> "torch.Tensor":
    """Filter a distribution of pred_probs using top-k and/or nucleus (top-p) filtering.

    Args:
        logits (torch.Tensor, shape [batch size, vocabulary size]): Token pred_probs distribution.
        top_k (int): Number of tokens with highest probability to keep.
            Defaults to 0.
        top_p (float): Tokens with cumulative probability >= top_p to keep (nucleus filtering).
            Defaults to 1.0.
        filter_value (float): Value to insert as pred_probs for filtered tokens.
            Defaults to negative infinity (0 after softmax).
        min_tokens_to_keep (int): Minimum number of tokens to retain per batch example in the output.
            Defaults to 1.

    Adapted from: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
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
