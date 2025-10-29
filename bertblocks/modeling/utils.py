import torch
from torch import nn


class LogLinearNoise(nn.Module):
    """Log Linear noise schedule.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def _rate_noise(self, t: "torch.Tensor") -> "torch.Tensor":
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def _total_noise(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1 - self.eps) * t)

    def forward(self, t: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
        """Compute the current total noise and rate of change of noise for the given timestep."""
        return self._total_noise(t), self._rate_noise(t)


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
