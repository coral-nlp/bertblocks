import functools
import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import _score_mod_signature


class SinusoidalPositionalEncoding(nn.Module):
    """Implementation of Sinusoidal Positional Encodings.

    Args:
        dim: int
            Embedding dimension, usually set to embed_dim // num_heads
        max_seq_len: int
            Maximum sequence length for the model.
        base: int
            The base used to compute frequencies.

    References:
        - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        sin = self._build_cache(dim, max_seq_len, base)
        self.register_buffer("sin", sin)

    @staticmethod
    def _build_cache(dim: int, max_seq_len: int, base: float = 10000.0) -> "torch.Tensor":
        pos = torch.arange(max_seq_len).unsqueeze(1)
        denom = torch.exp(torch.arange(0, dim, 2) * (-math.log(base) / dim))
        enc = torch.zeros(1, max_seq_len, dim)
        enc[0, :, 0::2] = torch.sin(pos * denom)
        enc[0, :, 1::2] = torch.cos(pos * denom)
        return enc

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add sinusoidal positional encoding to a given tensor.

        Args:
            x: Tensor, shape `[batch_size, seq_len, embedding_dim]`
                The tensor to add positional encoding to.

        Returns:
            Tensor, shape `[batch_size, seq_len, embedding_dim]`
            The tensor after adding positional encoding.

        """
        return x + self.sin[: x.size(1), :]


class RotaryPositionalEncoding(nn.Module):
    """Implementation of Rotary Positional Encodings.

    Args:
        dim: int
            Embedding dimension, usually set to embed_dim // num_heads
        max_seq_len: int
            Maximum expected sequence length for the model, if exceeded the cached freqs will be recomputed
        base: int
            The base used to compute rotation angles

    References:
        - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (https://arxiv.org/abs/2104.09864)

    """

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim, self.base = dim, base
        cos, sin = self._build_cache(dim, max_seq_len, base)  # Populate cos/sin cache
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def _build_cache(dim: int, max_seq_len: int, base: float) -> "tuple[torch.Tensor, torch.Tensor]":
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        seq = torch.arange(max_seq_len, dtype=theta.dtype)
        freqs = torch.outer(seq, theta)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[:, None, None, :]
        sin = emb.sin()[:, None, None, :]
        return cos, sin

    @staticmethod
    def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add RoPE to a given tensor.

        Args:
            x: Tensor, shape `[seq_len, batch_size, num_heads, embedding_dim]`
                The tensor to add RoPE to.

        Returns:
            Tensor, shape `[seq_len, batch_size, num_heads, embedding_dim]`
            The tensor after adding RoPE.

        """
        if x.shape[0] > self.cos.shape[0]:  # type: ignore
            # Rebuild cache if maximum sequence length is exceeded
            self.cos, self.sin = self._build_cache(self.dim, x.shape[0], self.base)
        return x * self.cos[: x.shape[0], :, :, :] + self._rotate_half(x) * self.sin[: x.shape[0], :, :, :]


class AlibiPositionalEncoding:
    """Add Attention with Linear Biases (ALIBI) score modification.

    ALIBI adds a linear bias to attention scores based on the distance between
    query and key positions, with head-specific slopes. This enables length
    extrapolation beyond training sequence lengths.

    Args:
        num_heads: int
            Number of attention heads.

    References:
        - "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (https://arxiv.org/abs/2108.12409)

    """

    def __init__(self, num_heads: int) -> None:
        self.slopes = torch.exp2(torch.arange(num_heads, dtype=torch.float32) * (-8.0 / num_heads))

    @staticmethod
    def _alibi(
        score: "torch.Tensor",
        _b: "torch.Tensor",
        h: "torch.Tensor",
        q_idx: "torch.Tensor",
        kv_idx: "torch.Tensor",
        slopes: "torch.Tensor",
    ) -> "torch.Tensor":
        """FlexAttention score mod function that adds ALiBi bias.

        Args:
            score: Raw attention scores of shape (batch_size, num_heads, seq_len, seq_len).
            _b: Batch index tensor (unused in this implementation).
            h: Head index tensor of shape (num_heads,).
            q_idx: Query position indices of shape (seq_len,).
            kv_idx: Key-value position indices of shape (seq_len,).
            slopes: Head-specific slope values of shape (num_heads,). Smaller slopes
                   for heads that focus on closer positions.

        Returns:
            Modified attention scores with ALiBi bias applied, same shape as input score.

        """
        scale = slopes[h]
        bias = (kv_idx - q_idx) * scale
        return score + bias

    def __call__(self) -> "_score_mod_signature":
        """Create an ALIBI score modification function.

        This method generates head-specific slopes and returns a partially applied
        ALIBI function that can be used with flex_attention.

        Args:
            num_heads: Number of attention heads. Must be positive.

        Returns:
            A partially applied ALIBI score modification function with the signature
            expected by flex_attention.

        """
        return functools.partial(AlibiPositionalEncoding._alibi, slopes=self.slopes)


class RelativePositionalEncoding:
    """Add Relative Positional Encoding score modification.

    References:
        - "Self-Attention with Relative Position Representations" (https://arxiv.org/pdf/1803.02155)

    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _relative_positional(
        score: "torch.Tensor",
        _b: "torch.Tensor",
        _h: "torch.Tensor",
        q_idx: "torch.Tensor",
        kv_idx: "torch.Tensor",
    ) -> "torch.Tensor":
        """FlexAttention score mod function that adds relative positional bias.

        Args:
            score: Raw attention scores of shape (batch_size, num_heads, seq_len, seq_len).
            _b: Batch index tensor (unused).
            _h: Head index tensor of shape (unused).
            q_idx: Query position indices of shape (seq_len,).
            kv_idx: Key-value position indices of shape (seq_len,).

        Returns:
            Modified attention scores with relative positional bias applied, same shape as input score.

        """
        return score + (q_idx - kv_idx)

    def __call__(self) -> "_score_mod_signature":
        """Create a relative positional score modification function.

        Returns:
            A score relative positional score modification function with the signature expected by flex_attention.

        """
        return RelativePositionalEncoding._relative_positional
