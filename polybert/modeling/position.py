import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Implementation of Sinusoidal Positional Encodings.

    Args:
        dim (int): Embedding dimension, usually set to embed_dim // num_heads.
        max_seq_len (int): Maximum sequence length for the model.
        base (float): The base used to compute frequencies.

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
            x (torch.Tensor, shape [batch_size, seq_len, embedding_dim]): The tensor to add positional encoding to.

        Returns:
            torch.Tensor: The tensor after adding positional encoding. Shape [batch_size, seq_len, embedding_dim].

        """
        return x + self.sin[: x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    """Implementation of Learned Positional Encodings.

    Args:
        dim (int): Hidden size of the model.
        max_seq_len (int): Maximum sequence length for the model.

    """

    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.embd = nn.Embedding(max_seq_len, dim)
        self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)), persistent=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add learned positional encodings to a given tensor.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, embedding_dim]): The tensor to add positional encodings to.

        Returns:
            torch.Tensor: The tensor after adding learned positional encodings.
                Shape [batch_size, seq_len, embedding_dim].

        """
        return x + self.embd(self.position_ids)


class RotaryPositionalEncoding(nn.Module):
    """Implementation of Rotary Positional Encodings.

    Args:
        dim (int): Embedding dimension, usually set to embed_dim // num_heads.
        max_seq_len (int): Maximum expected sequence length for the model, if exceeded
            the cached freqs will be recomputed.
        base (float): The base used to compute rotation angles.

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
        """Add RoPE positional encodings to a given tensor.

        Args:
            x (torch.Tensor, shape [seq_len, batch_size, num_heads, embedding_dim]): The tensor to add
                RoPE positional encodings to.

        Returns:
            torch.Tensor: The tensor after adding RoPE positional encodings.
                Shape [seq_len, batch_size, num_heads, embedding_dim].

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

    This class provides static methods for ALIBI computation and is designed
    to be used without instantiation.

    References:
        - "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (https://arxiv.org/abs/2108.12409)

    """

    @staticmethod
    def score_mod(
        score: "torch.Tensor",
        _b: "torch.Tensor",
        h: "torch.Tensor",
        q_idx: "torch.Tensor",
        kv_idx: "torch.Tensor",
        slopes: "torch.Tensor",
    ) -> "torch.Tensor":
        """FlexAttention score mod function that adds ALiBi bias.

        Args:
            score (torch.Tensor, shape [batch_size, num_heads, seq_len, seq_len]): Raw attention scores.
            _b (torch.Tensor): Batch index tensor (unused in this implementation).
            h (torch.Tensor, shape [num_heads]): Head index tensor.
            q_idx (torch.Tensor, shape [seq_len]): Query position indices.
            kv_idx (torch.Tensor, shape [seq_len]): Key-value position indices.
            slopes (torch.Tensor, shape [num_heads]): Head-specific slope values. Smaller slopes
                   for heads that focus on closer positions.

        Returns:
            torch.Tensor: Modified attention scores with ALiBi bias applied, same shape as input score.

        """
        scale = slopes[h]
        bias = (kv_idx - q_idx) * scale
        return score + bias


class RelativePositionalEncoding:
    """Add Relative Positional Encoding score modification.

    References:
        - "Self-Attention with Relative Position Representations" (https://arxiv.org/pdf/1803.02155)

    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def score_mod(
        score: "torch.Tensor",
        _b: "torch.Tensor",
        _h: "torch.Tensor",
        q_idx: "torch.Tensor",
        kv_idx: "torch.Tensor",
    ) -> "torch.Tensor":
        """FlexAttention score mod function that adds relative positional bias.

        Args:
            score (torch.Tensor, shape [batch_size, num_heads, seq_len, seq_len]): Raw attention scores.
            _b (torch.Tensor): Batch index tensor (unused).
            _h (torch.Tensor): Head index tensor (unused).
            q_idx (torch.Tensor, shape [seq_len]): Query position indices.
            kv_idx (torch.Tensor, shape [seq_len]): Key-value position indices.

        Returns:
            torch.Tensor: Modified attention scores with relative positional bias applied, same shape as input score.

        """
        return score + (q_idx - kv_idx)
