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

    def __init__(self, dim: int, max_seq_len: int):
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


def get_alibi_slopes(
    nheads: int, device: "torch.device | str" = "cuda", dtype: "torch.dtype" = torch.float32
) -> "torch.Tensor":
    """Construct ALiBi slopes."""

    def __inner__(nheads: int) -> list:
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        return [start * start**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        out = __inner__(nheads)
    else:
        # If not a power of 2, find closest one
        po2 = 2 ** math.floor(math.log2(nheads))
        # Fill remaining to actual head size with doubled base value and step size
        out = __inner__(po2) + __inner__(2 * po2)[0::2][: nheads - po2]

    return torch.Tensor(out).to(device=device, dtype=dtype)
