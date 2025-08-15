import math

import torch
from torch import nn

from polybert.modeling.config import PolyBertConfig


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


def get_alibi_slopes(nheads: int, device: "torch.device | str | None" = None) -> "torch.Tensor":
    """Construct ALiBi slopes."""

    def __inner__(nheads: int) -> list:
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return torch.Tensor(__inner__(nheads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            __inner__(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )


class RotaryEmbedding(nn.Module):
    """Implementation of Learned Positional Encodings.

    Args:
        config (PolyBertConfig): Mode config.

    """

    inv_freq: torch.Tensor

    def __init__(self, config: PolyBertConfig):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            self._inv_freq(config.pos_kwargs.get("theta", 10_000), config.hidden_size // config.num_attention_heads),
            persistent=False,
        )
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def _inv_freq(theta: float, dim: int) -> "torch.Tensor":
        """Initialize the inverse frequency tensor."""
        return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

    @staticmethod
    def _rotate_half(x: "torch.Tensor") -> "torch.Tensor":
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @torch.no_grad()
    def _update_rope_cache(self, x: "torch.Tensor", position_ids: "torch.Tensor") -> None:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        self.cos = cos.to(dtype=x.dtype)
        self.sin = sin.to(dtype=x.dtype)

    def forward(self, qkv: "torch.Tensor", unsqueeze_dim: int = 1) -> "torch.Tensor":
        """Apply Rotary Position Embedding to the qkv tensor.

        Args:
            qkv (`torch.Tensor`): The qkv tensor.
            unsqueeze_dim (int, optional): The dimension along which to unsqueeze. Defaults to 1.

        Returns:
            `tuple(torch.Tensor)` comprising of the qkv rotated using the Rotary Position Embedding.

        """
        cos = self.cos.unsqueeze(unsqueeze_dim)
        sin = self.sin.unsqueeze(unsqueeze_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return torch.cat([q, k, v], dim=-1)
