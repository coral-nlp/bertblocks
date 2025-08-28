import math

import torch
from torch import nn
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEncoding
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary
else:
    FlashRotaryEncoding = object
    apply_rotary = None


class SinusoidalPositionalEncoding(nn.Module):
    """Implementation of Sinusoidal Positional Encodings.

    References:
        - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)

    """

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        """Initialize sinusoidal positional encodings.

        Args:
            dim (int): Embedding dimension, usually set to embed_dim // num_heads.
            max_seq_len (int): Maximum sequence length for the model.
            base (float): The base used to compute frequencies.

        """
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

    def forward(self, x: "torch.Tensor", cu_seqlens: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Add sinusoidal positional encoding to a given tensor.

        Args:
            x (torch.Tensor): The tensor to add positional encoding to.
                - For unpadded: shape [total_seq_len, embedding_dim]
                - For padded: shape [batch_size, seq_len, embedding_dim]
            cu_seqlens (torch.Tensor, shape [batch_size + 1,], optional):
                Cumulative sequence lengths for unpadded sequences. If None, assumes padded format.

        Returns:
            torch.Tensor: The tensor after adding positional encoding, same shape as input.

        """
        if cu_seqlens is not None:
            # Unpadded sequence format - create position IDs for each sequence
            total_seq_len = cu_seqlens[-1].item()
            seq_ids = torch.zeros(total_seq_len, device=cu_seqlens.device, dtype=torch.long)
            seq_ids[cu_seqlens[1:-1]] = 1
            seq_ids = seq_ids.cumsum(dim=0)
            pos_ids = torch.arange(total_seq_len, device=cu_seqlens.device, dtype=torch.long) - cu_seqlens[seq_ids]
            return x + self.sin[0, pos_ids, :]
        else:
            # Padded sequence format
            seq_len = x.size(1) if x.dim() == 3 else x.size(0)
            return x + self.sin[0, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encodings."""

    def __init__(self, dim: int, max_seq_len: int):
        """Initialize learned positional encodings.

        Args:
            dim (int): Hidden size of the model.
            max_seq_len (int): Maximum sequence length for the model.

        """
        super().__init__()
        self.embd = nn.Embedding(max_seq_len, dim)

    @staticmethod
    def _get_position_ids_from_cu_seqlens(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
        total_seq_len = cu_seqlens[-1].item()
        # Tensor with sequence ids for each position
        seq_ids = torch.zeros(total_seq_len, device=cu_seqlens.device, dtype=torch.long)
        seq_ids[cu_seqlens[1:-1]] = 1
        seq_ids = seq_ids.cumsum(dim=0)
        # Create position indices and subtract by corresponding sequence length to get per-sequence runs
        pos_ids = torch.arange(total_seq_len, device=cu_seqlens.device, dtype=torch.long) - cu_seqlens[seq_ids]
        return pos_ids

    def forward(self, x: "torch.Tensor", cu_seqlens: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Add learned positional encodings to a given tensor.

        Args:
            x (torch.Tensor): The tensor to add positional encodings to.
                - For unpadded: shape [total_seq_len, embedding_dim]
                - For padded: shape [batch_size, seq_len, embedding_dim]
            cu_seqlens (torch.Tensor, shape [batch_size + 1,], optional):
                Cumulative sequence lengths for unpadded sequences. If None, assumes padded format.

        Returns:
            torch.Tensor: The tensor after adding learned positional encodings, same shape as input.

        """
        if cu_seqlens is not None:
            # Unpadded sequence format
            pos_ids = self._get_position_ids_from_cu_seqlens(cu_seqlens)
        else:
            # Padded sequence format - create standard position IDs
            batch_size, seq_len = x.shape[:2]
            pos_ids = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        return x + self.embd(pos_ids)


def get_alibi_slopes(
    nheads: int, device: "torch.device | str" = "cuda", dtype: "torch.dtype" = torch.float32
) -> "torch.Tensor":
    """Construct ALiBi slopes."""

    def __inner__(nheads: int) -> list:
        base = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        return [base * base**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        out = __inner__(nheads)
    else:
        # If not a power of 2, find the closest one
        po2 = 2 ** math.floor(math.log2(nheads))
        # Fill remaining to actual head size with doubled base value and step size
        out = __inner__(po2) + __inner__(2 * po2)[0::2][: nheads - po2]

    return torch.Tensor(out).to(device=device, dtype=dtype)


class UnpaddedRotaryEncoding(FlashRotaryEncoding):
    """Rotary position encodings implemented for unpadded sequences."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        device: "torch.device | None" = None,
    ):
        super().__init__(dim=dim, base=base, device=device, interleaved=False)

    def forward(
        self,
        x: "torch.Tensor",
        cu_seqlens: "torch.Tensor",
        max_seqlen: int | None = None,
    ) -> "torch.Tensor":
        """Apply rotary embedding to input tensor."""
        self._update_cos_sin_cache(max_seqlen, device=x.device, dtype=x.dtype)
        x = apply_rotary(
            x,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return x


class PaddedRotaryEncoding(FlashRotaryEncoding):
    """Rotary position encodings implemented for padded sequences; just wraps FA2."""

    pass
