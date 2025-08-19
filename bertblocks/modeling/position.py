import math
from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    FlashRotaryEmbedding = object


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

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Add sinusoidal positional encoding to a given tensor.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, embedding_dim]): The tensor to add positional encoding to.

        Returns:
            torch.Tensor: The tensor after adding positional encoding. Shape [batch_size, seq_len, embedding_dim].

        """
        return x + self.sin[: x.size(1), :]


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
    def _get_position_ids(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
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
            x (torch.Tensor, shape [total_seq_len, embedding_dim]): The tensor to add positional encodings to.
            cu_seqlens (torch.Tensor, shape [batch_size + 1,]): Cumulative sequence lengths to infer positions from.

        Returns:
            torch.Tensor: The tensor after adding learned positional encodings.
                Shape [total_seq_len, embedding_dim].

        """
        cu_seqlens = (
            cu_seqlens
            if cu_seqlens is not None
            else torch.Tensor([0, x.shape[0]]).to(device=x.device, dtype=torch.int32)
        )
        print(x.shape, cu_seqlens)
        return x + self.embd(self._get_position_ids(cu_seqlens))


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


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    """Autograd for rotary positional encodings."""

    @staticmethod
    def forward(
        ctx: Any,
        qkv: "torch.Tensor",
        cos: "torch.Tensor",
        sin: "torch.Tensor",
        cu_seqlens: "torch.Tensor | None" = None,
        max_seqlen: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Forward pass."""
        # (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx: Any, do: Any) -> Any:
        """Backward pass."""
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    qkv: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    cu_seqlens: "torch.Tensor | None" = None,
    max_seqlen: int | None = None,
) -> "torch.Tensor":
    """Apply rotary positional encoding to unpadded qkv tensors.

    Arguments:
        qkv: (total_nnz, 3, nheads, headdim) - input tensor for packed QKV.
        cos: (seqlen_rotary, rotary_dim / 2)
        sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.

    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class RotaryEmbedding(FlashRotaryEmbedding):
    """The rotary position embeddings applied directly to unpadded sequences."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: int | None = None,
        device: "torch.device | None" = None,
        dtype: "torch.dtype | None" = None,
    ):
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: "torch.Tensor",
        cu_seqlens: "torch.Tensor",
        max_seqlen: int | None = None,
    ) -> "torch.Tensor":
        """Apply rotary embedding inplace to qkv."""
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        """Construct string representation."""
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"
