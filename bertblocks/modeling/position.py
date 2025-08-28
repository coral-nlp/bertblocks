import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, device, dtype

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    # We have flash attention, so we can use their kernel with some modifications
    from flash_attn.ops.triton.rotary import apply_rotary as flash_apply_rotary

    @torch.library.triton_op("bertblocks::apply_rotary", mutates_args={})
    def apply_rotary(
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        interleaved: bool = False,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> "Tensor":
        return flash_apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=False,
        )

    def apply_rotary_backward(ctx, grad):  # type: ignore
        cos, sin, cu_seqlens = ctx.saved_tensors
        dx = flash_apply_rotary(
            grad,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=False,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None

    def apply_rotary_setup_context(ctx, inputs, output):  # type: ignore
        x, cos, sin, interleaved, cu_seqlens, max_seqlen = inputs
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.interleaved = interleaved
        ctx.max_seqlen = max_seqlen

    apply_rotary.register_autograd(apply_rotary_backward, setup_context=apply_rotary_setup_context)

else:
    # We don't have flash attention, so native torch it is

    def rotate_half(x, interleaved=False):  # type: ignore
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return rearrange(torch.stack((-x2, x1), dim=-1), "... d t -> ... (d t)", t=2)

    def apply_rotary(
        x: "Tensor",
        cos: "Tensor",
        sin: "Tensor",
        interleaved: bool = False,
        cu_seqlens: "Tensor | None" = None,
        max_seqlen: int | None = None,
    ) -> "Tensor":
        """Native torch implementation of apply_rotary, for use if flash attention is not available.

        Args:
            x (Tensor, shape [batch_size, seq_len, num_heads, head_dim]): tensor to apply rotary encoding to.
            cos (Tensor, shape [seq_len, head_dim/2]): Cosine rotary tensor.
            sin (Tensor, shape [seq_len, head_dim/2]): Sine rotary tensor.
            cu_seqlens (Tensor, shape [batch_size + 1,], optional): tensor of cumulative sequence lengths in unpadded
                data.
            max_seqlen (int, optional): maximum sequence length in batch.

        Returns:
            Tensor, shape [batch_size, seq_len, num_heads, head_dim]: input tensor with rotary encoding applied.
        """
        if cu_seqlens is None:
            # Padded code path
            dim = cos.shape[-1] * 2
            cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
            sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
            return torch.cat([x[..., :dim] * cos + rotate_half(x[..., :dim], interleaved) * sin, x[..., dim:]], dim=-1)
        else:
            # Unpadded path (we will likely have flash attention here?)
            raise NotImplementedError


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


class RotaryPositionalEncoding(nn.Module):
    """Implementation of rotary positional encodings.

    Args:
        dim (int): dimensionality of positional encoding. Should be head_dim // 2.
        base (float, optional): frequency base for positional encodings. Defaults to 10_000.0
        interleaved (bool, optional): indicates whether to rotate pairs of even and odd dimensions (True, GPT-J style)
            instead of 1st half and 2nd half (False, GPT-NeoX style). Defaults to False.
        device (torch.device, optional): device on which to allocate the frequency buffer. Defaults to None (cpu).

    References:
        - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (https://arxiv.org/abs/2104.09864)
        - "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" (https://arxiv.org/abs/2204.06745)
    """

    def __init__(
        self,
        dim: int,
        base: float | None = 10_000.0,
        interleaved: bool | None = False,
        max_seq_len: int = 512,
        device: "device | str" = "cuda",
    ):
        super().__init__()
        self.interleaved = interleaved
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("_inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = max_seq_len
        self._cos_cached = None
        self._sin_cached = None
        self._update_cos_sin_cache(max_seq_len, device, torch.float32)

    def _update_cos_sin_cache(
        self, seqlen: int, device: "device | str | None" = None, dtype: "dtype | None" = None
    ) -> None:
        """Recompute the sin/cos cache if the device or maximum sequence length changed.

        Args:
            seqlen (int): maximum sequence length to update the buffers to.
            device (torch.device, optional): device on which to allocate the frequency buffer. Defaults to None.
            dtype (torch.dtype, optional): type with which to allocate the frequency buffer. Defaults to None.
        """
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self._inv_freq)
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(
        self,
        qkv: "Tensor",
        num_heads: int,
        head_dim: int,
        cu_seqlens: "Tensor | None" = None,
        max_seqlen: int | None = None,
    ) -> "Tensor | tuple[Tensor, Tensor]":
        """Apply rotary positional encoding to qkv.

        Args:
            qkv (Tensor, shape [batch, seqlen, 3 * num_heads * head_dim] if padded or
                shape [total_seqlen, 3 * num_heads * head_dim] if unpadded): combined query/key/value tensor.
            number
            cu_seqlens (Tensor, shape [total_seq_len + 1,], optional): Cumulative sequence lengths if qkv is unpadded.
                Defaults to None.
            max_seqlen (int, optional): Maximum sequence length in batch. Defaults to None.

        Returns:
            Tensor, same shape as qkv; qkv with rotary position encoding applied.
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        if cu_seqlens is not None:  # Unpadded
            q, k, v = rearrange(qkv, "s (t h d) -> t s h d", t=3, h=num_heads, d=head_dim)
        else:
            batch_size, seq_len, _ = qkv.shape
            q, k, v = rearrange(qkv, "b s (t h d) -> t b s h d", b=batch_size, s=seq_len, t=3, h=num_heads, d=head_dim)
        q = apply_rotary(
            q,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        k = apply_rotary(
            k,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if cu_seqlens is not None:
            qkv = rearrange(torch.stack([q, k, v], 0), "t s h d -> s (t h d)", t=3, h=num_heads, d=head_dim)
        else:
            batch_size, seq_len, _ = qkv.shape
            qkv = rearrange(
                torch.stack([q, k, v], 0),
                "t b s h d -> b s (t h d)",
                b=batch_size,
                s=seq_len,
                t=3,
                h=num_heads,
                d=head_dim,
            )
        return qkv


__all__ = ["SinusoidalPositionalEncoding", "LearnedPositionalEncoding", "get_alibi_slopes", "RotaryPositionalEncoding"]
