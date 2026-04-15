import math

import torch
from torch import Tensor, nn
from transformers.utils import is_flash_attn_2_available

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
        attention_mask: Tensor | None = None,
    ) -> "Tensor":
        if attention_mask is not None and cu_seqlens is not None:
            raise ValueError("flash attention rotary embeddings do not support unpadded inputs with attention masks")

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
        return dx, None, None, None, None, None, None

    def apply_rotary_setup_context(ctx, inputs, output):  # type: ignore
        _, cos, sin, interleaved, cu_seqlens, max_seqlen, _ = inputs
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
            return torch.stack((-x2, x1), dim=-1).flatten(-2)  # ... d t -> ... (d t)

    def apply_rotary(
        x: "Tensor",
        cos: "Tensor",
        sin: "Tensor",
        interleaved: bool = False,
        cu_seqlens: "Tensor | None" = None,
        max_seqlen: int | None = None,
        attention_mask: "Tensor | None" = None,
    ) -> "Tensor":
        """Native torch implementation of apply_rotary, for use if flash attention is not available.

        Args:
            x (Tensor, shape [batch_size, seq_len, num_heads, head_dim]): tensor to apply rotary encoding to.
            cos (Tensor, shape [seq_len, head_dim/2]): Cosine rotary tensor.
            sin (Tensor, shape [seq_len, head_dim/2]): Sine rotary tensor.
            cu_seqlens (Tensor, shape [batch_size + 1,], optional): tensor of cumulative sequence lengths in unpadded
                data.
            max_seqlen (int, optional): maximum sequence length in batch.
            attention_mask (Tensor, optional): used as a non-None sentinel to select the padded code path (arange
                position indices). The actual values are not read; any tensor shape is accepted.

        Returns:
            Tensor, shape [batch_size, seq_len, num_heads, head_dim]: input tensor with rotary encoding applied.
        """
        if cu_seqlens is not None:
            total_len = cu_seqlens[-1]
            position_idcs = torch.arange(total_len, device=x.device)
            seq_idx = torch.searchsorted(cu_seqlens[1:], position_idcs, right=True)
            position_idcs = position_idcs - cu_seqlens[:-1][seq_idx]
        elif attention_mask is not None:
            position_idcs = torch.arange(x.shape[1], device=x.device)[None].expand(x.shape[0], -1)
        else:
            raise ValueError(
                "Neither cu_seqlens nor attention_mask provided for apply_rotary. "
                "One of these must be provided to determine position indices."
            )

        cos = cos[position_idcs]
        sin = sin[position_idcs]

        dim = cos.shape[-1] * 2
        if not interleaved:
            cos = torch.cat([cos, cos], dim=-1).unsqueeze(-2)  # ... d -> ... 1 (2 d)
            sin = torch.cat([sin, sin], dim=-1).unsqueeze(-2)  # ... d -> ... 1 (2 d)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)  # ... d -> ... 1 (d 2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)  # ... d -> ... 1 (d 2)
        return torch.cat([x[..., :dim] * cos + rotate_half(x[..., :dim], interleaved) * sin, x[..., dim:]], dim=-1)


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
            total_seq_len = cu_seqlens[-1]
            seq_ids = torch.zeros(total_seq_len, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
            seq_ids[cu_seqlens[1:-1]] = 1
            seq_ids = seq_ids.cumsum(dim=0)
            pos_ids = (
                torch.arange(total_seq_len, device=cu_seqlens.device, dtype=cu_seqlens.dtype) - cu_seqlens[seq_ids]
            )
            return x + self.sin[0, pos_ids, :]
        else:
            # Padded sequence format
            seq_len = x.size(1) if x.dim() == 3 else x.size(0)
            return x + self.sin[0, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encodings.

    Attributes:
        embd (nn.Embedding): The embedding layer encoding position.

    Args:
            dim (int): Hidden size of the model.
            max_seq_len (int): Maximum sequence length for the model.
    """

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.embd = nn.Embedding(max_seq_len, dim)

    @staticmethod
    def _get_position_ids_from_cu_seqlens(cu_seqlens: "torch.Tensor") -> "torch.Tensor":
        total_seq_len = cu_seqlens[-1]
        # Tensor with sequence ids for each position
        seq_ids = torch.zeros(total_seq_len, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        seq_ids[cu_seqlens[1:-1]] = 1
        seq_ids = seq_ids.cumsum(dim=0)
        # Create position indices and subtract by corresponding sequence length to get per-sequence runs
        pos_ids = torch.arange(total_seq_len, device=cu_seqlens.device, dtype=cu_seqlens.dtype) - cu_seqlens[seq_ids]
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
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        return x + self.embd(pos_ids)


class AlibiPositionalEncoding(nn.Module):
    """Alibi Positional Encodings.

    Attributes:
        slopes (torch.Tensor): The alibi slope tensor indicating degree of positional bias for each head.

    Args:
        num_heads (int): Number of attention heads.

    """

    slopes: torch.Tensor

    def __init__(self, num_heads: int):
        super().__init__()
        slopes = self.get_slopes(num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def get_slopes(num_heads: int) -> "torch.Tensor":
        """Construct ALiBi slopes."""

        def __inner__(num_heads: int) -> list:
            base = 2 ** (-(2 ** -(math.log2(num_heads) - 3)))
            return [base * base**i for i in range(num_heads)]

        if math.log2(num_heads).is_integer():
            out = __inner__(num_heads)
        else:
            # If not a power of 2, find the closest one
            po2 = 2 ** math.floor(math.log2(num_heads))
            # Fill remaining to actual head size with doubled base value and step size
            out = __inner__(po2) + __inner__(2 * po2)[0::2][: num_heads - po2]

        return torch.Tensor(out)

    def forward(self, attention_mask: "torch.Tensor") -> "torch.Tensor":
        """Add AliBi biases to a given attention mask.

        Args:
            attention_mask (torch.Tensor, shape [batch_size, num_heads, seq_len, seq_len]): The attention mask.

        Returns:
            torch.Tensor: The attention mask after adding alibi biases. Same shape as input.
        """
        seqlen = attention_mask.shape[2]
        pos = torch.arange(seqlen, device=attention_mask.device)
        pos_diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        alibi_bias = -1.0 * torch.einsum("h, i j -> h i j", self.slopes, pos_diff).unsqueeze(0)

        if attention_mask.dtype == torch.bool:
            attention_bias = torch.zeros_like(attention_mask, device=alibi_bias.device, dtype=alibi_bias.dtype)
            attention_bias = attention_bias.masked_fill(~attention_mask, -float("inf"))
        else:
            attention_bias = attention_mask

        attention_mask = attention_bias + alibi_bias
        return attention_mask


class RotaryPositionalEncoding(nn.Module):
    """Implementation of rotary positional encodings.

    Args:
        rope_dim (int): dimensionality of positional encoding. Equal to head_dim for full RoPE.
        head_dim (int): dimensionality of attention heads.
        base (float, optional): frequency base for positional encodings. Defaults to 10_000.0
        interleaved (bool, optional): indicates whether to rotate pairs of even and odd dimensions (True, GPT-J style)
            instead of 1st half and 2nd half (False, GPT-NeoX style). Defaults to False.
        max_seq_len (int, optional): initial maximum sequence length for the cos/sin cache. Defaults to 512.

    References:
        - "RoFormer: Enhanced Transformer with Rotary Position Embedding" (https://arxiv.org/abs/2104.09864)
        - "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" (https://arxiv.org/abs/2204.06745)
        - "Round and Round We Go! What makes Rotary Positional Encodings useful?" (https://arxiv.org/abs/2410.06205)
    """

    def __init__(
        self,
        rope_dim: int,
        head_dim: int,
        base: float = 10_000.0,
        interleaved: bool = False,
        max_seq_len: int = 512,
    ):
        super().__init__()

        if rope_dim < 2 or rope_dim > head_dim:
            raise ValueError(f"partial_rope_dim must be in [2, head_dim] (was: {rope_dim})")

        self.interleaved = interleaved

        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2) / rope_dim))

        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)
        cos, sin = torch.cos(freqs), torch.sin(freqs)

        self.register_buffer("_cos_cached", cos, persistent=True)
        self.register_buffer("_sin_cached", sin, persistent=True)

    def forward(
        self,
        q: "Tensor",
        k: "Tensor",
        cu_seqlens: "Tensor | None" = None,
        max_seqlen: int | None = None,
        attention_mask: "Tensor | None" = None,
    ) -> "tuple[Tensor, Tensor]":
        """Apply rotary positional encoding to query and key tensors.

        Args:
            q (Tensor, shape [batch, seqlen, num_heads, head_dim] if padded or
                shape [total_seqlen, num_heads, head_dim] if unpadded): Query tensor.
            k (Tensor, shape [batch, seqlen, num_kv_heads, head_dim] if padded or
                shape [total_seqlen, num_kv_heads, head_dim] if unpadded): Key tensor.
            cu_seqlens (Tensor, shape [batch_size + 1,], optional): Cumulative sequence lengths if unpadded.
                Defaults to None.
            max_seqlen (int, optional): Maximum sequence length in batch. Defaults to None.
            attention_mask (Tensor, shape [batch_size, seq_len], optional): Attention mask for padded data.
                Defaults to None.

        Returns:
            tuple[Tensor, Tensor]: (q, k) with rotary position encoding applied, same shapes as input.
        """
        q, k = self._apply_rope(q, k, cu_seqlens, max_seqlen, attention_mask)

        return q, k

    def _apply_rope(self, q, k, cu_seqlens, max_seqlen, attention_mask):  # type: ignore
        q = apply_rotary(
            q,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
        )
        k = apply_rotary(
            k,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
        )

        return q, k


__all__ = [
    "AlibiPositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEncoding",
    "SinusoidalPositionalEncoding",
]
