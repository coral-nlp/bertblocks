"""Attention backend implementations with unified interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from bertblocks import BertBlocksConfig

if TYPE_CHECKING:
    from torch import Tensor

import torch
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func


class AttentionBackend(ABC):
    """Abstract base class for attention backends."""

    def forward_unpadded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with unpadded sequences.

        Args:
            q (Tensor, shape [total_seq_len, num_heads, head_dim]): Query tensor.
            k (Tensor, shape [total_seq_len, num_kv_heads, head_dim]): Key tensor.
            v (Tensor, shape [total_seq_len, num_kv_heads, head_dim]): Value tensor.
            cu_seqlens (Tensor, shape [batch_size + 1]): Cumulative sequence lengths.
            max_seq_len (int): Maximum sequence length in batch.
            alibi_slopes (Tensor, optional): ALiBi slopes for positional bias.
            local_attention (tuple[int, int]): Local attention window size.
            dropout_p (float): Dropout probability.
            deterministic (bool): Whether to use deterministic attention.

        Returns:
            tuple[Tensor, Tensor | None]: Output tensor [total_seq_len, num_heads * head_dim] and optional attention
                weights.
        """
        return self._forward_unpadded(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
            alibi_slopes=alibi_slopes,
            local_attention=local_attention,
            dropout_p=dropout_p,
            deterministic=deterministic,
        )

    def forward_padded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        attention_mask: "Tensor",
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with padded sequences.

        Args:
            q (Tensor, shape [batch_size, seq_len, num_heads, head_dim]): Query tensor.
            k (Tensor, shape [batch_size, seq_len, num_kv_heads, head_dim]): Key tensor.
            v (Tensor, shape [batch_size, seq_len, num_kv_heads, head_dim]): Value tensor.
            attention_mask (Tensor): Attention mask.
            dropout_p (float): Dropout probability.
            deterministic (bool): Whether to use deterministic attention.

        Returns:
            tuple[Tensor, Tensor | None]: Output tensor [batch_size, seq_len, num_heads * head_dim] and optional
                attention weights.
        """
        return self._forward_padded(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            dropout_p=dropout_p,
            deterministic=deterministic,
        )

    @abstractmethod
    def _forward_unpadded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """Implement the unpadded forward pass."""
        pass

    @abstractmethod
    def _forward_padded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        attention_mask: "Tensor",
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """Implement the padded forward pass."""
        pass


class FlashBackend(AttentionBackend):
    """Flash Attention 2 backend."""

    def _forward_unpadded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor]":
        """Flash attention forward pass without padding."""
        orig_dtype = q.dtype
        needs_cast = orig_dtype not in (torch.float16, torch.bfloat16)

        if alibi_slopes is not None and alibi_slopes.dtype != torch.float32:
            alibi_slopes = alibi_slopes.to(torch.float32)
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)

        x, _, w = flash_attn_varlen_func(
            q.to(torch.bfloat16) if needs_cast else q,
            k.to(torch.bfloat16) if needs_cast else k,
            v.to(torch.bfloat16) if needs_cast else v,
            cu_seqlens,
            cu_seqlens,
            max_seq_len,
            max_seq_len,
            dropout_p=dropout_p,
            causal=False,
            softcap=0.0,
            window_size=local_attention,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )

        if needs_cast:
            x = x.to(orig_dtype)
            w = w.to(orig_dtype) if w is not None else None
        x = x.flatten(-2)  # s h d -> s (h d)
        return x, w

    def _forward_padded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        attention_mask: "Tensor",
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """FlashAttention backend does not support padded sequences."""
        raise NotImplementedError


class SDPABackend(AttentionBackend):
    """PyTorch SDPA backend - works efficiently with padded sequences."""

    def _forward_padded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        attention_mask: "Tensor",
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """SDPA forward pass with padded sequences."""
        num_heads = q.shape[2]
        num_kv_heads = k.shape[2]

        # Transpose to [b, h, s, d] for SDPA
        q = q.transpose(1, 2)  # b s h d -> b h s d
        k = k.transpose(1, 2)  # b s h d -> b h s d
        v = v.transpose(1, 2)  # b s h d -> b h s d

        # Expand K/V heads if using GQA
        if num_kv_heads < num_heads:
            num_kv_groups = num_heads // num_kv_heads
            k = k.repeat_interleave(num_kv_groups, dim=1)  # b h s d -> b (h g) s d
            v = v.repeat_interleave(num_kv_groups, dim=1)  # b h s d -> b (h g) s d

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        output = output.transpose(1, 2).flatten(-2)  # b h s d -> b s (h d)
        return output, None

    def _forward_unpadded(self, *args, **kwargs):  # type: ignore
        """SDPA backend does not support unpadded sequences."""
        raise NotImplementedError


class EagerBackend(AttentionBackend):
    """Native PyTorch backend."""

    def _forward_padded(
        self,
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        attention_mask: "Tensor",
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Eager attention forward pass with padded sequences."""
        num_heads = q.shape[2]
        num_kv_heads = k.shape[2]
        head_dim = q.shape[3]

        # Transpose to [b, h, s, d] for attention computation
        q = q.transpose(1, 2)  # b s h d -> b h s d
        k = k.transpose(1, 2)  # b s h d -> b h s d
        v = v.transpose(1, 2)  # b s h d -> b h s d

        # Expand K/V heads if using GQA
        if num_kv_heads < num_heads:
            num_kv_groups = num_heads // num_kv_heads
            k = k.repeat_interleave(num_kv_groups, dim=1)  # b h s d -> b (h g) s d
            v = v.repeat_interleave(num_kv_groups, dim=1)  # b h s d -> b (h g) s d

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * (head_dim**-0.5)

        if attention_mask.dtype == torch.bool:
            # Regular boolean mask
            scores = scores.masked_fill(~attention_mask, -float("inf"))
        else:
            # Float mask (includes ALiBi bias)
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)

        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, 0.0)

        if dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

        output = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        output = output.transpose(1, 2).flatten(-2)  # b h s d -> b s (h d)
        return output, None

    def _forward_unpadded(self, *args, **kwargs):  # type: ignore
        """Eager backend does not support unpadded sequences."""
        raise NotImplementedError


def get_attention(config: "BertBlocksConfig") -> "AttentionBackend":
    """Get the Attention backend specified in the configuration.

    This factory function returns the appropriate attention backend based on
    the configuration.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters.

    Returns:
        An attention backend module.

    Raises:
        ValueError: If the specified attention backend is not supported.

    Supported backends:

        - `flash_attention_2`: Flash Attention
        - `sdpa`: torch scaled dot product attention
        - `eager`: native torch attention

    """
    match config._attn_implementation:
        case "flash_attention_2":
            return FlashBackend()
        case "sdpa":
            return SDPABackend()
        case "eager":
            return EagerBackend()
        case _:
            raise ValueError(
                f"Unsupported _backend: {config._attn_implementation}, "
                "expected one of 'sdpa', 'eager', 'flash_attention_2'"
            )
