"""Attention _backend implementations with unified interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from bertblocks import BertBlocksConfig

if TYPE_CHECKING:
    from torch import Tensor


import torch
from einops import einsum, rearrange
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_qkvpacked_func


class AttentionBackend(ABC):
    """Abstract base class for attention backends."""

    def forward_unpadded(
        self,
        qkv: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with unpadded sequences."""
        return self._forward_unpadded(
            qkv=qkv,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            alibi_slopes=alibi_slopes,
            local_attention=local_attention,
            dropout_p=dropout_p,
            deterministic=deterministic,
        )

    def forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with padded sequences."""
        return self._forward_padded(
            qkv=qkv,
            attention_mask=attention_mask,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_p=dropout_p,
            deterministic=deterministic,
        )

    @abstractmethod
    def _forward_unpadded(
        self,
        qkv: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
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
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """Implement the padded forward pass."""
        pass


class FlashBackend(AttentionBackend):
    """Flash Attention 2 _backend."""

    def _forward_unpadded(
        self,
        qkv: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor]":
        """Flash attention forward pass without padding."""
        qkv = rearrange(qkv, "s (t h d) -> s t h d", t=3, h=num_heads, d=head_dim)
        orig_dtype = qkv.dtype
        qkv = qkv.to(torch.bfloat16)
        x, _, w = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens.to(torch.int32),
            max_seq_len,
            dropout_p=dropout_p,
            causal=False,
            softcap=0.0,
            window_size=local_attention,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        x = x.to(orig_dtype)
        x = rearrange(x, "s h d -> s (h d)")
        return x, w

    def _forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """FlashAttention _backend does not support unpadded sequences."""
        raise NotImplementedError


class SDPABackend(AttentionBackend):
    """PyTorch SDPA _backend - works efficiently with padded sequences."""

    def _forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """SDPA forward pass with padded sequences."""
        batch_size, seqlen, _ = qkv.shape
        q, k, v = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=num_heads, d=head_dim)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        output = rearrange(output, "b h s d -> b s (h d)")
        return output, None

    def _forward_unpadded(self, *args, **kwargs):  # type: ignore
        """SDPA _backend does not support unpadded sequences."""
        raise NotImplementedError


class EagerBackend(AttentionBackend):
    """Native PyTorch _backend."""

    def _forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Eager attention forward pass with padded sequences."""
        q, k, v = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=num_heads, d=head_dim)
        scores = einsum(q, k, "b h i d, b h j d -> b h i j") * (head_dim**-0.5)

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

        output = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")
        output = rearrange(output, "b h s d -> b s (h d)")
        return output, None

    def _forward_unpadded(self, *args, **kwargs):  # type: ignore
        """Eager _backend does not support unpadded sequences."""
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

        - `fa2`: Flash Attention
        - `sdpa`: torch scaled dot product attention
        - `eager`: native torch attention

    """
    match config.attn_implementation:
        case "fa2":
            return FlashBackend()
        case "sdpa":
            return SDPABackend()
        case "eager":
            return EagerBackend()
        case _:
            raise ValueError(
                f"Unsupported _backend: {config.attn_implementation}, expected one of 'sdpa', 'eager', 'fa2'"
            )
