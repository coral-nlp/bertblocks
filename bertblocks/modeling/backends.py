"""Attention backend implementations with unified interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor


import torch
from einops import einsum, rearrange
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func


class AttentionBackend(ABC):
    """Abstract base class for attention backends."""

    @property
    @abstractmethod
    def supports_unpadded(self) -> bool:
        """Whether this backend supports unpadded sequences."""
        pass

    @property
    @abstractmethod
    def supports_padded(self) -> bool:
        """Whether this backend supports padded sequences."""
        pass

    @property
    @abstractmethod
    def supports_alibi(self) -> bool:
        """Whether this backend supports ALiBi positional bias."""
        pass

    @property
    @abstractmethod
    def supports_local_attention(self) -> bool:
        """Whether this backend supports local attention."""
        pass

    def _compatible(
        self,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        mode: Literal["unpadded", "padded"] | None = None,
    ) -> None:
        if alibi_slopes is not None and not self.supports_alibi:
            raise NotImplementedError(f"{self.__class__.__name__} does not support ALIBI positional encoding")
        if local_attention != (-1, -1) and not self.supports_local_attention:
            raise NotImplementedError(f"{self.__class__.__name__} does not support local attention")
        if mode == "unpadded" and not self.supports_unpadded:
            raise NotImplementedError(f"{self.__class__.__name__} does not support unpadded sequences")
        if mode == "padded" and not self.supports_padded:
            raise NotImplementedError(f"{self.__class__.__name__} does not support padded sequences")

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
        self._compatible(alibi_slopes, local_attention, "unpadded")
        return self._forward_unpadded(
            qkv,
            cu_seqlens,
            max_seq_len,
            num_heads,
            head_dim,
            alibi_slopes,
            local_attention,
            dropout_p,
            deterministic,
        )

    def forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with padded sequences."""
        self._compatible(alibi_slopes, local_attention, "padded")
        return self._forward_padded(
            qkv,
            attention_mask,
            num_heads,
            head_dim,
            alibi_slopes,
            local_attention,
            dropout_p,
            deterministic,
        )

    @abstractmethod
    def _forward_unpadded(self, *args, **kwargs) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """Implement the unpadded forward pass."""
        pass

    @abstractmethod
    def _forward_padded(self, *args, **kwargs) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """Implement the padded forward pass."""
        pass


class FlashBackend(AttentionBackend):
    """Flash Attention 2 backend."""

    @property
    def supports_unpadded(self) -> bool:
        """Whether this backend supports unpadded sequences."""
        return True

    @property
    def supports_padded(self) -> bool:
        """Whether this backend supports padded sequences."""
        return True

    @property
    def supports_alibi(self) -> bool:
        """Whether this backend supports ALiBi positional bias."""
        return True

    @property
    def supports_local_attention(self) -> bool:
        """Whether this backend supports local attention."""
        return True

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
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":  # type: ignore
        """Forward pass for flash attention with padding."""
        orig_dtype = qkv.dtype
        qkv = qkv.to(torch.bfloat16)
        qkv = rearrange(qkv, "b s (t h d) -> b s t h d", t=3, h=num_heads, d=head_dim)
        x, _, w = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            causal=False,
            softcap=0.0,
            window_size=local_attention,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=True,
        )
        x = x.to(orig_dtype)
        x = rearrange(x, "b s h d -> b s (h d)")
        return x, w


class SDPABackend(AttentionBackend):
    """PyTorch SDPA backend - works efficiently with padded sequences."""

    @property
    def supports_unpadded(self) -> bool:
        """Whether this backend supports unpadded sequences."""
        return False

    @property
    def supports_padded(self) -> bool:
        """Whether this backend supports padded sequences."""
        return True

    @property
    def supports_alibi(self) -> bool:
        """Whether this backend supports ALiBi positional bias."""
        return False

    @property
    def supports_local_attention(self) -> bool:
        """Whether this backend supports local attention."""
        return False

    def _forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """SDPA forward pass with padded sequences."""
        batch_size, seqlen, _ = qkv.shape
        q, k, v = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=num_heads, d=head_dim)

        attn_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, dtype=q.dtype, tgt_len=seqlen)
        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        output = rearrange(output, "b h s d -> b s (h d)")
        return output, None

    def _forward_unpadded(self, *args, **kwargs):  # type: ignore
        """SDPA backend does not support unpadded sequences."""
        raise NotImplementedError


class EagerBackend(AttentionBackend):
    """Native PyTorch backend."""

    @property
    def supports_unpadded(self) -> bool:
        """Whether this backend supports unpadded sequences."""
        return False

    @property
    def supports_padded(self) -> bool:
        """Whether this backend supports padded sequences."""
        return True

    @property
    def supports_alibi(self) -> bool:
        """Whether this backend supports ALiBi positional bias."""
        return True

    @property
    def supports_local_attention(self) -> bool:
        """Whether this backend supports local attention."""
        return True

    def _forward_padded(
        self,
        qkv: "Tensor",
        attention_mask: "Tensor",
        num_heads: int,
        head_dim: int,
        alibi_slopes: "Tensor | None" = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> "tuple[Tensor, Tensor | None]":
        """Eager attention forward pass with padded sequences."""
        q, k, v = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=num_heads, d=head_dim)
        scores = einsum(q, k, "b h i d, b h j d -> b h i j") * (head_dim**-0.5)
        mask = (attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)).bool()

        if local_attention != (-1, -1) and local_attention[0] > 0:
            window_size = local_attention[0]
            pos = torch.arange(qkv.shape[1], device=qkv.device)
            local_mask = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs() <= window_size
            mask = mask & local_mask.unsqueeze(0)

        if alibi_slopes is not None:
            pos = torch.arange(qkv.shape[1], device=qkv.device)
            pos_diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
            alibi_bias = einsum(alibi_slopes, pos_diff, "h, i j -> h i j").unsqueeze(0)
            scores = scores + alibi_bias

        scores = scores.masked_fill(~mask.unsqueeze(1), -float("inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(1), 0.0)

        if dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

        output = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")
        output = rearrange(output, "b h s d -> b s (h d)")
        return output, None

    def _forward_unpadded(self, *args, **kwargs):  # type: ignore
        """Eager backend does not support unpadded sequences."""
        raise NotImplementedError


# Registry of available backends
ATTENTION_BACKENDS = {
    "fa2": FlashBackend(),
    "sdpa": SDPABackend(),
    "eager": EagerBackend(),
}
