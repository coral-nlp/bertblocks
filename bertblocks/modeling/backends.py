"""Attention backend implementations with unified interface."""

from abc import ABC, abstractmethod

import torch
from einops import einsum, rearrange
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_qkvpacked_func

    # Otherwise triggers graph break?
    torch._dynamo.config.capture_scalar_outputs = True

from bertblocks.modeling.position import RotaryEmbedding


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

    @property
    @abstractmethod
    def supports_rope(self) -> bool:
        """Whether this backend supports RoPE positional encoding."""
        pass

    def forward_unpadded(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        rotary_emb: RotaryEmbedding | None = None,
        alibi_slopes: torch.Tensor | None = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with unpadded sequences."""
        if not self.supports_unpadded:
            raise NotImplementedError(f"{self.__class__.__name__} does not support unpadded sequences")
        return self._forward_unpadded(
            qkv,
            cu_seqlens,
            max_seq_len,
            num_heads,
            head_dim,
            rotary_emb,
            alibi_slopes,
            local_attention,
            dropout_p,
            deterministic,
        )

    def forward_padded(
        self,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        rotary_emb: RotaryEmbedding | None = None,
        alibi_slopes: torch.Tensor | None = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with padded sequences."""
        if not self.supports_padded:
            raise NotImplementedError(f"{self.__class__.__name__} does not support padded sequences")
        return self._forward_padded(
            qkv,
            attention_mask,
            num_heads,
            head_dim,
            rotary_emb,
            alibi_slopes,
            local_attention,
            dropout_p,
            deterministic,
        )

    @abstractmethod
    def _forward_unpadded(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore
        """Implement the unpadded forward pass."""
        pass

    @abstractmethod
    def _forward_padded(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore
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
        return False

    @property
    def supports_alibi(self) -> bool:
        """Whether this backend supports ALiBi positional bias."""
        return True

    @property
    def supports_local_attention(self) -> bool:
        """Whether this backend supports local attention."""
        return True

    @property
    def supports_rope(self) -> bool:
        """Whether this backend supports RoPE positional encoding."""
        return True

    def _forward_unpadded(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        rotary_emb: RotaryEmbedding | None = None,
        alibi_slopes: torch.Tensor | None = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flash attention forward pass."""
        qkv = rearrange(qkv, "... (t h d) -> ... t h d", t=3, h=num_heads, d=head_dim)

        if rotary_emb is not None:
            qkv = rotary_emb(qkv, cu_seqlens, max_seq_len)

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
        x = rearrange(x, "... h d -> ... (h d)")
        return x, w

    def _forward_padded(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore
        """Flash attention does not support padded sequences."""
        # TODO: this would be possible.
        raise NotImplementedError("Flash attention only supports unpadded sequences")


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

    @property
    def supports_rope(self) -> bool:
        """Whether this backend supports RoPE positional encoding."""
        return False

    def _forward_padded(
        self,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        rotary_emb: RotaryEmbedding | None = None,
        alibi_slopes: torch.Tensor | None = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """SDPA forward pass with padded sequences."""
        if rotary_emb is not None:
            raise NotImplementedError("RoPE is not supported with SDPA backend")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi is not supported with SDPA backend")
        if local_attention != (-1, -1):
            raise NotImplementedError("Local attention is not supported with SDPA backend")

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

    def _forward_unpadded(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore
        """SDPA backend does not support unpadded sequences."""
        raise NotImplementedError("SDPA backend only supports padded sequences")


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

    @property
    def supports_rope(self) -> bool:
        """Whether this backend supports RoPE positional encoding."""
        return False

    def _forward_padded(
        self,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        rotary_emb: RotaryEmbedding | None = None,
        alibi_slopes: torch.Tensor | None = None,
        local_attention: tuple[int, int] = (-1, -1),
        dropout_p: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Eager attention forward pass with padded sequences."""
        if rotary_emb is not None:
            raise NotImplementedError("RoPE is not supported with native backend")

        q, k, v = rearrange(qkv, "b s (t h d) -> t b h s d", t=3, h=num_heads, d=head_dim)
        scores = einsum(q, k, "b h i d, b h j d -> b h i j") * (head_dim**-0.5)
        mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)

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

    def _forward_unpadded(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:  # type: ignore
        """Eager backend does not support unpadded sequences."""
        raise NotImplementedError("Native backend only supports padded sequences")


# Registry of available backends
ATTENTION_BACKENDS = {
    "fa2": FlashBackend(),
    "sdpa": SDPABackend(),
    "eager": EagerBackend(),
}
