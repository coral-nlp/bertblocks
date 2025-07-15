import functools
import math
from typing import Literal

import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    _score_mod_signature,
    create_block_mask,
    flex_attention,
)

from polybert.modeling.config import PolyBertConfig

# Type aliases for configuration
ScoreModType = Literal["none", "alibi", "relative", "rope", "sinusoidal"]
MaskType = Literal["none", "causal", "doc", "sliding", "dilated", "packed_doc"]


class ScoreModFunctions:
    """Collection of score modification functions for flex-attention.
    Functions are partially applied to improve compile-ability.
    """

    @staticmethod
    def _alibi(
        score: torch.Tensor,
        _b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        slopes: torch.Tensor,
    ) -> torch.Tensor:
        """Implementation of ALIBI score mod

        Reference
        ---------
        Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation: https://arxiv.org/abs/2108.12409
        """
        scale = slopes[h]
        bias = (kv_idx - q_idx) * scale
        return score + bias

    @staticmethod
    def alibi(num_heads: int) -> _score_mod_signature:
        """Alibi bias score_mod."""
        slopes = torch.exp2(torch.arange(num_heads, dtype=torch.float32) * (-8.0 / num_heads))
        return functools.partial(ScoreModFunctions._alibi, slopes=slopes)

    @staticmethod
    def _relative(
        score: torch.Tensor,
        _b: torch.Tensor,
        _h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        max_distance: int,
    ) -> torch.Tensor:
        rel_pos = q_idx - kv_idx
        rel_pos = torch.clamp(rel_pos, -max_distance, max_distance)
        return score + rel_pos.float()

    @staticmethod
    def relative(max_distance: int = 128) -> _score_mod_signature:
        """Relative positional score_mod."""
        return functools.partial(ScoreModFunctions._relative, max_distance=max_distance)


class MaskFunctions:
    """Collection of mask creation functions for flex-attention.
    Functions are partially applied to improve compile-ability
    """

    @staticmethod
    def _causal(_b: torch.Tensor, _h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return q_idx >= kv_idx

    @staticmethod
    def causal() -> _mask_mod_signature:
        """Causal mask."""
        return MaskFunctions._causal

    @staticmethod
    def _sliding(
        _b: torch.Tensor, _h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
    ) -> torch.Tensor:
        return torch.abs(q_idx - kv_idx) <= window_size

    @staticmethod
    def sliding(window_size: int) -> _mask_mod_signature:
        """Sliding window mask."""
        return functools.partial(MaskFunctions._sliding, window_size=window_size)

    @staticmethod
    def _dilated(
        _b: torch.Tensor,
        _h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        dilation_rate: int,
        window_size: int,
    ) -> torch.Tensor:
        distance = torch.abs(q_idx - kv_idx)
        return (distance <= window_size) | (distance % dilation_rate == 0)

    @staticmethod
    def dilated(dilation_rate: int, window_size: int) -> _mask_mod_signature:
        """Dilated mask function."""
        return functools.partial(MaskFunctions._dilated, dilation_rate=dilation_rate, window_size=window_size)

    @staticmethod
    def _doc(
        b: torch.Tensor, 
        _h: torch.Tensor, 
        q_idx: torch.Tensor, 
        kv_idx: torch.Tensor, 
        cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Prevents attention across document boundaries in packed sequences."""
        batch_idx = b.item()
        if batch_idx < len(cu_seqlens) - 1:
            doc_start = cu_seqlens[batch_idx]
            doc_end = cu_seqlens[batch_idx + 1]
            
            # Only allow attention within the same document
            q_in_doc = (q_idx >= doc_start) & (q_idx < doc_end)
            kv_in_doc = (kv_idx >= doc_start) & (kv_idx < doc_end)
            
            return q_in_doc & kv_in_doc
        else:
            # Fallback: allow all attention for this batch
            return torch.ones_like(q_idx, dtype=torch.bool)

    @staticmethod
    def doc(cu_seqlens: torch.Tensor, document_separator_token_id: int | None = None) -> _mask_mod_signature:
        """Document mask for packed sequences."""
        return functools.partial(
            MaskFunctions._doc,
            cu_seqlens=cu_seqlens, 
            document_separator_token_id=document_separator_token_id
        )


class PolyBertAttention(nn.Module):
    """Extended PolyBERT attention with configurable score mods and masks.
    """

    def __init__(self, config: PolyBertConfig, **kwargs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused linear layers for better performance
        self.proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()

        self.score_mod_type = config.attn_score_mod_type
        self.mask_type = config.attn_mask_type
        self.kwargs = kwargs

        # Initialize score modification and mask functions at init time
        self._init_score_mod()
        self._init_mask_params()

    def _init_score_mod(self):
        """Initialize score modification based on type."""
        match self.score_mod_type:
            case "none":
                self.score_mod = None
            case "alibi":
                self.score_mod = ScoreModFunctions.alibi(self.num_heads)
            case "relative":
                max_distance = self.kwargs.get("max_relative_distance", 128)
                self.score_mod = ScoreModFunctions.relative(max_distance)
            case "rope":
                raise NotImplementedError
            case "sinusoidal":
                raise NotImplementedError
            case _:
                raise ValueError(f"Unknown score_mod_type: {self.score_mod_type}")

    def _init_mask_params(self):
        """Initialize mask parameters that can be set at init time."""
        self.mask_params = {}
        match self.mask_type:
            case "sliding":
                self.mask_params["window_size"] = self.kwargs.get("window_size", 128)
            case "dilated":
                self.mask_params["dilation_rate"] = self.kwargs.get("dilation_rate", 2)
                self.mask_params["window_size"] = self.kwargs.get("window_size", 64)

    def create_mask_fn(
        self, 
        attention_mask: torch.Tensor | None = None, 
        cu_seqlens: torch.Tensor | None = None
    ) -> _mask_mod_signature | None:
        """Create mask function based on type."""
        match self.mask_type:
            case "none":
                return None
            case "causal":
                return MaskFunctions.causal()
            case "doc":
                if attention_mask is None:
                    return None
                return MaskFunctions.doc(attention_mask)
            case "sliding":
                return MaskFunctions.sliding(self.mask_params["window_size"])
            case "dilated":
                return MaskFunctions.dilated(self.mask_params["dilation_rate"], self.mask_params["window_size"])
            case "packed":
                if cu_seqlens is None:
                    return None
                return MaskFunctions.doc(cu_seqlens)
            case _:
                raise ValueError(f"Unknown mask_type: {self.mask_type}")

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        output_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n_batch, n_seq, _ = x.shape

        # Projection for Q, K, V
        qkv = self.proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Create mask if needed
        mask_fn = self.create_mask_fn(cu_seqlens)
        block_mask = None
        if mask_fn is not None:
            block_mask = create_block_mask(
                mask_fn,
                n_batch,
                self.num_heads,
                n_seq,
                n_seq,
            )

        # Apply flex attention kernel
        x, w = flex_attention(
            q, k, v, score_mod=self.score_mod, block_mask=block_mask, scale=self.scale, return_lse=True
        )

        # Reshape back
        x = x.transpose(1, 2).contiguous().reshape(n_batch, n_seq, -1)

        # Output projection and dropout
        x = self.ffwd(x)
        x = self.drop(x)

        return x, w if output_attention else None
