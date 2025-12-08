from typing import TYPE_CHECKING

import torch
from einops import rearrange

if TYPE_CHECKING:
    from torch import Tensor

import torch.nn.functional as F
from torch import nn

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.backends import get_attention
from bertblocks.modeling.position import AlibiPositionalEncoding, RotaryPositionalEncoding


class Attention(nn.Module):
    """Attention with configurable positional encodings.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension size of attention heads.
        max_seq_len (int): Maximum sequence length.
        dropout_p (float): Dropout probability for attention.
        local_attention (tuple[int, int]): Local attention size, if applied.
        deterministic (bool): Whether to use deterministic attention.
        proj (nn.Linear): Fused QKV projection layer.
        ffwd (nn.Linear): Feed-forward layer to combine heads after attention.
        qk_norm (bool): Whether to apply query-key-normalization.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `num_attention_heads`: Number of attention heads in multi-head attention
                - `hidden_size`: Dimensionality of hidden layers (must be divisible by num_attention_heads)
                - `max_sequence_length`: Maximum sequence length for positional encodings
                - `attn_proj_bias`: Whether to include bias in QKV projection
                - `attn_out_bias`: Whether to include bias in output projection
                - `attn_dropout_prob`: Dropout probability for attention weights
                - `block_pos_enc_kind`: Type of positional embedding ("alibi", "rope", "relative", etc.)

            layer_id (int): layer id indicating index in the encoder stack.
    """

    def __init__(self, config: "BertBlocksConfig", layer_id: int):
        super().__init__()
        # General hyperparameters
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = config.max_sequence_length
        self.dropout_p = config.attn_dropout_prob
        if config.global_attention_every_n_layers != 0:
            self.local_attention = (
                config.local_attention if layer_id % config.global_attention_every_n_layers != 0 else (-1, -1)
            )
        else:
            self.local_attention = (-1, -1)
        self.deterministic = True
        # Layers
        self.norm_qk = config.norm_qk
        self.proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=config.attn_proj_bias)
        self.ffwd = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attn_out_bias)
        # Private inits
        self._rotary_enc = self._get_rope(config, layer_id=layer_id)
        self._backend = get_attention(config)

    def _get_rope(self, config: "BertBlocksConfig", layer_id: int) -> "RotaryPositionalEncoding | None":
        """Initialize rotary positional encoding if needed.

        Args:
            config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
                other submodules. Keys used at top level:

                    - `block_pos_enc_kind`: Type of positional embedding ("alibi", "rope", "relative", etc.)
                    - `block_pos_enc_kwargs`: Additional positional encoding arguments
                    - `num_attention_heads`: Number of attention heads in multi-head attention
                    - `global_attention_every_n_layers`: Global attention layer stride

            layer_id (int): layer id indicating index in the encoder stack.
        """
        if config.block_pos_enc_kind == "rope":
            theta = (
                config.block_pos_enc_kwargs.get("base_local") or config.block_pos_enc_kwargs.get("base", 10_000.0)
                if config.global_attention_every_n_layers > 0 and layer_id % config.global_attention_every_n_layers > 0
                else config.block_pos_enc_kwargs.get("base_global") or config.block_pos_enc_kwargs.get("base", 10_000.0)
            )
            return RotaryPositionalEncoding(
                dim=config.block_pos_enc_kwargs["dim"],
                base=theta,
                interleaved=config.block_pos_enc_kwargs.get("interleaved", False),
                max_seq_len=config.max_sequence_length,
            )
        else:
            return None

    def _apply_qknorm(self, qkv: torch.Tensor) -> torch.Tensor:
        """Apply the given norm selectively to the q & k part of the combined input.

        Args:
            qkv (torch.Tensor, shape [total_seq_len, (3 * num_heads * head_dim)] or [batch_size, seq_len,
                (3 * num_heads * head_dim)]): projected combined QKV tensor.

        Returns:
            torch.Tensor, same shape as input: combined QKV tensor after selectively applying norm to QK part.

        References:
            - https://arxiv.org/abs/2010.04245
        """
        if qkv.dim() == 2:  # Unpadded
            q, k, v = rearrange(qkv, "s (t h d) -> t s h d", t=3, h=self.num_heads, d=self.head_dim)
        else:  # Padded
            batch_size, seq_len, _ = qkv.shape
            q, k, v = rearrange(
                qkv,
                "b s (t h d) -> t b s h d",
                b=batch_size,
                s=seq_len,
                t=3,
                h=self.num_heads,
                d=self.head_dim,
            )

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        if qkv.dim() == 2:  # Unpadded
            qkv = rearrange(torch.stack([q, k, v], 0), "t s h d -> s (t h d)", t=3, h=self.num_heads, d=self.head_dim)
        else:  # Padded
            batch_size, seq_len, _ = qkv.shape
            qkv = rearrange(
                torch.stack([q, k, v], 0),
                "t b s h d -> b s (t h d)",
                b=batch_size,
                s=seq_len,
                t=3,
                h=self.num_heads,
                d=self.head_dim,
            )
        return qkv

    def forward(
        self,
        x: "Tensor",
        attention_mask: "Tensor | None" = None,
        cu_seqlens: "Tensor | None" = None,
        max_seq_len: int | None = None,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass of the attention mechanism.

        Automatically routes to padded or unpadded implementation based on backend capabilities.

        Args:
            x (torch.Tensor): Input hidden state
            indices (torch.Tensor, optional): Sequence indices for unpadded sequences
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for unpadded sequences
            max_seq_len (int, optional): Maximum sequence length for unpadded sequences
            attention_mask (torch.Tensor, optional): Attention mask for padded sequences

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Output and optional attention weights
        """
        # Fused projection
        qkv = self.proj(x)
        # Rotary encoding if applicable
        if self._rotary_enc is not None:
            qkv = self._rotary_enc(qkv, self.num_heads, self.head_dim, cu_seqlens, max_seq_len)
        # QK-Norm if applicable
        if self.norm_qk:
            qkv = self._apply_qknorm(qkv)

        if cu_seqlens is not None and max_seq_len is not None:
            x, w = self._backend.forward_unpadded(
                qkv,
                cu_seqlens,
                max_seq_len,
                self.num_heads,
                self.head_dim,
                alibi_slopes=AlibiPositionalEncoding.get_slopes(self.num_heads, device=qkv.device),
                local_attention=self.local_attention,
                dropout_p=self.dropout_p if self.training else 0.0,
                deterministic=self.deterministic,
            )
        elif attention_mask is not None:
            x, w = self._backend.forward_padded(
                qkv,
                attention_mask,
                self.num_heads,
                self.head_dim,
                dropout_p=self.dropout_p if self.training else 0.0,
                deterministic=self.deterministic,
            )
        else:
            raise ValueError(
                "Neither `attention_mask` nor `cu_seqlens` were provided, no attention implementation applicable"
            )
        # Fuse heads back together
        x = self.ffwd(x)
        return x, w


__all__ = ["Attention"]
