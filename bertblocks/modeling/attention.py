from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

import torch.nn.functional as F
from torch import nn

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.backends import get_attention
from bertblocks.modeling.initialization import TilableMixin, TileLinear, TileMode, tile_linear
from bertblocks.modeling.position import AlibiPositionalEncoding, RotaryPositionalEncoding


class Attention(TilableMixin, nn.Module):
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
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
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
        # QKV projection: Q has num_heads, K and V have num_kv_heads each
        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.proj = nn.Linear(self.hidden_size, qkv_dim, bias=config.attn_proj_bias)
        self.attn_gate = AttentionGate(config) if config.attention_gate is not None else None
        self.ffwd = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attn_out_bias)
        # Private inits
        self._rotary_enc = self._get_rope(config, layer_id=layer_id)
        self._backend = get_attention(config)
        if config.block_pos_enc_kind == "alibi":
            self.register_buffer("_slopes", AlibiPositionalEncoding.get_slopes(self.num_heads))
        else:
            self._slopes = None

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
                rope_dim=config.block_pos_enc_kwargs["rope_dim"],
                head_dim=self.head_dim,
                base=theta,
                interleaved=config.block_pos_enc_kwargs.get("interleaved", False),
                max_seq_len=config.max_sequence_length,
            )
        else:
            return None

    def _split_qkv(self, qkv: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Split combined QKV tensor into separate Q, K, V tensors.

        Args:
            qkv (torch.Tensor, shape [..., (num_heads + 2*num_kv_heads) * head_dim]):
                Combined QKV projection.

        Returns:
            tuple of (q, k, v) tensors with shapes:
                - q: [..., num_heads, head_dim]
                - k: [..., num_kv_heads, head_dim]
                - v: [..., num_kv_heads, head_dim]
        """
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        if qkv.dim() == 2:  # Unpadded: [s, qkv_dim]
            q = qkv[..., :q_dim].unflatten(-1, (self.num_heads, self.head_dim))  # s (h d) -> s h d
            k = qkv[..., q_dim : q_dim + kv_dim].unflatten(-1, (self.num_kv_heads, self.head_dim))  # s (h d) -> s h d
            v = qkv[..., q_dim + kv_dim :].unflatten(-1, (self.num_kv_heads, self.head_dim))  # s (h d) -> s h d
        else:  # Padded: [b, s, qkv_dim]
            q = qkv[..., :q_dim].unflatten(-1, (self.num_heads, self.head_dim))  # b s (h d) -> b s h d
            k = qkv[..., q_dim : q_dim + kv_dim].unflatten(
                -1, (self.num_kv_heads, self.head_dim)
            )  # b s (h d) -> b s h d
            v = qkv[..., q_dim + kv_dim :].unflatten(-1, (self.num_kv_heads, self.head_dim))  # b s (h d) -> b s h d

        return q, k, v

    def _apply_qknorm(self, q: torch.Tensor, k: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """Apply L2 normalization to query and key tensors.

        Args:
            q (torch.Tensor, shape [..., num_heads, head_dim]): Query tensor.
            k (torch.Tensor, shape [..., num_kv_heads, head_dim]): Key tensor.

        Returns:
            tuple of normalized (q, k) tensors with same shapes as input.

        References:
            - https://arxiv.org/abs/2010.04245
        """
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        return q, k

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
            x (torch.Tensor, shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]): Hidden state.
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for unpadded sequences.
            max_seq_len (int, optional): Maximum sequence length for unpadded sequences.
            attention_mask (torch.Tensor, optional): Attention mask for padded sequences.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Output and optional attention weights
        """
        # Fused projection and split into Q, K, V
        qkv = self.proj(x)
        q, k, v = self._split_qkv(qkv)

        # Rotary encoding if applicable
        if self._rotary_enc is not None:
            q, k = self._rotary_enc(q, k, cu_seqlens, max_seq_len)

        # QK-Norm if applicable
        if self.norm_qk:
            q, k = self._apply_qknorm(q, k)

        if cu_seqlens is not None and max_seq_len is not None:
            x_out, w = self._backend.forward_unpadded(
                q,
                k,
                v,
                cu_seqlens,
                max_seq_len,
                alibi_slopes=self._slopes,
                local_attention=self.local_attention,
                dropout_p=self.dropout_p if self.training else 0.0,
                deterministic=self.deterministic,
            )
        elif attention_mask is not None:
            x_out, w = self._backend.forward_padded(
                q,
                k,
                v,
                attention_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                deterministic=self.deterministic,
            )
        else:
            raise ValueError(
                "Neither `attention_mask` nor `cu_seqlens` were provided, no attention implementation applicable"
            )

        # Apply attention gating
        if self.attn_gate is not None:
            x_out = self.attn_gate(q, x_out)

        # Fuse heads back together
        x_out = self.ffwd(x_out)
        return x_out, w

    def tile_from(self, pretrained: "Attention", mode: str | TileMode = TileMode.tile_weights_from_middle) -> None:
        """Tile weights from a smaller pretrained Attention module.

        Handles the fused QKV projection (proj) and output projection (ffwd) separately,
        using the correct tiling strategy for each.

        Args:
            pretrained: Smaller pretrained Attention module to tile from.
            mode: Tiling strategy to use.
        """
        tile_linear(pretrained.proj, self.proj, linear_type=TileLinear.wqkv, mode=mode)
        tile_linear(pretrained.ffwd, self.ffwd, mode=mode)


class AttentionGate(nn.Module):
    """A multiplicative attention gate that should be positioned ahead of the final feed-forward module.

    Gating values are computed from the query vectors, which act as the input signal.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension size of attention heads.
        attention_gate_type (AttentionGate): Attention gate type.
        gate_proj (nn.Linear): Gating layer.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules.

    References:
        - Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
          (https://openreview.net/pdf?id=1b7whO4SfY)
    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.config = config

        self.num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        self.head_dim = hidden_size // self.num_heads

        self.attention_gate_type = config.attention_gate
        if self.attention_gate_type == "elementwise":
            self.gate_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        elif self.attention_gate_type == "headwise":
            self.gate_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        else:
            raise ValueError(f"Unknown type of attention gate: {self.attention_gate_type}")

    def forward(self, q: "Tensor", x: "Tensor") -> "Tensor":
        """Forward pass of the attention gate.

        Args:
            q (torch.Tensor, shape [batch_size, seq_len, num_heads, head_dim]
                or [total_seq_len, num_heads, head_dim]): Query tensor.
            x (torch.Tensor, shape [batch_size, seq_len, num_heads * head_dim]
                or [total_seq_len, num_heads * head_dim]): Hidden state after attention.

        Returns: torch.Tensor
            Hidden state modulated by query projection.
        """
        # Flatten q for projection: [..., h, d] -> [..., h*d]
        q_flat = q.flatten(-2)

        gate_output = self.gate_proj(q_flat)  # [s, h*d] or [b, s, h*d] or [s, h] or [b, s, h]

        if self.attention_gate_type == "headwise":
            gate_output = gate_output.unsqueeze(-1)
        else:  # elementwise
            # Reshape to [..., h, d]
            gate_output = gate_output.view(*q.shape)

        gate_output = torch.sigmoid(gate_output)

        # Apply gating
        x_out = x.view(*x.shape[:-1], self.num_heads, self.head_dim)
        x_out = x_out * gate_output

        # [..., h, d] -> [..., h*d]
        return x_out.flatten(-2)


__all__ = ["Attention", "AttentionGate"]
