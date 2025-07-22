from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import _score_mod_signature

import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from polybert.modeling.config import PolyBertConfig
from polybert.modeling.position import AlibiPositionalEncoding, RotaryPositionalEncoding


class PolyBertAttention(nn.Module):
    """Extended PolyBERT attention mechanism with configurable positional encodings.

    This class implements a flexible attention mechanism using flex_attention for efficient
    computation. Applies block masking for document-level attention patterns.

    The attention mechanism follows the standard transformer architecture but
    with configurable positional biases that can be applied either to the
    query-key projections or as score modifications.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        scale: Scaling factor for attention scores (1/sqrt(head_dim)).
        proj: Fused linear projection for Q, K, V (3 * hidden_size output).
        ffwd: Output projection layer.
        drop: Dropout layer for attention weights.
        qk_mod: Query-key modification module (for positional encodings like RoPE).
        score_mod: Score modification function (for positional biases like ALIBI).

    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBERT attention mechanism.

        Args:
            config (PolyBertConfig): Configuration object determining model hyperparameters.

        """
        super().__init__()
        # General hyperparameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.max_seq_len = config.max_sequence_length
        # Fused linear layers for better performance
        self.proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()
        # Positional embeddings (they apply at different stages of attention depending on type)
        self.qk_mod = self._qk_mod(config.pos_emb_kind)
        self.score_mod = self._score_mod(config.pos_emb_kind)

    def _score_mod(self, score_mod_type: str) -> "_score_mod_signature | None":
        """Initialize score modification function based on positional encoding type.

        Args:
            score_mod_type: Type of positional encoding to use. Supported values:
                          - "sinusoidal": No score modification (handled in embeddings)
                          - "alibi": ALIBI linear bias score modification
                          - "rope": No score modification (handled in qk_mod)
                          - Any other value: No score modification

        Returns:
            Score modification function compatible with flex_attention, or None
            if no score modification is needed for the given type.

        """
        match score_mod_type:
            case "sinusoidal":
                return None
            case "alibi":
                return AlibiPositionalEncoding(self.num_heads)()
            case "rope":
                return None
            case _:
                return None

    def _qk_mod(self, qk_mod_type: str) -> "nn.Module":
        """Initialize query-key modification module based on positional encoding type.

        Args:
            qk_mod_type: Type of positional encoding to use. Supported values:
                        - "sinusoidal": No modification (handled in embeddings)
                        - "alibi": No modification (handled in score_mod)
                        - "rope": Rotary Position Embedding (not yet implemented)
                        - Any other value: No modification

        Returns:
            PyTorch module that modifies query-key projections, or Identity module
            if no modification is needed.

        Raises:
            NotImplementedError: If "rope" is specified (RoPE not yet implemented).

        """
        match qk_mod_type:
            case "sinusoidal":
                return nn.Identity()
            case "alibi":
                return nn.Identity()
            case "rope":
                return RotaryPositionalEncoding(self.head_dim, self.max_seq_len)
            case _:
                return nn.Identity()

    def forward(
        self,
        x: "torch.Tensor",
        doc_mask: "BlockMask",
        output_attention: bool = False,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass of the PolyBERT attention mechanism.

        Computes multi-head self-attention with configurable positional encodings
        and block masking. Uses PyTorch's flex_attention for efficient computation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            doc_mask: Block mask defining attention patterns, typically used to
                     prevent attention across document boundaries.
            output_attention: Whether to return attention weights along with the output.
                            Default is False for efficiency.

        Returns:
            A tuple containing:
            - output: Attention output tensor of shape (batch_size, seq_len, hidden_size).
            - attention_weights: Log-sum-exp attention weights if output_attention is True,
                               otherwise None. Shape is (batch_size, num_heads, seq_len, seq_len).

        """
        n_batch, n_seq, _ = x.shape

        # Projection for Q, K, V
        qkv = self.proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Add qk-mod (RoPE) if applicable (will be nn.Identity otherwise)
        q = self.qk_mod(q)
        k = self.qk_mod(k)

        # Apply flex attention kernel
        x, w = flex_attention(q, k, v, score_mod=self.score_mod, block_mask=doc_mask, scale=self.scale, return_lse=True)

        # Reshape back
        x = x.transpose(1, 2).contiguous().reshape(n_batch, n_seq, -1)

        # Output projection and dropout
        x = self.ffwd(x)
        x = self.drop(x)

        return x, w if output_attention else None
