from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import _score_mod_signature


import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from polybert.modeling.config import PolyBertConfig
from polybert.modeling.position import AlibiPositionalEncoding, RelativePositionalEncoding, RotaryPositionalEncoding


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
            config (PolyBertConfig): Configuration object containing:
                - num_attention_heads: Number of attention heads in multi-head attention
                - hidden_size: Dimensionality of hidden layers (must be divisible by num_attention_heads)
                - max_sequence_length: Maximum sequence length for positional encodings
                - attn_proj_bias: Whether to include bias in QKV projection
                - attn_out_bias: Whether to include bias in output projection
                - attn_dropout_prob: Dropout probability for attention weights
                - pos_emb_kind: Type of positional embedding ("alibi", "rope", "relative", etc.)

        """
        super().__init__()
        # General hyperparameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_sequence_length
        # Fused linear layers for better performance
        self.proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attn_proj_bias)
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()
        # Positional embeddings (they apply at different stages of attention depending on type)
        self.qk_mod = self._qk_mod(config.pos_emb_kind)
        self.score_mod = self._score_mod(config.pos_emb_kind)

    def _score_mod(self, score_mod_type: str) -> "_score_mod_signature | None":
        """Initialize score modification function based on positional encoding type.

        Args:
            score_mod_type (str): Type of positional encoding to use. Supported values:
                          - "alibi": ALIBI linear bias score modification
                          - "relative": relative positional bias score modification
                          - Any other value: No score modification

        Returns:
            _score_mod_signature | None: Score modification function compatible with flex_attention, or None
            if no score modification is needed for the given type.

        """
        match score_mod_type:
            case "alibi":
                return AlibiPositionalEncoding(self.num_heads)()
            case "relative":
                return RelativePositionalEncoding()()
            case _:
                return None

    def _qk_mod(self, qk_mod_type: str) -> "nn.Module":
        """Initialize query-key modification module based on positional encoding type.

        Args:
            qk_mod_type (str): Type of positional encoding to use. Supported values:
                        - "rope": Rotary Position Embedding
                        - Any other value: No modification

        Returns:
            nn.Module: PyTorch module that modifies query-key projections, or Identity module
            if no modification is needed.

        """
        match qk_mod_type:
            case "rope":
                return RotaryPositionalEncoding(self.head_dim, self.max_seq_len)
            case _:
                return nn.Identity()

    def forward(
        self,
        x: "torch.Tensor",
        attention_mask: "BlockMask",
        output_attention: bool = False,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass of the PolyBERT attention mechanism.

        Computes multi-head self-attention with configurable positional encodings
        and block masking. Uses PyTorch's flex_attention.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): The input hidden state.
            attention_mask (BlockMask): Flex attention block mask to ignore padding tokens.
            output_attention (bool): Whether to return attention weights along with the output.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - output: Attention output tensor of shape [batch_size, seq_len, hidden_size].
                - attention_weights: Log-sum-exp attention weights if output_attention is True,
                  otherwise None. Shape [batch_size, num_heads, seq_len, seq_len].

        """
        n_batch, n_seq = x.size()[:-1]

        # Projection for Q, K, V
        qkv = self.proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(n_batch, n_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(n_batch, n_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(n_batch, n_seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Add qk-mod (RoPE) if applicable (will be nn.Identity otherwise)
        # q = self.qk_mod(q)
        # k = self.qk_mod(k)

        # Apply flex attention kernel
        x, w = flex_attention(q, k, v, block_mask=attention_mask, score_mod=self.score_mod, return_lse=True)
        w = None
        # Reshape back
        x = x.permute(0, 2, 1, 3).contiguous().reshape(n_batch, n_seq, -1)

        # Output projection
        x = self.ffwd(x)

        return x, w if output_attention else None
