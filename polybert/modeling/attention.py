import functools
import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import (
    BlockMask,
    _score_mod_signature,
    flex_attention,
)

from polybert.modeling.config import PolyBertConfig


class ScoreModFunctions:
    """Collection of score modification functions for flex-attention.

    This class provides static methods for various positional encoding and attention
    bias schemes that can be used with PyTorch's flex_attention mechanism. Functions
    are partially applied to improve compile-ability and performance.

    The score modification functions modify the attention scores before the softmax
    operation, allowing for various forms of positional biases and attention patterns.
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
        """Add Attention with Linear Biases (ALIBI) score modification.

        ALIBI adds a linear bias to attention scores based on the distance between
        query and key positions, with head-specific slopes. This enables length
        extrapolation beyond training sequence lengths.

        Args:
            score: Raw attention scores of shape (batch_size, num_heads, seq_len, seq_len).
            _b: Batch index tensor (unused in this implementation).
            h: Head index tensor of shape (num_heads,).
            q_idx: Query position indices of shape (seq_len,).
            kv_idx: Key-value position indices of shape (seq_len,).
            slopes: Head-specific slope values of shape (num_heads,). Smaller slopes
                   for heads that focus on closer positions.

        Returns:
            Modified attention scores with ALIBI bias applied, same shape as input score.

        Note:
            The bias is calculated as (kv_idx - q_idx) * slope[h], creating a linear
            penalty based on distance that varies per attention head.

        References:
            - "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (https://arxiv.org/abs/2108.12409)

        """
        scale = slopes[h]
        bias = (kv_idx - q_idx) * scale
        return score + bias

    @staticmethod
    def alibi(num_heads: int) -> _score_mod_signature:
        """Create an ALIBI score modification function for the given number of heads.

        This method generates head-specific slopes and returns a partially applied
        ALIBI function that can be used with flex_attention.

        Args:
            num_heads: Number of attention heads. Must be positive.

        Returns:
            A partially applied ALIBI score modification function with the signature
            expected by flex_attention.

        Note:
            Slopes are computed as 2^(-(8/num_heads) * head_idx) for each head,
            creating an exponential decay pattern where different heads have
            different sensitivities to positional distance.

        """
        slopes = torch.exp2(torch.arange(num_heads, dtype=torch.float32) * (-8.0 / num_heads))
        return functools.partial(ScoreModFunctions._alibi, slopes=slopes)


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

    def __init__(self, config: PolyBertConfig):
        """Initialize the PolyBERT attention mechanism.

        Args:
            config: PolyBERT configuration object containing model hyperparameters
                   including hidden_size, num_attention_heads, attn_dropout_prob,
                   and pos_emb_kind.

        """
        super().__init__()
        # General hyperparameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
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

        Note:
            Different positional encoding schemes apply modifications at different
            stages: ALIBI modifies scores, while RoPE modifies query/key projections.

        """
        match score_mod_type:
            case "sinusoidal":
                return None
            case "alibi":
                return ScoreModFunctions.alibi(self.num_heads)
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

        Note:
            This method returns the transformation applied to the fused QKV projection
            before splitting into separate Q, K, V tensors.

        """
        match qk_mod_type:
            case "sinusoidal":
                return nn.Identity()
            case "alibi":
                return nn.Identity()
            case "rope":
                raise NotImplementedError
            case _:
                return nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        doc_mask: BlockMask,
        output_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        Note:
            The attention computation follows these steps:
            1. Project input to Q, K, V using fused linear layer
            2. Apply positional encoding modifications (if any)
            3. Reshape for multi-head attention
            4. Compute attention using flex_attention with score_mod and block_mask
            5. Reshape and apply output projection with dropout

        """
        n_batch, n_seq, _ = x.shape

        # Projection for Q, K, V
        qkv = self.proj(x)
        qkv = self.qk_mod(qkv)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(n_batch, n_seq, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply flex attention kernel
        x, w = flex_attention(q, k, v, score_mod=self.score_mod, block_mask=doc_mask, scale=self.scale, return_lse=True)

        # Reshape back
        x = x.transpose(1, 2).contiguous().reshape(n_batch, n_seq, -1)

        # Output projection and dropout
        x = self.ffwd(x)
        x = self.drop(x)

        return x, w if output_attention else None
