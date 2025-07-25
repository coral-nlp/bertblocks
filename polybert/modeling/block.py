from typing import TYPE_CHECKING

from torch.nn.attention.flex_attention import BlockMask

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.attention import PolyBertAttention
from polybert.modeling.mlp import get_mlp
from polybert.modeling.norms import get_norm


class PolyBertBlock(nn.Module):
    """A single transformer block implementation for PolyBert.

    This class implements a standard transformer block with attention and feed-forward
    layers, supporting both pre-normalization and post-normalization schemes.

    The block consists of:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Optional layer normalization (pre/post/both/none)

    Args:
        config (PolyBertConfig): Configuration object determining model hyperparameters.

    References:
         - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)
         - "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/pdf/2002.04745)

    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize a PolyBert transformer block.

        Sets up the attention mechanism, feed-forward network, and normalization
        layers based on configuration. Normalization layers are set to Identity
        when not needed according to the norm_kind setting.

        Args:
            config (PolyBertConfig): Configuration object containing:
                - norm_kind: When to apply normalization ("pre", "post", "both", "none")
                - attn_dropout_prob: Dropout probability for attention weights
                - hidden_dropout_prob: Dropout probability for hidden layer outputs

        """
        super().__init__()
        self.attn = PolyBertAttention(config)
        self.ffwd = get_mlp(config)
        self.pre_norm_attn = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.pre_norm_ffwd = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm_attn = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.post_norm_ffwd = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.attn_drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()
        self.hidden_drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()

    def forward(
        self, x: "torch.Tensor", attention_mask: "BlockMask", output_attention: "bool | None" = False
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass through the transformer block.

        Processes input through attention and feed-forward layers with residual
        connections. Supports both pre-normalization and post-normalization schemes.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): The hidden state of
                the previous transformer block.
            attention_mask (BlockMask): Block mask for efficient attention computation,
                typically created using torch.nn.attention.flex_attention.create_block_mask.
            output_attention (bool | None): Whether to return attention weights.
                Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - output (torch.Tensor): Transformed hidden state with same shape as input
                - attention_weights (torch.Tensor | None): Attention weights if requested,
                  None otherwise. Shape [batch_size, seq_len, seq_len]

        """
        # Attention component
        residual = x
        x = self.pre_norm_attn(x)
        x, w = self.attn(x, attention_mask, output_attention)
        x = self.attn_drop(x)
        x = self.post_norm_attn(x + residual)
        # Feed-forward component
        residual = x
        x = self.pre_norm_ffwd(x)
        x = self.ffwd(x)
        x = self.hidden_drop(x)
        x = self.post_norm_ffwd(x + residual)
        return x, w


class PolyBertEncoder(nn.Module):
    """Multi-layer transformer encoder for PolyBert.

    This class stacks multiple PolyBertBlock instances to create a deep transformer
    encoder. It handles sequence packing for efficient processing of variable-length
    sequences and supports outputting intermediate hidden states and attention weights.

    The encoder uses sequence packing to handle batches with sequences of different
    lengths efficiently, reducing computational overhead from padding tokens.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert encoder.

        Creates a stack of transformer blocks. Each block is independently initialized with the same configuration.

        Args:
            config (PolyBertConfig): Configuration object containing:
                - num_blocks: Number of transformer layers in the model
                - num_attention_heads: Number of attention heads (used for block mask creation)

        """
        super().__init__()
        self.blocks = nn.ModuleList([PolyBertBlock(config) for _ in range(config.num_blocks)])
        self.num_heads = config.num_attention_heads

    def forward(
        self,
        x: "torch.FloatTensor",
        block_mask: "BlockMask | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]":
        """Forward pass through the multi-layer transformer encoder.

        Processes input through all transformer blocks. Creates document-level block masks to
        prevent attention across document boundaries.

        Args:
            x (torch.FloatTensor, shape [batch_size, seq_len, hidden_size]): Input embeddings tensor.
            block_mask (BlockMask| None, optional): Flex-attention block mask indicating which tokens
                should attend to each other (inferred from attention_mask).
            output_attentions (bool | None, optional): Whether to return attention weights from
                all layers. Defaults to False.
            output_hidden_states (bool | None, optional): Whether to return hidden states from
                all layers. Defaults to False.

        Returns:
            tuple containing:
                - last_hidden_state (torch.Tensor): Output of the final transformer layer.
                  Shape [batch_size, seq_len, hidden_size].
                - all_hidden_states (tuple[torch.Tensor, ...] | None): Hidden states from all layers
                  if output_hidden_states=True, None otherwise. Each tensor has shape
                  [batch_size, seq_len, hidden_size].
                - all_attentions (tuple[torch.Tensor, ...] | None): Attention weights from all layers
                  if output_attentions=True, None otherwise. Shape depends on attention implementation.

        """
        all_attentions = []
        all_hidden_states = [x]
        # Apply layers
        for block in self.blocks:
            x, w = block(x, block_mask, output_attentions)
            if output_attentions:
                all_attentions.append(w)
            if output_hidden_states:
                all_hidden_states.append(x)
        return (
            x,
            tuple(all_hidden_states) if output_hidden_states else None,
            tuple(all_attentions) if output_attentions else None,
        )
