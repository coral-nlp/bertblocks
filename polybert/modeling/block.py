from typing import TYPE_CHECKING

from torch.nn.attention.flex_attention import BlockMask, create_block_mask

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.attention import PolyBertAttention
from polybert.modeling.config import ConfigMixin
from polybert.modeling.mlp import get_mlp
from polybert.modeling.norms import get_norm
from polybert.modeling.packing import doc_mask, pack_seq, unpack_seq


class PolyBertBlock(nn.Module, ConfigMixin):
    """A single transformer block implementation for PolyBert.

    This class implements a standard transformer block with attention and feed-forward
    layers, supporting both pre-normalization and post-normalization schemes.

    The block consists of:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Optional layer normalization (pre/post/both/none)

    Args:
        config (PolyBertConfig): Configuration object containing model hyperparameters.
            Must include norm_kind to determine normalization placement.

    References:
         - "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/pdf/2002.04745).

    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize a PolyBert transformer block.

        Sets up the attention mechanism, feed-forward network, and normalization
        layers based on the configuration. Normalization layers are set to Identity
        when not needed according to the norm_kind setting.

        Args:
            config (PolyBertConfig): Configuration object containing:
                - norm_kind: Normalization placement ("pre", "post", "both", "none")
                - Other hyperparameters for attention and MLP layers

        """
        super().__init__(config=config)
        self.attn = PolyBertAttention(config)
        self.ffwd = get_mlp(config)
        self.pre_norm = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(
        self, x: "torch.Tensor", block_mask: "BlockMask", output_attention: "bool | None" = False
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass through the transformer block.

        Processes input through attention and feed-forward layers with residual
        connections. Supports both pre-normalization and post-normalization schemes.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size)
                or (packed_seq_len, hidden_size) if sequence packing is used.
            block_mask (BlockMask): Block mask for efficient attention computation,
                typically created using torch.nn.attention.flex_attention.create_block_mask.
            output_attention (bool | None, optional): Whether to return attention weights.
                Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - output (torch.Tensor): Transformed tensor with same shape as input
                - attention_weights (torch.Tensor | None): Attention weights if requested,
                  None otherwise. Shape depends on attention implementation.

        Note:
            The forward pass follows the standard transformer architecture:
            1. x = post_norm(pre_norm(x) + attention(pre_norm(x)))
            2. x = post_norm(pre_norm(x) + ffwd(pre_norm(x)))

            The normalization functions may be identity operations based on configuration.

        """
        # Attention component
        residual = x
        x = self.pre_norm(x)
        x, w = self.attn(x, block_mask, output_attention)
        x = self.post_norm(x + residual)
        # Linear component
        residual = x
        x = self.pre_norm(x)
        x = self.ffwd(x)
        x = self.post_norm(x + residual)
        return x, w


class PolyBertEncoder(nn.Module, ConfigMixin):
    """Multi-layer transformer encoder for PolyBert.

    This class stacks multiple PolyBertBlock instances to create a deep transformer
    encoder. It handles sequence packing for efficient processing of variable-length
    sequences and supports outputting intermediate hidden states and attention weights.

    The encoder uses sequence packing to handle batches with sequences of different
    lengths efficiently, reducing computational overhead from padding tokens.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBert encoder.

        Creates a stack of transformer blocks and stores configuration parameters.
        Each block is independently initialized with the same configuration.

        Args:
            config (PolyBertConfig): Configuration object containing model hyperparameters.

        """
        super().__init__(config=config)
        self.blocks = nn.ModuleList([PolyBertBlock(config) for _ in range(config.num_blocks)])
        self.num_heads = self.config.num_attention_heads

    def forward(
        self,
        x: "torch.FloatTensor",
        attention_mask: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]":
        """Forward pass through the multi-layer transformer encoder.

        Processes input through all transformer blocks with sequence packing for
        efficiency. Creates document-level block masks to prevent attention across
        document boundaries in packed sequences.

        Args:
            x (torch.FloatTensor): Input embeddings tensor of shape (batch_size, seq_len, hidden_size).
                Each sequence in the batch may have different lengths when using attention_mask.
            attention_mask (torch.Tensor | None, optional): Boolean mask indicating which tokens
                should attend to each other. Shape (batch_size, seq_len) where True means the token
                should be attended to. None means all tokens attend to each other. Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights from
                all layers. Defaults to False.
            output_hidden_states (bool | None, optional): Whether to return hidden states from
                all layers. Defaults to False.

        Returns:
            tuple containing:
                - last_hidden_state (torch.Tensor): Output of the final transformer layer.
                  Shape (batch_size, seq_len, hidden_size).
                - all_hidden_states (tuple[torch.Tensor, ...] | None): Hidden states from all layers
                  if output_hidden_states=True, None otherwise. Each tensor has shape
                  (batch_size, seq_len, hidden_size).
                - all_attentions (tuple[torch.Tensor, ...] | None): Attention weights from all layers
                  if output_attentions=True, None otherwise. Shape depends on attention implementation.

        Note:
            This method uses sequence packing internally for efficiency:
            1. Sequences are packed into a single tensor to remove padding
            2. Document block masks prevent cross-document attention
            3. Results are unpacked back to original batch format

        """
        all_attentions = []
        all_hidden_states = [x]
        B, S = x.shape
        x, indices, cu_seqlens, max_seq_len = pack_seq(x, attention_mask)
        # Create document block mask to reuse throughout all layers for this batch
        block_mask = create_block_mask(doc_mask(cu_seqlens), None, None, S, S, _compile=True)
        # Apply layers
        for block in self.blocks:
            x, w = block(x, block_mask, output_attentions)
            if output_attentions:
                all_attentions.append(w)
            if output_hidden_states:
                all_hidden_states.append(unpack_seq(x, indices, B, max_seq_len))
        x = unpack_seq(x, indices, B, max_seq_len)
        return (
            x,
            tuple(all_hidden_states) if output_hidden_states else None,
            tuple(all_attentions) if output_attentions else None,
        )
