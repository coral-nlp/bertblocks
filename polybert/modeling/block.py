import warnings
from typing import TYPE_CHECKING

from transformers.modeling_utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn.bert_padding import pad_input, unpad_input
else:
    raise ImportError("This implementation currently critically depends on flash_attn. ")

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

    def __init__(self, config: "PolyBertConfig", layer_id: int):
        """Initialize a PolyBert transformer block.

        Sets up the attention mechanism, feed-forward network, and normalization
        layers based on configuration. Normalization layers are set to Identity
        when not needed according to the norm_kind setting.

        Args:
            config (PolyBertConfig): Configuration object containing:
                - norm_kind: When to apply normalization ("pre", "post", "both", "none")
                - attn_dropout_prob: Dropout probability for attention weights
                - hidden_dropout_prob: Dropout probability for hidden layer outputs
            layer_id (int): layer id indicating index in the encoder stack.

        """
        super().__init__()
        self.attn = PolyBertAttention(config, layer_id=layer_id)
        self.ffwd = get_mlp(config)
        self.pre_norm_attn = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.pre_norm_ffwd = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm_attn = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.post_norm_ffwd = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.attn_drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()
        self.hidden_drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        x: "torch.Tensor",
        cu_seqlens: "torch.Tensor",
        max_seq_len: int,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Forward pass through the transformer block.

        Processes input through attention and feed-forward layers with residual
        connections. Supports both pre-normalization and post-normalization schemes.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): The hidden state of
                the previous transformer block.
            cu_seqlens (torch.Tensor, shape [batch_size + 1]): Cumulative sequence lengths of batch.
            max_seq_len (int): Maximum sequence length of batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - output (torch.Tensor): Transformed hidden state with same shape as input
                - attention_weights (torch.Tensor | None): Attention weights. Shape [batch_size, seq_len, seq_len]

        """
        # Attention component
        residual = x
        x = self.pre_norm_attn(x)
        x, w = self.attn(x, cu_seqlens, max_seq_len)
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

    Uses sequence packing.
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
        self.blocks = nn.ModuleList([PolyBertBlock(config, layer_id) for layer_id in range(config.num_blocks)])
        self.num_heads = config.num_attention_heads

    def forward(
        self,
        x: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]":
        """Forward pass through the multi-layer transformer encoder.

        Processes input through all transformer blocks. Creates document-level block masks to
        prevent attention across document boundaries.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): Input embeddings tensor.
            attention_mask (torch.Tensor | None, optional): Binary mask indicating which tokens
                should be attended to.
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
        if output_attentions:
            warnings.warn("Returning attentions is currently not supported, will return None.", stacklevel=2)
            output_attentions = False

        # Unpad input sequence
        B, S, _ = x.shape
        x, indices, cu_seqlens, max_seq_len, _ = unpad_input(x, attention_mask=attention_mask)

        # Keep track of states throughout block stack
        all_attentions = []
        all_hidden_states = [x]
        # Apply layers
        for block in self.blocks:
            x, w = block(x, cu_seqlens, max_seq_len)
            if output_attentions:
                all_attentions.append(w)
            if output_hidden_states:
                all_hidden_states.append(x)

        x = pad_input(x, indices, B, S)
        if output_hidden_states:
            all_hidden_states = [pad_input(h, indices, B, S) for h in all_hidden_states]

        return (
            x,
            tuple(all_hidden_states) if output_hidden_states else None,
            tuple(all_attentions) if output_attentions else None,
        )
