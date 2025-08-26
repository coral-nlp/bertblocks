from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from bertblocks.config import BertBlocksConfig

from torch import nn

from bertblocks.modeling.attention import Attention
from bertblocks.modeling.mlp import get_mlp
from bertblocks.modeling.norms import get_norm


class Block(nn.Module):
    """A single transformer block.

    Implements a standard transformer block with attention and feed-forward
    layers, supporting both pre-normalization and post-normalization schemes.

    The block consists of:

        - Multi-head self-attention with residual connection
        - Feed-forward network with residual connection
        - Layer normalization (pre/post/both/none)

    Attributes:
        layer_id (int): index position of the layer in the models' encoder stack.
        attn (Attention): Attention module.
        ffwd (nn.Module): Feed-forward module.
        pre_norm_attn (nn.Module): Pre-normalization layer for attention module. Falls back to `nn.Identity` if not
            configured.
        pre_norm_ffwd (nn.Module): Pre-normalization layer for feed-forward module. Falls back to `nn.Identity` if not
            configured.
        post_norm_attn (nn.Module): Pre-normalization function for attention module. Falls back to `nn.Identity` if not
            configured.
        post_norm_ffwd (nn.Module): Post-normalization function for feed-forward module. Falls back to `nn.Identity` if
            not configured.
        attn_drop (nn.Dropout): Post-attention dropout layer. Falls back to `nn.Identity` if not configured.
        ffwd_drop (nn.Dropout): Post-Feed-forward dropout layer. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `norm_kind`: Normalization layer type
                - `attn_dropout_prob`: Dropout probability for attention layer
                - `hidden_dropout_prob`: Dropout probability for feed-forward layers

        layer_id (int): layer id indicating index in the encoder stack.


    References:
         - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)
         - "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/pdf/2002.04745)

    """

    def __init__(self, config: "BertBlocksConfig", layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(config, layer_id=layer_id)
        self.ffwd = get_mlp(config)
        self.pre_norm_attn = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.pre_norm_ffwd = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm_attn = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.post_norm_ffwd = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.attn_drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()
        self.ffwd_drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        x: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
        cu_seqlens: "torch.Tensor | None" = None,
        max_seq_len: int | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Hidden state (unpadded or padded).
            attention_mask (torch.Tensor | None): Attention mask (for padded sequences).
            cu_seqlens (torch.Tensor | None): Cumulative sequence lengths (for unpadded sequences).
            max_seq_len (int | None): Maximum sequence length (for unpadded sequences).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:

                - `output` (torch.Tensor): Transformed hidden state with same shape as input
                - `attention_weights` (torch.Tensor | None): Attention weights

        """
        # Attention component
        if self.layer_id == 0:
            x = self.pre_norm_attn(x)
            residual = x
        else:
            residual = x
            x = self.pre_norm_attn(x)
        x, w = self.attn(x, attention_mask, cu_seqlens, max_seq_len)
        x = self.attn_drop(x)
        x = self.post_norm_attn(x + residual)
        # Feed-forward component
        residual = x
        x = self.pre_norm_ffwd(x)
        x = self.ffwd(x)
        x = self.ffwd_drop(x)
        x = self.post_norm_ffwd(x + residual)
        return x, w


class Encoder(nn.Module):
    """Multi-layer transformer encoder.

    Uses sequence packing for higher efficiency.

    Attributes:
        blocks (nn.ModuleList): Stack of Block modules.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `num_blocks`: Number of transformer blocks
                - `num_attention_heads`: Number of transformer attention heads

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config, layer_id) for layer_id in range(config.num_blocks)])

    def forward(
        self,
        x: "torch.Tensor",
        attention_mask: "torch.Tensor | None",
        cu_seqlens: "torch.Tensor | None",
        max_seq_len: int | None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]":
        """Forward pass of the encoder.

        Processes input hidden state sequentially through all transformer blocks.

        Args:
            x (torch.Tensor): Hidden state (unpadded or padded).
            attention_mask (torch.Tensor | None): Attention mask (for padded sequences).
            cu_seqlens (torch.Tensor | None): Cumulative sequence lengths (for unpadded sequences).
            max_seq_len (int | None): Maximum sequence length (for unpadded sequences).
            output_attentions (bool | None, optional): Whether to return attention weights from
                all layers. Defaults to False.
            output_hidden_states (bool | None, optional): Whether to return hidden states from
                all layers. Defaults to False.

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]:

                - `last_hidden_state` (torch.Tensor): Output of the final transformer layer.
                - `all_hidden_states` (tuple[torch.Tensor, ...] | None): Hidden states from all layers
                  if output_hidden_states=True, None otherwise.
                - `all_attentions` (tuple[torch.Tensor, ...] | None): Attention weights from all layers
                  if output_attentions=True, None otherwise.

        """
        # Keep track of states throughout block stack
        all_attentions = []
        all_hidden_states = [x]
        # Apply layers
        for block in self.blocks:
            x, w = block(x, attention_mask, cu_seqlens, max_seq_len)
            if output_hidden_states:
                all_hidden_states.append(x)
            if output_attentions:
                all_attentions.append(w)

        return (
            x,
            tuple(all_hidden_states) if output_hidden_states else None,
            tuple(all_attentions) if output_attentions else None,
        )


__all__ = ["Encoder", "Block"]
