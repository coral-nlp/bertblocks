from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from bertblocks.modeling.norms import get_norm
from bertblocks.modeling.scale import LayerScaler

if TYPE_CHECKING:
    from torch import Tensor

    from bertblocks.config import BertBlocksConfig

import torch
from torch import nn

from bertblocks.modeling.attention import Attention
from bertblocks.modeling.mlp import get_mlp


def convert_to_4d_attention_mask(attention_mask: "Tensor") -> "Tensor":
    """Convert a 2D attention mask to 4D.

    Args:
        attention_mask (Tensor, shape [batch_size, seq_length]): The input attention mask.

    Returns:
        Tensor: The converted 4D attention mask.
    """
    attention_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
    attention_mask = attention_mask.unsqueeze(1)
    return attention_mask


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

        layer_id (int): zero-indexed layer id indicating index in the encoder stack.


    References:
         - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)
         - "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/pdf/2002.04745)
         - "The Curse of Depth in Large Language Models" (https://arxiv.org/pdf/2502.05795)
    """

    def __init__(self, config: "BertBlocksConfig", layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(config, layer_id=layer_id)
        self.ffwd = get_mlp(config)
        self.norm_kind = config.norm_kind
        self.pre_norm_attn = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.pre_norm_ffwd = get_norm(config) if config.norm_kind in ("pre", "both") else nn.Identity()
        self.post_norm_attn = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.post_norm_ffwd = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.pre_scaler = (
            LayerScaler(layer_id) if config.norm_scaling and config.norm_kind in ("pre", "both") else nn.Identity()
        )
        self.post_scaler = (
            LayerScaler(layer_id) if config.norm_scaling and config.norm_kind in ("post", "both") else nn.Identity()
        )
        self.attn_drop = nn.Dropout(config.attn_dropout_prob) if config.attn_dropout_prob > 0 else nn.Identity()
        self.ffwd_drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()
        self.residual_first_layer = config.residual_first_layer

    def forward(
        self,
        x: "Tensor",
        attention_mask: "Tensor | None" = None,
        cu_seqlens: "Tensor | None" = None,
        max_seq_len: int | None = None,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass of the transformer block.

        Args:
            x (Tensor): Hidden state (unpadded or padded).
            attention_mask (Tensor | None): Attention mask (for padded sequences).
            cu_seqlens (Tensor | None): Cumulative sequence lengths (for unpadded sequences).
            max_seq_len (int | None): Maximum sequence length (for unpadded sequences).

        Returns:
            tuple[Tensor, Tensor | None]:

                - `output` (Tensor): Transformed hidden state with same shape as input
                - `attention_weights` (Tensor | None): Attention weights

        """
        # Attention component
        if self.layer_id == 0 and self.residual_first_layer:
            x = self.pre_norm_attn(x)
            x = self.pre_scaler(x)
            residual = x
        else:
            residual = x
            x = self.pre_norm_attn(x)
            x = self.pre_scaler(x)
        x, w = self.attn(x, attention_mask, cu_seqlens, max_seq_len)
        x = self.attn_drop(x)
        x = self.post_norm_attn(x + residual)
        x = self.post_scaler(x)
        # Feed-forward component
        residual = x
        x = self.pre_norm_ffwd(x)
        x = self.pre_scaler(x)
        x = self.ffwd(x)
        x = self.ffwd_drop(x)
        # Without explicit cast, torch.compile breaks with AMP for some reason?
        x = self.post_norm_ffwd(x + residual.to(dtype=x.dtype))
        x = self.post_scaler(x)
        return x, w


class EnhancedMaskingBlock(Block):
    """A single transformer block.

    Implements an enhanced masking transformer block which allows for custom
    modifications of the attention mask.

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
        masking_strategy (str): Masking strategy to use.
            Available options: "random".
        masking_probability (float): Probability of masking tokens. Defaults to 0.5.

    References:
         - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)
         - "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/pdf/2002.04745)
         - Paper for enhanced masking?

    """

    def __init__(
        self,
        config: "BertBlocksConfig",
        layer_id: int,
        masking_strategy: Literal["random"],
        masking_probability: float = 0.5,
    ):
        config = deepcopy(config)
        config.attn_implementation = "sdpa"
        super().__init__(config, layer_id)
        self.masking_strategy = masking_strategy
        self.masking_probability = masking_probability

    def forward(
        self,
        x: "Tensor",
        attention_mask: "Tensor | None" = None,
        cu_seqlens: "Tensor | None" = None,
        max_seq_len: int | None = None,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass of the transformer block.

        Args:
            x (Tensor): Hidden state (unpadded or padded).
            attention_mask (Tensor | None): Attention mask (for padded sequences).
            cu_seqlens (Tensor | None): Cumulative sequence lengths (for unpadded sequences).
            max_seq_len (int | None): Maximum sequence length (for unpadded sequences).

        Returns:
            tuple[Tensor, Tensor | None]:

                - `output` (Tensor): Transformed hidden state with same shape as input
                - `attention_weights` (Tensor | None): Attention weights

        """
        attention_mask = torch.ones(x.shape[:2], dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        attention_mask = convert_to_4d_attention_mask(attention_mask)
        if attention_mask is None:
            raise RuntimeError("Attention mask is required for an enhanced masking block.")
        match self.masking_strategy:
            case "random":
                attention_mask = self._apply_random_mask(attention_mask)
            case _:
                raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")
        # set diagonal to 0 so that a token cannot attend to itself
        attention_mask[..., range(attention_mask.shape[-2]), range(attention_mask.shape[-1])] = 0
        return super().forward(x, attention_mask, cu_seqlens, max_seq_len)

    def _apply_random_mask(self, attention_mask: "Tensor") -> "Tensor":
        """Apply random masking to the attention mask.

        Args:
            attention_mask (Tensor, shape [batch_size, seq_len]): The original attention mask.

        Returns:
            Tensor: The modified attention mask with random masking applied.
        """
        attention_mask = attention_mask & (
            torch.rand(attention_mask.shape, device=attention_mask.device) > self.masking_probability
        )
        return attention_mask


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
        x: "Tensor",
        attention_mask: "Tensor | None",
        cu_seqlens: "Tensor | None",
        max_seq_len: int | None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "tuple[Tensor, tuple[Tensor, ...] | None, tuple[Tensor, ...] | None]":
        """Forward pass of the encoder.

        Processes input hidden state sequentially through all transformer blocks.

        Args:
            x (Tensor): Hidden state (unpadded or padded).
            attention_mask (Tensor | None): Attention mask (for padded sequences).
            cu_seqlens (Tensor | None): Cumulative sequence lengths (for unpadded sequences).
            max_seq_len (int | None): Maximum sequence length (for unpadded sequences).
            output_attentions (bool | None, optional): Whether to return attention weights from
                all layers. Defaults to False.
            output_hidden_states (bool | None, optional): Whether to return hidden states from
                all layers. Defaults to False.

        Returns:
            tuple[Tensor, tuple[Tensor, ...] | None, tuple[Tensor, ...] | None]:

                - `last_hidden_state` (Tensor): Output of the final transformer layer.
                - `all_hidden_states` (tuple[Tensor, ...] | None): Hidden states from all layers
                  if output_hidden_states=True, None otherwise.
                - `all_attentions` (tuple[Tensor, ...] | None): Attention weights from all layers
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


__all__ = ["Block", "Encoder"]
