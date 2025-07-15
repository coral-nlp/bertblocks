from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.attention import PolyBertAttention
from polybert.modeling.config import ConfigMixin
from polybert.modeling.mlp import get_mlp
from polybert.modeling.norms import get_norm
from polybert.modeling.packing import unpack_seq, pack_seq


class PolyBertBlock(nn.Module, ConfigMixin):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
        self.attn = PolyBertAttention(config)
        self.ffwd = get_mlp(config)
        self.pre_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()

    def forward(
        self, 
        x: "torch.Tensor",
        cu_seqlens: "torch.LongTensor | None" = None,
        output_attention: "bool | None" = False
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward function for single transformer block.
        Supports both pre- and post-norm flows as described in https://arxiv.org/pdf/2002.04745
        Pre and post norm may be identity functions as set in constructor.
        """
        # Attention component
        residual = x
        x = self.pre_norm(x)
        x, w = self.attn(x, cu_seqlens, output_attention)
        x = self.post_norm(x + residual)
        # Linear component
        residual = x
        x = self.pre_norm(x)
        x = self.ffwd(x)
        x = self.post_norm(x + residual)
        return x, w


class PolyBertEncoder(nn.Module, ConfigMixin):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__(config)
        self.blocks = nn.ModuleList([PolyBertBlock(config) for _ in range(config.num_hidden_layers)])
        self.num_heads = self.config.num_attention_heads

    def forward(
        self,
        x: "torch.FloatTensor",
        attention_mask: "torch.Tensor | None" = None,
        output_attentions: "bool | None" = False,
        output_hidden_states: "bool | None" = False,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None]":
        all_attentions = []
        all_hidden_states = [x]
        B = x.shape[0]
        x, indices, cu_seqlens, max_seq_len = pack_seq(x, attention_mask)
        for block in self.blocks:
            x, w = block(x, cu_seqlens, output_attentions)
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