from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn


class PolyBertEmbeddings(nn.Module):
    def __init__(self, config: "PolyBertConfig"):
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()

    def forward(
        self, 
        input_ids: "torch.LongTensor",
    ) -> "torch.Tensor":
        x = self.embd(input_ids)
        x = self.drop(x)
        return x
