from typing import TYPE_CHECKING

from polybert.modeling.norms import get_norm

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.position import SinusoidalPositionalEncoding


class PolyBertEmbeddings(nn.Module):
    """Token embedding layer for PolyBert poly_model.

    This class implements the token embedding layer that converts input token IDs
    to dense vector representations. Optionally applies sinusoidal positional encodings
    and dropout for regularization.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the embedding layer.

        Args:
            config (PolyBertConfig): Configuration object determining poly_model hyperparameters.

        """
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.penc = (
            SinusoidalPositionalEncoding(config.hidden_size, config.max_sequence_length)
            if config.pos_emb_kind == "sinusoidal"
            else nn.Identity()
        )
        # Only post norm needed, as input is not a dense representation
        self.post_norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.drop = nn.Dropout(config.emb_dropout_prob) if config.emb_dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        input_ids: "torch.LongTensor",
    ) -> "torch.Tensor":
        """Forward pass through the embedding layer.

        Args:
            input_ids: Token IDs to embed. Shape: (batch_size, sequence_length)

        Returns:
            Embedded token representations with optional dropout applied.
            Shape: (batch_size, sequence_length, hidden_size)

        """
        x = self.embd(input_ids)
        x = self.penc(x)
        x = self.drop(x)
        return x
