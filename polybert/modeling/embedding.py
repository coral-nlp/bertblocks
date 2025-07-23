from typing import TYPE_CHECKING

from polybert.modeling.norms import get_norm

if TYPE_CHECKING:
    from polybert.modeling.config import PolyBertConfig

import torch
from torch import nn

from polybert.modeling.position import LearnedPositionalEncoding, SinusoidalPositionalEncoding


class PolyBertTokenTypeEmbedding(nn.Module):
    """Token type embeddings."""

    def __init__(self, config: "PolyBertConfig"):
        """Instantiate token type embedding module.

        Args:
            config (PolyBertConfig): PolyBERT model config to instantiate token type embeddings with.

        """
        super().__init__()
        self.embd = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.register_buffer(
            "token_type_ids", torch.zeros(config.max_sequence_length, dtype=torch.long), persistent=False
        )

    def forward(self, x: "torch.Tensor", token_type_ids: "torch.Tensor" = None) -> "torch.Tensor":
        """Add token type embeddings to the hidden state.

        Uses supplied token type ids if given, otherwise defaults to constant token type ids.

        Args:
            x: Tensor, shape: (batch_size, sequence_length, hidden_size)
                Hidden state to add token type ids to.
            token_type_ids: Tensor, shape: (batch_size, sequence_length)
                Indicates the token type of each token in the sequence.

        Returns:
            Hidden state with token type embedding added.
            Shape: (batch_size, sequence_length, hidden_size)

        """
        if token_type_ids is not None:
            return x + self.embd(token_type_ids)
        return x + self.embd(self.token_type_ids)


class PolyBertEmbeddings(nn.Module):
    """Token embedding layer for PolyBert model.

    This class implements the token embedding layer that converts input token IDs
    to dense vector representations. Optionally applies sinusoidal positional encodings
    and dropout for regularization.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the embedding layer.

        Args:
            config (PolyBertConfig): Configuration object determining model hyperparameters.

        """
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        match config.pos_emb_kind:
            case "sinusoidal":
                self.pose = SinusoidalPositionalEncoding(config.hidden_size, config.max_sequence_length)
            case "learned":
                self.pose = LearnedPositionalEncoding(config.hidden_size, config.max_sequence_length)
            case _:
                self.pose = nn.Identity()
        self.tokt = PolyBertTokenTypeEmbedding(config) if config.add_token_type_emb else nn.Identity()
        # Only post norm needed, as input is not a dense representation
        self.norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
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
        # Input embeddings
        x = self.embd(input_ids)
        # Token type embeddings (optional)
        x = self.tokt(x)
        # Additive positional encoding (optional)
        x = self.pose(x)
        # Normalization (optional)
        x = self.norm(x)
        # Dropout (optional)
        x = self.drop(x)
        return x
