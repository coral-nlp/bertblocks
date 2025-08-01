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
            config (PolyBertConfig): Configuration object containing:
                - type_vocab_size: Size of the token type vocabulary
                - hidden_size: Dimensionality of embeddings and hidden states
                - max_sequence_length: Maximum sequence length for token type buffer

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
            x (torch.Tensor, shape [batch_size, sequence_length, hidden_size]): Hidden state to add token type ids to.
            token_type_ids (torch.Tensor, shape [batch_size, sequence_length], optional): Indicates the token type of
                each token in the sequence.

        Returns:
            torch.Tensor: Hidden state with token type embedding added.
                Shape [batch_size, sequence_length, hidden_size].

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
            config (PolyBertConfig): Configuration object containing:
                - vocab_size: Size of the vocabulary for token embeddings
                - hidden_size: Dimensionality of embeddings and hidden states
                - pad_token_id: Token ID used for padding sequences
                - pos_emb_kind: Type of positional embedding ("sinusoidal", "learned", etc.)
                - max_sequence_length: Maximum sequence length for positional encodings
                - add_token_type_emb: Whether to add token type embeddings
                - norm_kind: When to apply normalization ("post", "both", etc.)
                - emb_dropout_prob: Dropout probability for embedding layer output

        """
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        match config.pos_emb_kind:
            case "sinusoidal":
                self.pose = SinusoidalPositionalEncoding(**config.pos_emb_kwargs)
            case "learned":
                self.pose = LearnedPositionalEncoding(**config.pos_emb_kwargs)
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
            input_ids (torch.LongTensor, shape [batch_size, sequence_length]): Token IDs to embed.

        Returns:
            torch.Tensor: Embedded token representations with optional dropout applied.
            Shape [batch_size, sequence_length, hidden_size].

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
