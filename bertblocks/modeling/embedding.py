from typing import TYPE_CHECKING

from bertblocks.modeling.norms import get_norm

if TYPE_CHECKING:
    from bertblocks.modeling.config import BertBlocksConfig

import torch
from torch import nn

from bertblocks.modeling.position import LearnedPositionalEncoding, SinusoidalPositionalEncoding


class BertBlocksTokenTypeEmbedding(nn.Module):
    """Token type embeddings."""

    def __init__(self, config: "BertBlocksConfig"):
        """Instantiate token type embedding module.

        Args:
            config (BertBlocksConfig): Configuration object containing:
                - type_vocab_size: Size of the token type vocabulary
                - hidden_size: Dimensionality of embeddings and hidden states
                - max_sequence_length: Maximum sequence length for token type buffer

        """
        super().__init__()
        self.embd = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def forward(self, x: "torch.Tensor", token_type_ids: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Add token type embeddings to the hidden state.

        Uses supplied token type ids if given, otherwise defaults to constant token type ids.

        Args:
            x (torch.Tensor, shape [total_sequence_length, hidden_size]): Hidden state to add token type ids to.
            token_type_ids (torch.Tensor, shape [total_sequence_length], optional): Indicates the token type of
                each token in the sequence.

        Returns:
            torch.Tensor: Hidden state with token type embedding added.
                Shape [total_sequence_length, hidden_size].

        """
        token_type_ids = (
            token_type_ids if token_type_ids is not None else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        )
        return x + self.embd(token_type_ids)


class BertBlocksEmbeddings(nn.Module):
    """Token embedding layer for BertBlocks model.

    This class implements the token embedding layer that converts input token IDs
    to dense vector representations. Optionally applies sinusoidal positional encodings
    and dropout for regularization.
    """

    def __init__(self, config: "BertBlocksConfig"):
        """Initialize the embedding layer.

        Args:
            config (BertBlocksConfig): Configuration object containing:
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
                self.pose = SinusoidalPositionalEncoding(dim=config.hidden_size, max_seq_len=config.max_sequence_length)
            case "learned":
                self.pose = LearnedPositionalEncoding(dim=config.hidden_size, max_seq_len=config.max_sequence_length)
            case _:
                self.pose = None  # type: ignore
        self.tokt = BertBlocksTokenTypeEmbedding(config) if config.add_token_type_emb else None
        # Only post norm needed, as input is not a dense representation
        self.norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.drop = nn.Dropout(config.emb_dropout_prob) if config.emb_dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        input_ids: "torch.LongTensor",
        cu_seqlens: "torch.LongTensor | None" = None,
        token_type_ids: "torch.LongTensor | None" = None,
    ) -> "torch.Tensor":
        """Forward pass through the embedding layer.

        Args:
            input_ids (torch.LongTensor, shape [batch_size, sequence_length]): Token IDs to embed.
            cu_seqlens (torch.Tensor, shape [batch_size + 1,], optional): Cumulative sequence lengths.
            token_type_ids (torch.LongTensor, shape [batch_size, sequence_length], optional): Tensor of token types.

        Returns:
            torch.Tensor: Embedded token representations with optional dropout applied.
            Shape [batch_size, sequence_length, hidden_size].

        """
        # Input embeddings
        x = self.embd(input_ids)
        # Token type embeddings (optional)
        if self.tokt is not None:
            x = self.tokt(x, token_type_ids=token_type_ids)
        # Additive positional encoding (optional)
        if self.pose is not None:
            x = self.pose(x, cu_seqlens=cu_seqlens)
        # Normalization (optional)
        x = self.norm(x)
        # Dropout (optional)
        x = self.drop(x)
        return x
