from typing import TYPE_CHECKING

from bertblocks.modeling.norms import get_norm

if TYPE_CHECKING:
    from bertblocks.config import BertBlocksConfig

import torch
from torch import nn

from bertblocks.modeling.position import LearnedPositionalEncoding, SinusoidalPositionalEncoding


class TokenTypeEmbedding(nn.Module):
    """Token type embedding layer.

    Implements the token type embedding layer that converts token type IDs to dense vector representations.

    Attributes:
        embd (nn.Embedding): Token type embedding layer.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `type_vocab_size`: Size of the token type vocabulary
                - `hidden_size`: Dimensionality of embeddings and hidden states

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.embd = nn.Embedding(config.type_vocab_size, config.hidden_size)

    def forward(self, x: "torch.Tensor", token_type_ids: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Forward pass of the token type embeddings.

        Uses supplied token type ids if given, otherwise defaults to constant token type ids.

        Args:
            x (torch.Tensor, shape [total_seq_len, hidden_size] or [batch_size, seq_len, hidden_size]): Hidden state
                to add token type ids to.
            token_type_ids (torch.Tensor, shape [total_seq_len, hidden_size] or [batch_size, seq_len, hidden_size],
                optional): Indicates the token type of each token in the sequence.

        Returns:
            torch.Tensor: Hidden state with token type embedding added, shape [total_seq_len, hidden_size] or
                [batch_size, seq_len, hidden_size].

        """
        shape = x.shape[:2] if len(x.shape) == 3 else x.shape[:1]
        token_type_ids = token_type_ids if token_type_ids is not None else torch.zeros(shape, device=x.device)
        return x + self.embd(token_type_ids.to(torch.int32))


class TokenEmbedding(nn.Module):
    """Token embedding layer.

    Implements the token embedding layer that converts input token IDs to dense vector representations.
    Optionally applies positional encodings and/or token type encodings.

    Attributes:
        embd (nn.Embedding): Token type embedding layer.
        pose (nn.Module | None): Positional encoding layer.
        tokt (nn.Module | None): Token type embedding layer.
        norm (nn.Module): Normalization layer. Falls back to `nn.Identity` if not configured.
        drop (nn.Dropout): Dropout layer. Falls back to `nn.Identity` if not configured.

    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

            - `vocab_size` (int): Size of the vocabulary for token embeddings
            - `hidden_size`: Dimensionality of embeddings and hidden states
            - `pad_token_id`: Token ID used for padding sequences
            - `emb_pos_enc_kind`: Type of positional encoding ("sinusoidal", "learned", etc.)
            - `max_sequence_length`: Maximum sequence length for positional encodings
            - `add_token_type_emb`: Whether to add token type embeddings
            - `norm_kind`: When to apply normalization ("post", "both", etc.)
            - `emb_dropout_prob`: Dropout probability for embedding layer output

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        match config.emb_pos_enc_kind:
            case "sinusoidal":
                # TODO: emb_pos_enc_kwargs should be used to make "base" configurable
                self.pose = SinusoidalPositionalEncoding(dim=config.hidden_size, max_seq_len=config.max_sequence_length)
            case "learned":
                self.pose = LearnedPositionalEncoding(dim=config.hidden_size, max_seq_len=config.max_sequence_length)
            case _:
                self.pose = None  # type: ignore
        self.tokt = TokenTypeEmbedding(config) if config.add_token_type_emb else None
        # Only post norm needed, as input is not a dense representation
        self.norm = get_norm(config) if config.norm_kind in ("post", "both") else nn.Identity()
        self.drop = nn.Dropout(config.emb_dropout_prob) if config.emb_dropout_prob > 0 else nn.Identity()

    def forward(
        self,
        input_ids: "torch.LongTensor",
        cu_seqlens: "torch.LongTensor | None" = None,
        token_type_ids: "torch.LongTensor | None" = None,
    ) -> "torch.Tensor":
        """Forward pass of the token embedding layer.

        Combines token embeddings, optional token type embeddings, optional positional encodings,
        normalization, and dropout.

        Args:
            input_ids (torch.Tensor, shape [batch_size, seq_len] or [total_seq_len]):
                Token IDs to embed. For padded inputs, shape is [batch_size, seq_len].
                For unpadded inputs, shape is [total_seq_len].
            cu_seqlens (torch.Tensor, shape [batch_size + 1], optional):
                Cumulative sequence lengths for unpadded sequences. Used by positional encodings
                to compute per-sequence position indices. Defaults to None.
            token_type_ids (torch.Tensor, shape [batch_size, seq_len] or [total_seq_len], optional):
                Segment IDs indicating token type (e.g., 0 for sentence A, 1 for sentence B in NSP).
                Only used if `add_token_type_emb` is True in config. Defaults to None.

        Returns:
            torch.Tensor: Embedded token representations with shape:
                - [batch_size, seq_len, hidden_size] for padded inputs
                - [total_seq_len, hidden_size] for unpadded inputs

        References:
            - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
              (https://arxiv.org/abs/1810.04805)
            - "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762)
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


__all__ = ["TokenEmbedding", "TokenTypeEmbedding"]
