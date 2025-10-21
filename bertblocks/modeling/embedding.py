import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from bertblocks.modeling.norms import get_norm

if TYPE_CHECKING:
    from bertblocks.config import BertBlocksConfig

import torch
from torch import nn

from bertblocks.modeling.position import LearnedPositionalEncoding, SinusoidalPositionalEncoding


class MultiHotEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int = 0, **kwargs: Any) -> None:
        super().__init__()
        max_power_2 = math.ceil(math.log2(vocab_size))
        bit_positions = torch.arange(max_power_2, dtype=torch.long).unsqueeze(0)
        self.register_buffer("bit_positions", bit_positions)
        self.embd = nn.Linear(max_power_2, hidden_size)
        self.norm = nn.RMSNorm(hidden_size)

        self.to_multihot: Callable[[torch.LongTensor], torch.LongTensor] = self._to_binary_repr
        match kwargs.get("encoding"):
            case "binary":
                self.to_multihot = self._to_binary_repr
            case "gray":
                self.to_multihot = self._to_gray_repr
            case _:
                warnings.warn(
                    "Unknown value or no value for 'encoding' in inp_emb_kwargs, defaulting to 'binary'}", stacklevel=2
                )

    def _to_binary_repr(self, x: "torch.LongTensor") -> "torch.LongTensor":
        return (x.to(dtype=torch.long).unsqueeze(-1) >> self.bit_positions) & 1  # [seq_len, max_power_of_2]

    def _to_gray_repr(self, x: "torch.LongTensor") -> "torch.LongTensor":
        # Convert int IDs to binary
        x = self._to_binary_repr(x)  # [seq_len, max_power_of_2]
        # Convert binary to Gray code: out[i] = x[i+1] XOR x[i], with out[MSB] = x[MSB]
        x[..., :-1] = x[..., 1:] ^ x[..., :-1]
        return x

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # Input IDs to multi hot representation
        x = self.to_multihot(x).to(dtype=self.embd.weight.dtype)  # [..., max_power_of_2]
        # Multi hot representation to dense
        x = self.embd(x)  # [..., intermediate_size]
        x = self.norm(x)
        return x


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
        token_type_ids = (
            token_type_ids if token_type_ids is not None else torch.zeros(shape, dtype=torch.long, device=x.device)
        )
        return x + self.embd(token_type_ids)


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
            - `pos_emb_kind`: Type of positional embedding ("sinusoidal", "learned", etc.)
            - `max_sequence_length`: Maximum sequence length for positional encodings
            - `add_token_type_emb`: Whether to add token type embeddings
            - `norm_kind`: When to apply normalization ("post", "both", etc.)
            - `emb_dropout_prob`: Dropout probability for embedding layer output

    """

    def __init__(self, config: "BertBlocksConfig"):
        super().__init__()
        match config.inp_emb_kind:
            case "onehot":
                self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            case "multihot":
                self.embd = MultiHotEmbedding(
                    config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, **config.inp_emb_kwargs
                )

        match config.pos_emb_kind:
            case "sinusoidal":
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
        """Forward pass of the embedding layer.

        Args:
            input_ids (torch.Tensor, shape [total_seq_len,]): Unpadded token IDs.
            cu_seqlens (torch.Tensor, shape [batch_size + 1]): Cumulative sequence lengths in batch.
            token_type_ids (torch.LongTensor, shape [batch_size, sequence_length], optional): Tensor of token types.

        Returns:
            torch.Tensor: Embedded token representations, shape [total_seq_len, hidden_size].

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
