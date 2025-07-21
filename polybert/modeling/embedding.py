from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from polybert.modeling.config import PolyBertConfig

from torch import nn

from polybert.modeling.initialization import InitMixin


class PolyBertEmbeddings(nn.Module, InitMixin):
    """Token embedding layer for PolyBert model.

    This class implements the token embedding layer that converts input token IDs
    to dense vector representations. It includes optional dropout for regularization.
    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the embedding layer.

        Args:
            config: PolyBert configuration object containing:
                - vocab_size: Size of the vocabulary
                - hidden_size: Dimensionality of the embedding vectors
                - pad_token_id: Token ID used for padding
                - hidden_dropout_prob: Dropout probability (0 means no dropout)

        """
        super(InitMixin, self).__init__(config)
        self.embd = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.drop = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0 else nn.Identity()

    def init_weights(self) -> None:
        """Initialize weights."""
        """Initialize the embedding layer weights."""
        self._init_module_weights(self.embd, "embedding")

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
        x = self.drop(x)
        return x
