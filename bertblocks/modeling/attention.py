from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from torch import nn

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.backends import ATTENTION_BACKENDS
from bertblocks.modeling.position import RotaryPositionalEncoding, get_alibi_slopes


class Attention(nn.Module):
    """Attention with configurable positional encodings.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension size of attention heads.
        max_seq_len (int): Maximum sequence length.
        dropout_p (float): Dropout probability for attention.
        local_attention (tuple[int, int]): Local attention size, if applied.
        deterministic (bool): Whether to use deterministic attention.
        proj (nn.Linear): Fused QKV projection layer.
        ffwd (nn.Linear): Feed-forward layer to combine heads after attention.


    Args:
        config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
            other submodules. Keys used at top level:

                - `num_attention_heads`: Number of attention heads in multi-head attention
                - `hidden_size`: Dimensionality of hidden layers (must be divisible by num_attention_heads)
                - `max_sequence_length`: Maximum sequence length for positional encodings
                - `attn_proj_bias`: Whether to include bias in QKV projection
                - `attn_out_bias`: Whether to include bias in output projection
                - `attn_dropout_prob`: Dropout probability for attention weights
                - `pos_emb_kind`: Type of positional embedding ("alibi", "rope", "relative", etc.)

            layer_id (int): layer id indicating index in the encoder stack.
    """

    def __init__(self, config: "BertBlocksConfig", layer_id: int):
        super().__init__()
        # General hyperparameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_sequence_length
        self.dropout_p = config.attn_dropout_prob
        if config.global_attention_every_n_layers != 0:
            self.local_attention = (
                config.local_attention if layer_id % config.global_attention_every_n_layers != 0 else (-1, -1)
            )
        else:
            self.local_attention = (-1, -1)
        self.deterministic = True
        # Layers
        self.proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attn_proj_bias)
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        # Private inits
        self._initialize_pos_buffers(config, layer_id=layer_id)
        self.backend = ATTENTION_BACKENDS[config.attn_implementation]

    def _initialize_pos_buffers(self, config: "BertBlocksConfig", layer_id: int) -> None:
        """Initialize positional encoding buffers if needed.

        Args:
            config (BertBlocksConfig): Configuration object determining model hyperparameters. May be passed to
                other submodules. Keys used at top level:

                    - `attn_implementation`: Attention backend to use
                    - `pos_emb_kind`: Type of positional embedding ("alibi", "rope", "relative", etc.)
                    - `pos_emb_kwargs`: Additional positional encoding arguments
                    - `num_attention_heads`: Number of attention heads in multi-head attention
                    - `global_attention_every_n_layers`: Global attention layer stride

            layer_id (int): layer id indicating index in the encoder stack.

        """
        self.rotary_emb = None
        self.slopes = None
        match config.pos_emb_kind:
            case "alibi":
                self.slopes = nn.Parameter(get_alibi_slopes(config.num_attention_heads), requires_grad=False)
            case "learned_alibi":
                self.slopes = nn.Parameter(get_alibi_slopes(config.num_attention_heads), requires_grad=True)
            case "rope":
                match config.attn_implementation:
                    case "fa2":
                        interleaved = config.pos_emb_kwargs.get("interleaved", False)
                        if "base_global" in config.pos_emb_kwargs:
                            theta_global = config.pos_emb_kwargs["base_global"]
                        else:
                            theta_global = config.pos_emb_kwargs.get("base", 10_000)
                        if "base_local" in config.pos_emb_kwargs:
                            theta_local = config.pos_emb_kwargs["base_local"]
                        else:
                            theta_local = config.pos_emb_kwargs.get("base", 10_000)

                        if config.global_attention_every_n_layers == 0:
                            theta = theta_global
                        else:
                            theta = (
                                theta_local if layer_id % config.global_attention_every_n_layers != 0 else theta_global
                            )

                        self.rotary_emb = RotaryPositionalEncoding(
                            dim=config.pos_emb_kwargs["dim"],
                            base=theta,
                            interleaved=interleaved,
                            max_seq_len=self.max_seq_len,
                        )

                    case _:
                        raise NotImplementedError("Only flash attention is supported as backend for rotary encodings.")
            case _:
                pass

    def forward_unpadded(
        self,
        x: "Tensor",
        cu_seqlens: "Tensor",
        max_seq_len: int,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with unpadded sequences."""
        # Fused projection
        qkv = self.proj(x)
        # Rotary encoding if applicable
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv, self.num_heads, self.head_dim, cu_seqlens, max_seq_len)
        # Attention backend call
        x, w = self.backend.forward_unpadded(
            qkv,
            cu_seqlens,
            max_seq_len,
            self.num_heads,
            self.head_dim,
            alibi_slopes=self.slopes,
            local_attention=self.local_attention,
            dropout_p=self.dropout_p if self.training else 0.0,
            deterministic=self.deterministic,
        )
        # Fuse heads back together
        x = self.ffwd(x)
        return x, w

    def forward_padded(
        self,
        x: "Tensor",
        attention_mask: "Tensor",
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass with padded sequences."""
        # Fused projection
        qkv = self.proj(x)
        # Rotary encoding if applicable
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv, self.num_heads, self.head_dim)
        # Attention backend call
        x, w = self.backend.forward_padded(
            qkv,
            attention_mask,
            self.num_heads,
            self.head_dim,
            dropout_p=self.dropout_p if self.training else 0.0,
            deterministic=self.deterministic,
        )
        # Fuse heads back together
        x = self.ffwd(x)
        return x, w

    def forward(
        self,
        x: "Tensor",
        attention_mask: "Tensor | None" = None,
        cu_seqlens: "Tensor | None" = None,
        max_seq_len: int | None = None,
    ) -> "tuple[Tensor, Tensor | None]":
        """Forward pass of the attention mechanism.

        Automatically routes to padded or unpadded implementation based on backend capabilities.

        Args:
            x (torch.Tensor): Input hidden state
            indices (torch.Tensor, optional): Sequence indices for unpadded sequences
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for unpadded sequences
            max_seq_len (int, optional): Maximum sequence length for unpadded sequences
            attention_mask (torch.Tensor, optional): Attention mask for padded sequences

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Output and optional attention weights
        """
        if cu_seqlens is not None and max_seq_len is not None:
            return self.forward_unpadded(x, cu_seqlens, max_seq_len)
        elif attention_mask is not None:
            return self.forward_padded(x, attention_mask)
        else:
            raise ValueError(
                "Neither `attention_mask` nor `cu_seqlens` were provided, no attention implementation applicable"
            )


__all__ = ["Attention"]
