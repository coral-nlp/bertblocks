import torch
from einops import rearrange
from torch import nn
from transformers.modeling_utils import is_flash_attn_2_available

from polybert.modeling.config import PolyBertConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.modules.mha import get_alibi_slopes

    # Otherwise triggers graph break
    torch._dynamo.config.capture_scalar_outputs = True
else:
    raise ImportError("This implementation currently critically depends on flash_attn. ")


class PolyBertAttention(nn.Module):
    """PolyBERT attention with configurable positional encodings.

    This class implements a flexible attention mechanism using flex_attention for efficient
    computation. Applies block masking for document-level attention patterns.

    The attention mechanism follows the standard transformer architecture but
    with configurable positional biases that can be applied either to the
    query-key projections or as score modifications.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        proj: Fused linear projection for Q, K, V (3 * hidden_size output).
        ffwd: Output projection layer.
        dropout_p: Dropout probability for attention weights.

    """

    def __init__(self, config: "PolyBertConfig"):
        """Initialize the PolyBERT attention mechanism.

        Args:
            config (PolyBertConfig): Configuration object containing:
                - num_attention_heads: Number of attention heads in multi-head attention
                - hidden_size: Dimensionality of hidden layers (must be divisible by num_attention_heads)
                - max_sequence_length: Maximum sequence length for positional encodings
                - attn_proj_bias: Whether to include bias in QKV projection
                - attn_out_bias: Whether to include bias in output projection
                - attn_dropout_prob: Dropout probability for attention weights
                - pos_emb_kind: Type of positional embedding ("alibi", "rope", "relative", etc.)

        """
        super().__init__()
        # General hyperparameters
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_seq_len = config.max_sequence_length
        # Fused linear layers for better performance
        self.proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attn_proj_bias)
        self.ffwd = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attn_out_bias)
        self.dropout_p = config.attn_dropout_prob
        self._initialize_pos_buffers(config)

    def _initialize_pos_buffers(self, config: PolyBertConfig) -> None:
        """Initialize positional encoding buffers if needed.

        Args:
            config (PolyBertConfig): Model config.

        """
        self.rotary_emb = None
        self.slopes = None
        match config.pos_emb_kind:
            case "alibi":
                self.slopes = nn.Parameter(
                    torch.tensor(get_alibi_slopes(config.num_attention_heads), dtype=torch.float32)
                )
            case "rope":
                self.rotary_emb = RotaryEmbedding(
                    dim=config.pos_emb_kwargs["dim"],
                    base=config.pos_emb_kwargs["base"],
                    scale_base=config.pos_emb_kwargs["scale"],
                    # If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
                    interleaved=config.pos_emb_kwargs["interleaved"],
                )
            case _:
                pass

    def forward(
        self,
        x: "torch.Tensor",
        cu_seqlens: "torch.Tensor",
        max_seq_len: int,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass of the PolyBERT attention mechanism.

        Computes multi-head self-attention with configurable positional encodings
        and block masking. Uses PyTorch's flex_attention.

        Args:
            x (torch.Tensor, shape [batch_size, seq_len, hidden_size]): The input hidden state.
            cu_seqlens (torch.Tensor, shape [batch_size + 1]): Cumulative sequence lengths of batch.
            max_seq_len (int): Maximum sequence length for positional encodings.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - output: Attention output tensor of shape [batch_size, seq_len, hidden_size].
                - attention_weights: Log-sum-exp attention weights of shape [batch_size, num_heads, seq_len, seq_len].

        """
        # Fused projection
        qkv = self.proj(x)
        # Reshape for multihead attention
        qkv = rearrange(qkv, "... (t h d) -> ... t h d", t=3, h=self.num_heads, d=self.head_dim)
        if self.rotary_emb is not None:
            qkv = self.rotary_emb(qkv, seqlen_offset=0, max_seqlen=None)
        orig_dtype = qkv.dtype
        qkv = qkv.to(torch.bfloat16)
        # Apply attention kernel
        x, _, w = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens.to(torch.int32),
            max_seq_len,
            dropout_p=self.dropout_p if self.training else 0.0,
            # softmax_scale=self.config.scale,
            causal=False,
            # window_size=self.window_size,
            softcap=0.0,  # 0.0 means deactivated
            alibi_slopes=self.slopes,
            # deterministic=self.deterministic,
            return_attn_probs=True,
        )
        # Reshape back
        x = x.to(orig_dtype)
        x = rearrange(x, "... h d -> ... (h d)")
        # Output projection
        x = self.ffwd(x)
        return x, w
