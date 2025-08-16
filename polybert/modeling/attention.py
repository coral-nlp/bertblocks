import torch
from einops import rearrange
from torch import nn
from transformers.modeling_utils import is_flash_attn_2_available

from polybert.modeling.config import PolyBertConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_qkvpacked_func

    # Otherwise triggers graph break
    torch._dynamo.config.capture_scalar_outputs = True


from polybert.modeling.position import RotaryEmbedding, get_alibi_slopes


def flash_attention_forward(
    qkv: "torch.Tensor",
    cu_seqlens: "torch.Tensor",
    max_seq_len: int,
    num_heads: int,
    rotary_emb: "RotaryEmbedding | None" = None,
    alibi_slopes: "torch.Tensor | None" = None,
    local_attention: tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
    deterministic: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Forward pass for flash attention backend."""
    if rotary_emb is not None:
        qkv = rotary_emb(qkv, cu_seqlens, max_seq_len)
    orig_dtype = qkv.dtype
    qkv = qkv.to(torch.bfloat16)
    x, _, w = flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens.to(torch.int32),
        max_seq_len,
        dropout_p=dropout_p,
        causal=False,
        softcap=0.0,  # 0.0 means deactivated
        window_size=local_attention,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    x = x.to(orig_dtype)
    return x, w


ATTENTION_FUNCTION = {
    "fa2": flash_attention_forward,
}


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

    def __init__(self, config: "PolyBertConfig", layer_id: int):
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
            layer_id (int): layer id indicating index in the encoder stack.

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
        if config.global_attention_every_n_layers != 0:
            self.local_attention = (
                config.local_attention if layer_id % config.global_attention_every_n_layers != 0 else (-1, -1)
            )
        else:
            self.local_attention = (-1, -1)
        self._initialize_pos_buffers(config, layer_id=layer_id)
        self._attention_fn = ATTENTION_FUNCTION[config.attn_implementation]
        self.deterministic = True

    def _initialize_pos_buffers(self, config: PolyBertConfig, layer_id: int) -> None:
        """Initialize positional encoding buffers if needed.

        Args:
            config (PolyBertConfig): Model config.
            layer_id (int): layer id indicating index in the encoder stack.

        """
        self.rotary_emb = None
        self.slopes = None
        match config.pos_emb_kind:
            case "alibi":
                self.slopes = nn.Parameter(get_alibi_slopes(config.num_attention_heads))
            case "rope":
                match config.attn_implementation:
                    case "fa2":
                        if "base_global" in config.pos_emb_kwargs:
                            theta_global = config.pos_emb_kwargs["base_global"]
                        else:
                            theta_global = config.pos_emb_kwargs.get("base", 10_000)
                        if "base_local" in config.pos_emb_kwargs:
                            theta_local = config.pos_emb_kwargs["base_local"]
                        else:
                            theta_local = config.pos_emb_kwargs.get("base", 10_000)

                        self.rotary_emb = RotaryEmbedding(
                            dim=config.pos_emb_kwargs["dim"],
                            base=theta_local
                            if layer_id % config.global_attention_every_n_layers != 0
                            else theta_global,
                        )
                    case _:
                        raise NotImplementedError("Only flash attention is supported as backend for rotary encodings.")
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
        # Apply attention mixer
        x, w = self._attention_fn(
            qkv,
            cu_seqlens,
            max_seq_len,
            num_heads=self.num_heads,
            rotary_emb=self.rotary_emb,
            alibi_slopes=self.slopes,
            local_attention=self.local_attention,
            dropout_p=self.dropout_p if self.training else 0.0,
            deterministic=self.deterministic,
        )
        # Reshape back to fuse heads
        x = rearrange(x, "... h d -> ... (h d)")
        # Output projection
        x = self.ffwd(x)
        return x, w
