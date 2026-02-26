from typing import Literal

from bertblocks.config import ModernBertConfig
from bertblocks.modeling.model import BertBlocksModel


def from_modernbert_model(
    pretrained_model_name_or_path: str,
    load_weights: bool = True,
    add_pooling_layer: bool = False,
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
) -> "BertBlocksModel":
    """Instantiate an equivalent BertBlocks model from pretrained HuggingFace ModernBERT weights and config.

    Converts a HuggingFace ModernBERT model to BertBlocks architecture with optional weight transfer.
    ModernBERT uses ALiBi positional encodings, GLU feed-forward layers, and supports local attention,
    which are all supported by BertBlocks.

    Args:
        pretrained_model_name_or_path (str): HuggingFace model identifier or local path to a
            pretrained ModernBERT model (e.g., "modernbert-base", "./path/to/model").
        load_weights (bool, optional): Whether to transfer weights from the pretrained ModernBERT model.
            If True, copies all embeddings, attention, feed-forward, and normalization layer weights.
            If False, only loads the configuration and initializes a fresh model. Defaults to True.
        add_pooling_layer (bool, optional): Whether to add a pooling layer that processes the
            [CLS] token. Useful for sequence-level classification tasks. Defaults to False.
        attn_implementation (Literal["flash_attention_2", "sdpa", "eager"], optional):
            Attention implementation backend to use:
            - "flash_attention_2": Use FlashAttention-2 for faster inference (requires flash-attn package)
            - "sdpa": Use PyTorch's scaled-dot-product attention (default, recommended for most cases)
            - "eager": Use manual attention implementation (slower, for compatibility)
            Defaults to "flash_attention_2".

    Returns:
        BertBlocksModel: A BertBlocks model with architecture matched to ModernBERT, optionally
            loaded with pretrained weights.

    Raises:
        ValueError: If the model config cannot be loaded or if the model type is not ModernBERT.
        OSError: If the model path does not exist or is not accessible.

    Note:
        - The weight transfer is exact and lossless; no approximation is used.
        - All layer parameters (embeddings, QKV projections, GLU layers, norms) are copied directly.
        - The pooling layer (if added) is initialized with new random weights.
        - Final normalization layer weights are transferred if included in the model.

    Example:
        >>> from bertblocks.integration import from_modernbert_model
        >>> # Load and convert a pretrained ModernBERT model with FlashAttention
        >>> model = from_modernbert_model("modernbert-base", load_weights=True)
        >>> # Load with SDPA backend for broader compatibility
        >>> model = from_modernbert_model("modernbert-base", attn_implementation="sdpa")

    References:
        - "Smashing Language Barriers with Multilingual Transformers"
          (https://arxiv.org/abs/2406.07581)
        - "ModernBERT" (https://github.com/AnswerDotAI/ModernBERT)
    """
    from transformers import ModernBertModel

    bertblocks_config = ModernBertConfig.from_huggingface(
        pretrained_model_name_or_path, attn_implementation=attn_implementation
    )
    bertblocks_model = BertBlocksModel(bertblocks_config, add_pooling_layer=add_pooling_layer)

    if load_weights:
        orig_model = ModernBertModel.from_pretrained(pretrained_model_name_or_path)
        # Embedding layer
        bertblocks_model.embd.embd.weight.data.copy_(orig_model.embeddings.tok_embeddings.weight.data)

        for i in range(len(bertblocks_model.encd.blocks)):
            # QKV Projection
            bertblocks_model.encd.blocks[i].attn.proj.weight.data.copy_(orig_model.layers[i].attn.Wqkv.weight.data)
            if bertblocks_config.attn_proj_bias:
                bertblocks_model.encd.blocks[i].attn.proj.bias.data.copy_(orig_model.layers[i].attn.Wqkv.bias.data)
            # Attention output projection
            bertblocks_model.encd.blocks[i].attn.ffwd.weight.data.copy_(orig_model.layers[i].attn.Wo.weight.data)
            if bertblocks_config.attn_out_bias:
                bertblocks_model.encd.blocks[i].attn.ffwd.bias.data.copy_(orig_model.layers[i].attn.Wo.bias.data)
            # Feed-forward up and down projection
            bertblocks_model.encd.blocks[i].ffwd.uprj.weight.data.copy_(orig_model.layers[i].mlp.Wi.weight.data)
            if bertblocks_config.mlp_in_bias:
                bertblocks_model.encd.blocks[i].ffwd.uprj.bias.data.copy_(orig_model.layers[i].mlp.Wi.bias.data)
            bertblocks_model.encd.blocks[i].ffwd.dprj.weight.data.copy_(orig_model.layers[i].mlp.Wo.weight.data)
            if bertblocks_config.mlp_out_bias:
                bertblocks_model.encd.blocks[i].ffwd.dprj.bias.data.copy_(orig_model.layers[i].mlp.Wo.bias.data)
            # Norms
            if i == 0:
                # If first layer, the norm is in the embedding (pre-norm)
                bertblocks_model.encd.blocks[i].pre_norm_attn.weight.data.copy_(orig_model.embeddings.norm.weight.data)
                if bertblocks_config.norm_bias:
                    bertblocks_model.encd.blocks[i].pre_norm_attn.bias.data.copy_(orig_model.embeddings.norm.bias.data)
            else:
                bertblocks_model.encd.blocks[i].pre_norm_attn.weight.data.copy_(
                    orig_model.layers[i].attn_norm.weight.data
                )
                if bertblocks_config.norm_bias:
                    bertblocks_model.encd.blocks[i].pre_norm_attn.bias.data.copy_(
                        orig_model.layers[i].attn_norm.bias.data
                    )

            bertblocks_model.encd.blocks[i].pre_norm_ffwd.weight.data.copy_(orig_model.layers[i].mlp_norm.weight.data)
            if bertblocks_config.norm_bias:
                bertblocks_model.encd.blocks[i].pre_norm_ffwd.bias.data.copy_(orig_model.layers[i].mlp_norm.bias.data)

            bertblocks_model.norm.weight.data.copy_(orig_model.final_norm.weight.data)
            if bertblocks_config.norm_bias:
                bertblocks_model.norm.bias.data.copy_(orig_model.final_norm.bias.data)

    return bertblocks_model
