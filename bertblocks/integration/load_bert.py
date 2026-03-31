import torch
from transformers import BertModel

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.model import BertBlocksModel


def from_huggingface_bert_model(
    pretrained_model_name_or_path: str, load_weights: bool = True, add_pooling_layer: bool = False
) -> BertBlocksModel:
    """Instantiate an equivalent BertBlocks model from pretrained HuggingFace BERT weights and config.

    Converts a HuggingFace BERT model to BertBlocks architecture with optional weight transfer.
    The BertBlocks model uses post-normalization and standard MLP architecture to match BERT.

    Args:
        pretrained_model_name_or_path (str): HuggingFace model identifier or local path to a
            pretrained BERT model (e.g., "bert-base-uncased", "./path/to/model").
        load_weights (bool, optional): Whether to transfer weights from the pretrained BERT model.
            If True, copies all embeddings, attention, feed-forward, and normalization layer weights.
            If False, only loads the configuration and initializes a fresh model. Defaults to True.
        add_pooling_layer (bool, optional): Whether to add a pooling layer that processes the
            [CLS] token. Useful for sequence-level classification tasks. Defaults to False.

    Returns:
        BertBlocksModel: A BertBlocks model with architecture matched to BERT, optionally
            loaded with pretrained weights.

    Raises:
        ValueError: If the model config cannot be loaded or if the model type is not BERT.
        OSError: If the model path does not exist or is not accessible.

    Note:
        - The weight transfer is exact and lossless; no approximation is used.
        - All layer parameters (embeddings, QKV projections, feed-forward, norms) are copied directly.
        - The pooling layer (if added) is initialized with new random weights.

    Example:
        >>> from bertblocks.integration import from_bert_model
        >>> # Load and convert a pretrained BERT model
        >>> model = from_bert_model("bert-base-uncased", load_weights=True)
        >>> # Or load just the config without weights for a fresh model
        >>> model = from_bert_model("bert-base-uncased", load_weights=False)

    References:
        - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
          (https://arxiv.org/abs/1810.04805)
    """
    bertblocks_config = BertBlocksConfig.from_huggingface_bert(pretrained_model_name_or_path)
    bertblocks_model = BertBlocksModel(bertblocks_config, add_pooling_layer=add_pooling_layer)

    if load_weights:
        orig_model = BertModel.from_pretrained(pretrained_model_name_or_path, add_pooling_layer=add_pooling_layer)
        bertblocks_model = from_bert_model(orig_model, add_pooling_layer=add_pooling_layer)

    return bertblocks_model


def from_bert_model(orig_model: BertModel, add_pooling_layer: bool = False) -> BertBlocksModel:
    """Instantiate an equivalent BertBlocks model from a HuggingFace BERT model instance.

    Converts a HuggingFace BERT model to BertBlocks architecture with weight transfer.
    The BertBlocks model uses post-normalization and standard MLP architecture to match BERT.

    Args:
        orig_model (BertModel): An instance of a HuggingFace BertModel that has been
            loaded with pretrained weights.
        add_pooling_layer (bool, optional): Whether to add a pooling layer that processes the
            [CLS] token. Defaults to False.

    Returns:
        BertBlocksModel: A BertBlocks model with architecture matched to BERT,
            loaded with pretrained weights.

    References:
        - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
          (https://arxiv.org/abs/1810.04805)
    """
    bertblocks_config = BertBlocksConfig.from_config(orig_model.config)
    bertblocks_model = BertBlocksModel(bertblocks_config, add_pooling_layer=add_pooling_layer)

    # Embedding layer
    bertblocks_model.embd.embd.weight.data.copy_(orig_model.embeddings.word_embeddings.weight.data)
    bertblocks_model.embd.pose.embd.weight.data.copy_(orig_model.embeddings.position_embeddings.weight.data)
    bertblocks_model.embd.norm.weight.data.copy_(orig_model.embeddings.LayerNorm.weight.data)
    bertblocks_model.embd.norm.bias.data.copy_(orig_model.embeddings.LayerNorm.bias.data)
    bertblocks_model.embd.tokt.embd.weight.data.copy_(orig_model.embeddings.token_type_embeddings.weight.data)  # type: ignore

    for i in range(len(bertblocks_model.encd.blocks)):
        # QKV Projection
        qkv_weight = torch.cat(
            [
                orig_model.encoder.layer[i].attention.self.query.weight,
                orig_model.encoder.layer[i].attention.self.key.weight,
                orig_model.encoder.layer[i].attention.self.value.weight,
            ],
            dim=0,
        )
        qkv_bias = torch.cat(
            [
                orig_model.encoder.layer[i].attention.self.query.bias,
                orig_model.encoder.layer[i].attention.self.key.bias,
                orig_model.encoder.layer[i].attention.self.value.bias,
            ],
            dim=0,
        )

        bertblocks_model.encd.blocks[i].attn.proj.weight.data.copy_(qkv_weight.data)
        bertblocks_model.encd.blocks[i].attn.proj.bias.data.copy_(qkv_bias.data)

        # Attention output projection
        bertblocks_model.encd.blocks[i].attn.ffwd.weight.data.copy_(
            orig_model.encoder.layer[i].attention.output.dense.weight.data
        )
        bertblocks_model.encd.blocks[i].attn.ffwd.bias.data.copy_(
            orig_model.encoder.layer[i].attention.output.dense.bias.data
        )

        # Feed-forward layers
        bertblocks_model.encd.blocks[i].ffwd.uprj.weight.data.copy_(
            orig_model.encoder.layer[i].intermediate.dense.weight.data
        )
        bertblocks_model.encd.blocks[i].ffwd.uprj.bias.data.copy_(
            orig_model.encoder.layer[i].intermediate.dense.bias.data
        )
        bertblocks_model.encd.blocks[i].ffwd.dprj.weight.data.copy_(
            orig_model.encoder.layer[i].output.dense.weight.data
        )
        bertblocks_model.encd.blocks[i].ffwd.dprj.bias.data.copy_(orig_model.encoder.layer[i].output.dense.bias.data)

        # Layer norms
        bertblocks_model.encd.blocks[i].post_norm_attn.weight.data.copy_(
            orig_model.encoder.layer[i].attention.output.LayerNorm.weight.data
        )
        bertblocks_model.encd.blocks[i].post_norm_attn.bias.data.copy_(
            orig_model.encoder.layer[i].attention.output.LayerNorm.bias.data
        )
        bertblocks_model.encd.blocks[i].post_norm_ffwd.weight.data.copy_(
            orig_model.encoder.layer[i].output.LayerNorm.weight.data
        )
        bertblocks_model.encd.blocks[i].post_norm_ffwd.bias.data.copy_(
            orig_model.encoder.layer[i].output.LayerNorm.bias.data
        )

    return bertblocks_model
