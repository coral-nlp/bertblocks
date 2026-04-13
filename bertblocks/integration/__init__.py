from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from bertblocks.modeling.model import BertBlocksModel

from transformers import AutoConfig, BertModel, PreTrainedModel

from .load_bert import from_bert_model, from_huggingface_bert_model
from .load_modernbert import from_huggingface_modernbert_model, from_modernbert_model


def from_huggingface(
    pretrained_model_name_or_path: str,
    load_weights: bool = True,
    add_pooling_layer: bool = False,
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa",
) -> "BertBlocksModel":
    """Instantiate an equivalent BertBlocksModel from HuggingFace pretrained models.

    Automatically detects the model type and routes to the appropriate conversion function.
    Supports BERT-like encoder models available on HuggingFace Hub.

    Args:
        pretrained_model_name_or_path (str): HuggingFace model identifier (e.g., "bert-base-uncased",
            "modernbert-base") or local path to a pretrained model directory.
        load_weights (bool, optional): Whether to transfer weights from the pretrained HuggingFace model.
            If True, copies all layer parameters. If False, only loads the configuration and initializes
            a fresh model with random weights. Defaults to True.
        add_pooling_layer (bool, optional): Whether to add a pooling layer that processes the [CLS] token.
            Useful for sequence-level classification tasks. Defaults to False.
        attn_implementation (str, optional): Attention backend. One of "flash_attention_2", "sdpa", or
            "eager". Defaults to "sdpa".

    Returns:
        BertBlocksModel: A BertBlocks model with architecture matched to the source HuggingFace model,
            optionally loaded with pretrained weights.

    Raises:
        ValueError: If the model type is not supported or cannot be detected.
    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    match config.model_type:
        case "bert":
            return from_huggingface_bert_model(
                pretrained_model_name_or_path,
                load_weights=load_weights,
                add_pooling_layer=add_pooling_layer,
                attn_implementation=attn_implementation,
            )
        case "modernbert":
            return from_huggingface_modernbert_model(
                pretrained_model_name_or_path,
                load_weights=load_weights,
                add_pooling_layer=add_pooling_layer,
                attn_implementation=attn_implementation,
            )
        case _:
            raise ValueError(f"Unsupported model_type: {config.model_type}. Supported types: bert, modernbert")


def from_model(
    model: PreTrainedModel,
    add_pooling_layer: bool = False,
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa",
) -> "BertBlocksModel":
    """Instantiate a BertBlocksModel from an already loaded HuggingFace model instance.

    Automatically detects the model type and dispatches to the appropriate conversion function.

    Args:
        model (PreTrainedModel): An instance of a HuggingFace PreTrainedModel.
        add_pooling_layer (bool, optional): Whether to add a pooling layer. Defaults to False.
        attn_implementation (str, optional): Attention backend. One of "flash_attention_2", "sdpa", or
            "eager". Defaults to "sdpa".

    Returns:
        BertBlocksModel: A BertBlocks model with architecture and weights matched to the source model.

    Raises:
        ValueError: If the model type is not supported or cannot be detected.
    """
    from transformers import ModernBertModel

    if isinstance(model, BertModel):
        return from_bert_model(model, add_pooling_layer=add_pooling_layer, attn_implementation=attn_implementation)
    if isinstance(model, ModernBertModel):
        return from_modernbert_model(
            model, add_pooling_layer=add_pooling_layer, attn_implementation=attn_implementation
        )
    raise ValueError(f"Unsupported model type: {type(model).__name__}. Supported types: BertModel, ModernBertModel")


__all__ = ["from_huggingface", "from_model"]
