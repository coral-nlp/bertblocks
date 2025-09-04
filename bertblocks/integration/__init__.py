from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertblocks.modeling.model import BertBlocksModel

from transformers import AutoConfig

from .load_bert import from_bert_model
from .load_modernbert import from_modernbert_model


def from_huggingface(
    pretrained_model_name_or_path: str, load_weights: bool = True, add_pooling_layer: bool = False
) -> "BertBlocksModel":
    """Instantiate an equivalent BertBlocksModel from Huggingface models.

    Automatically chooses the correct loader function based on model type.

    Args:
        pretrained_model_name_or_path (str): Path to Huggingface pretrained model or model identifier.
        load_weights (bool, optional): Whether to transfer the Huggingface model weights to the instantiated
            BertBlocks model. Defaults to True.
        add_pooling_layer (bool, optional): Whether to add a pooling layer to the BertBlocks model. Defaults to False.

    """
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    match config.model_type:
        case "modernbert":
            return from_modernbert_model(
                pretrained_model_name_or_path, load_weights=load_weights, add_pooling_layer=add_pooling_layer
            )
        case _:
            raise ValueError(f"Unknown model_type {config.model_type}")


__all__ = ["from_huggingface"]
