from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertblocks.modeling.model import BertBlocksModel

from transformers import AutoConfig

from .load_bert import from_bert_model
from .load_modernbert import from_modernbert_model


def from_huggingface(
    pretrained_model_name_or_path: str, load_weights: bool = True, add_pooling_layer: bool = False
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

    Returns:
        BertBlocksModel: A BertBlocks model with architecture matched to the source HuggingFace model,
            optionally loaded with pretrained weights.

    Raises:
        ValueError: If the model type is not supported or cannot be detected.
        OSError: If the model path does not exist or is not accessible.
        ImportError: If required HuggingFace transformers models are not installed.

    Supported Models:
        - BERT and variants (bert-base-uncased, bert-large-uncased, etc.)
        - ModernBERT (modernbert-base, modernbert-large, etc.)
        - Other BERT-like encoder models compatible with HuggingFace

    Example:
        >>> from bertblocks.integration import from_huggingface
        >>> # Load BERT model
        >>> bert_model = from_huggingface("bert-base-uncased")
        >>> # Load ModernBERT model
        >>> modernbert_model = from_huggingface("modernbert-base")
        >>> # Load without transferring weights
        >>> fresh_model = from_huggingface("bert-base-uncased", load_weights=False)

    References:
        - HuggingFace Model Hub: https://huggingface.co/models
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
