import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from datasets import Dataset, IterableDataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


class EmptyDataset(TorchDataset):
    """Empty dataset dummy to return when no data is loaded.

    https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset#70369304
    """

    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        """Return the number of examples in the dataset (i.e., empty list)."""
        return 0

    def __getitem__(self, index: int) -> None:
        """Pseudo-method to raise an error when trying to access an element of the empty dataset."""
        raise IndexError("Empty dataset cannot be indexed")


def _cache_paths(cache_dir: str | None) -> tuple[str | None, str | None, str | None]:
    train_cache = None
    val_cache = None
    test_cache = None

    if cache_dir is not None:
        train_cache = str(Path(cache_dir) / "train.arrow")
        val_cache = str(Path(cache_dir) / "val.arrow")
        test_cache = str(Path(cache_dir) / "test.arrow")

    return train_cache, val_cache, test_cache


def _load_dataset(
    dataset_name_or_path: str,
    name: str | None = None,
    split: str | None = None,
    file_format: str | None = None,
    streaming: bool = False,
    add_index: bool = False,
) -> "Dataset | IterableDataset":
    """Load a dataset, either from disk or from Huggingface Hub.

    Args:
        dataset_name_or_path: Path or name of dataset to load.
        split: Dataset split to load.
        add_index: Whether to add an explicit index column.

    Returns:
        Processed dataset ready for training.
    """
    # Load dataset
    if os.path.isdir(dataset_name_or_path):
        dataset = load_dataset(
            file_format or "json",
            data_dir=dataset_name_or_path,
            split=split,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_name_or_path,
            name=name,
            split=split,
            streaming=streaming,
        )
    # Add index column if packing (needed for restoring original order sometimes)
    if add_index and not streaming:
        dataset = dataset.add_column("_idx", list(range(len(dataset))))

    return dataset


def _tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: "PreTrainedTokenizerBase",
    text_column: str = "text",
    max_sequence_length: int = 512,
) -> dict[str, list[Any]]:
    res = tokenizer(
        batch[text_column or "text"],
        truncation=True,
        max_length=max_sequence_length,
        add_special_tokens=True,
        padding=False,
    )
    return {
        "input_ids": res["input_ids"],
        "attention_mask": res["attention_mask"],
        "length": [len(ids) for ids in res["input_ids"]],
    }


def _truncated_add_length(example: dict[str, Any], max_sequence_length: int = 512) -> dict[str, Any]:
    input_ids = example["input_ids"][:max_sequence_length]
    result = {"input_ids": input_ids, "length": len(input_ids)}
    if "attention_mask" in example:
        result["attention_mask"] = example["attention_mask"][:max_sequence_length]
    return result


__all__ = [
    "EmptyDataset",
    "_load_dataset",
    "_tokenize_batch",
    "_truncated_add_length",
]
