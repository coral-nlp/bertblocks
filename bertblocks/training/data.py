import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
    from transformers import PreTrainedTokenizerBase

from datasets import load_dataset
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
    **kwargs: dict[str, Any],
) -> "HFDataset | HFIterableDataset":
    """Load a dataset, either from disk or from Huggingface Hub.

    Args:
        dataset_name_or_path: Path or name of dataset to load.
        name: Dataset configuration name.
        split: Dataset split to load.
        file_format: Dataset file format to load when loading from local folder.
        streaming: Whether to stream data (return an IterableDataset).
        add_index: Whether to add an explicit index column.
        **kwargs: Additional keyword arguments to pass to load_dataset/load_from_disk.

    Returns:
        Processed dataset ready for training.
    """
    # Load dataset; if we have a special JSON file it might still be a HF dataset, just cached locally
    if os.path.isdir(dataset_name_or_path) and not os.path.join(dataset_name_or_path, "dataset_info.json"):
        dataset = load_dataset(
            file_format or "json",
            data_dir=dataset_name_or_path,
            split=split,
            streaming=streaming,
            **kwargs,
        )
    elif os.path.isdir(dataset_name_or_path) and os.path.exists(
        os.path.join(dataset_name_or_path, "dataset_info.json")
    ):
        warnings.warn("Loading HF dataset from disk; this ignores streaming parameters!", stacklevel=1)
        dataset = datasets.load_from_disk(dataset_name_or_path, **kwargs)
    else:
        dataset = load_dataset(
            dataset_name_or_path,
            name=name,
            split=split,
            streaming=streaming,
            **kwargs,
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


def _truncated_add_length(batch: dict[str, list[Any]], max_sequence_length: int = 512) -> dict[str, list[Any]]:
    input_ids = [ids[:max_sequence_length] for ids in batch["input_ids"]]
    result: dict[str, list[Any]] = {"input_ids": input_ids, "length": [len(ids) for ids in input_ids]}
    if "attention_mask" in batch:
        result["attention_mask"] = [mask[:max_sequence_length] for mask in batch["attention_mask"]]
    return result


__all__ = [
    "EmptyDataset",
    "_load_dataset",
    "_tokenize_batch",
    "_truncated_add_length",
]
