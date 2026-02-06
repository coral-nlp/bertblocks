"""Sequence packing using batch sampling and a wrapper around collators to produce full flat-packed batches."""

import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, Sampler


def is_packed_batch(attention_mask: "torch.Tensor | None") -> bool:
    """Detect if attention mask uses packed sequence index format.

    Packed format uses sequence indices (0, 1, 2, ...) with -1 for padding,
    while standard format uses binary 0/1 values.

    Args:
        attention_mask: Attention mask tensor to check, or None.

    Returns:
        True if packed format, False if standard format or None.
    """
    if attention_mask is None:
        return False
    # Packed format has values > 1 (sequence indices) or -1 (padding marker)
    return (attention_mask > 1).any() or (attention_mask < 0).any()  # type: ignore


class PackingBatchSampler(Sampler[list[int]]):
    """Batch sampler that greedily selects variable-length continuous runs of samples that fit inside the token budget.

    This sampler is supplied with a list of sequence lengths in tokens. In distributed mode, each rank independently
    samples from its assigned data shard, retaining DDP compatibility.

    Args:
        lengths: List of sequence lengths in tokens of the sampled dataset.
        token_budget: Maximum total tokens per batch (typically max_length * batch_size).
        shuffle: Whether to shuffle indices before packing. Defaults to False.
        drop_last: Drop the last incomplete batch. Defaults to False.
        rank: Rank for distributed training. If None, uses dist.get_rank() if available.
        world_size: World size for distributed training. If None, uses dist.get_world_size() if available.
        seed: Random seed for shuffle reproducibility.

    Example:

        >>> from datasets import load_dataset
        >>> from torch.utils.data import DataLoader
        >>> dataset = load_dataset("dataset_name", split="train")
        >>> dataset = dataset.map(...) # Add length column if not already present
        >>> batch_sampler = PackingBatchSampler(dataset, token_budget=4096)
        >>> collator = PackingCollatorWrapper(base_collator, token_budget=4096)
        >>> dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collator)

    """

    def __init__(
        self,
        lengths: list[int],
        token_budget: int,
        world_size: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        # Setup distributed training; fall back to single-GPU if dist not specified
        if dist.is_available() and dist.is_initialized():
            self.rank = rank if rank is not None else dist.get_rank()
            self.world_size = world_size if world_size is not None else dist.get_world_size()
        else:
            self.rank = rank if rank is not None else 0
            self.world_size = world_size if world_size is not None else 1

        self.token_budget = token_budget
        self.data = lengths

        self.epoch = 0
        self.drop_last = drop_last

        # How many samples this rank sees
        self.num_samples: int = math.ceil(len(self.data) / self.world_size)  # type: ignore[arg-type]
        # How many samples there are in the dataset
        self.total_size = len(self.data)  # type: ignore[arg-type]  # self.num_samples * self.world_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        """Provide an iterator over the subset of the dataset, yielding batches of sequences indices."""
        # Generate list of indices of this dataset
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        # Subsample the view of this batch
        indices = indices[self.rank : self.total_size : self.world_size]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        idx_ptr = 0
        while idx_ptr < len(indices):
            batch_indices: list[int] = []
            batch_tokens = 0

            # Fill the batch with continuous run of sequences
            while idx_ptr < len(indices):
                current_idx = indices[idx_ptr]
                seq_len = self.data[current_idx]

                # Always include at least one sequence per batch to ensure progress
                if len(batch_indices) == 0 or batch_tokens + seq_len <= self.token_budget:
                    # Fits (or first sequence), add to batch
                    batch_indices.append(current_idx)
                    batch_tokens += seq_len
                    idx_ptr += 1
                else:
                    # Doesn't fit, finalize this batch
                    break

            # Yield the batch if it has sufficient tokens (or still yield if not dropping last)
            if batch_indices and (not self.drop_last or batch_tokens >= self.token_budget * 0.5):
                yield batch_indices
            else:
                return

    def __len__(self) -> int:
        """Return the number of samples in this subset of the dataset."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class PackingIterableDataset(IterableDataset):
    """Wrapper for iterable datasets that packs continuous sequences into batches.

    This wrapper is designed to be used with PyTorch DataLoader for streaming datasets.
    Each iteration yields a list of samples that have been packed together based on
    their sequence lengths to maximize GPU utilization.

    The underlying dataset must include a length field (specified by length_column) in
    each sample. This can be added using dataset.map() before wrapping.

    In distributed training with num_workers > 0, sharding is handled automatically by
    PyTorch's DataLoader worker sharding mechanism for IterableDataset.

    Args:
        dataset: The underlying iterable dataset to wrap.
        token_budget: Maximum total tokens per batch (typically max_length * batch_size).
        length_column: Name of the field containing sequence length. Defaults to "length".
        drop_last: Drop the last incomplete batch. Defaults to False.

    Example:

        >>> from datasets import load_dataset
        >>> from torch.utils.data import DataLoader
        >>> dataset = load_dataset("dataset_name", streaming=True, split="train")
        >>> dataset = dataset.map(...) # Add length column if not already present
        >>> packed_dataset = PackingIterableDataset(dataset, token_budget=4096)
        >>> collator = PackingCollatorWrapper(base_collator, token_budget=4096)
        >>> dataloader = DataLoader(packed_dataset, batch_size=None, collate_fn=collator)

    """

    def __init__(
        self,
        dataset: IterableDataset,
        token_budget: int,
        length_column: str = "length",
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.token_budget = token_budget
        self.length_column = length_column
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        """Iterate over packed batches.

        Yields:
            List of samples that form a packed batch.
        """
        batch_samples = []
        batch_tokens = 0

        for sample in self.dataset:
            # Get sequence length from sample
            if self.length_column not in sample:
                raise ValueError(
                    f"Sample does not have '{self.length_column}' field. "
                    f"Add length information to dataset before using PackedIterableDataset."
                )

            seq_len = sample[self.length_column]

            if batch_tokens + seq_len <= self.token_budget:
                # Fits in current batch
                batch_samples.append(sample)
                batch_tokens += seq_len
            else:
                # Current batch is full, yield it
                if batch_samples:
                    yield batch_samples

                # Start new batch with current sample
                batch_samples = [sample]
                batch_tokens = seq_len

        # Yield remaining batch
        if batch_samples and (not self.drop_last or batch_tokens >= self.token_budget * 0.5):
            yield batch_samples


class PackingCollatorWrapper:
    """Wrapper that packs output from existing collators into flat tensors.

    This wrapper takes any existing collator (MLM, classification, token classification, etc.)
    and wraps its output into the packed format expected by the model:
    - Flat tensors of shape [1, total_tokens] (if token_budget > 0, padded to [1, token_budget])
    - Attention mask with sequence indices (0, 1, 2, ...) and -1 for padding

    The wrapper automatically detects label type and handles both:
    - Token-level labels (e.g., MLM): shape [batch_size, seq_len] -> packed to [1, total_tokens]
    - Sequence-level labels (e.g., classification): shape [batch_size] -> packed to [num_sequences]

    Args:
        base_collator: The underlying collator to wrap.
        token_budget: Maximum total tokens per packed batch. If 0, no padding is applied and
            output is returned with dynamic shape [1, total_tokens]. Defaults to 0.

    Example:
        >>> from bertblocks.training.objectives import MaskedLanguageModelingCollator
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> mlm_collator = MaskedLanguageModelingCollator(tokenizer, max_sequence_length=512)
        >>> # With padding for fixed-size batches
        >>> packing_collator = PackingCollatorWrapper(mlm_collator, token_budget=512 * 32)
        >>> # Without padding for dynamic-size batches
        >>> flat_collator = PackingCollatorWrapper(mlm_collator, token_budget=0)
    """

    def __init__(
        self,
        base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
        token_budget: int = 0,
    ) -> None:
        self.base_collator = base_collator
        self.token_budget = token_budget
        # Get pad_token_id from base collator if available
        if hasattr(base_collator, "pad_token_id"):
            self.pad_token_id = base_collator.pad_token_id  # type: ignore
        else:
            self.pad_token_id = 0
        # Get max_sequence_length for safety truncation
        if hasattr(base_collator, "max_sequence_length"):
            self.max_sequence_length = base_collator.max_sequence_length  # type: ignore
        else:
            self.max_sequence_length = None

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pack a batch of samples into flat tensors.

        Args:
            batch: List of samples from the dataset.

        Returns:
            Dictionary with packed tensors:
            - input_ids: [1, token_budget] (if token_budget > 0) or [1, total_tokens] (if token_budget == 0)
            - attention_mask: [1, token_budget] or [1, total_tokens] with 0-based sequence indices and -1 for padding
            - labels (if present):
                - Token-level: [1, token_budget] or [1, total_tokens]
                - Sequence-level: [num_sequences] or [num_sequences, num_classes]
        """
        # Call the base collator to get standard batched output
        collated = self.base_collator(batch)
        input_ids = collated["input_ids"]  # [B, L]
        attention_mask = collated["attention_mask"]  # [B, L]
        labels = collated.get("labels", None)

        batch_size, seq_len = input_ids.shape[:2]

        # Safety check: truncate sequences that exceed max_sequence_length
        if self.max_sequence_length is not None and seq_len > self.max_sequence_length:
            input_ids = input_ids[:, : self.max_sequence_length]
            attention_mask = attention_mask[:, : self.max_sequence_length]
            if labels is not None and labels.dim() > 1 and labels.shape[1] == seq_len:
                # Token-level labels, truncate
                labels = labels[:, : self.max_sequence_length]
            batch_size, seq_len = input_ids.shape[:2]

        # Detect label type: token-level vs sequence-level
        has_labels = labels is not None
        is_token_level_labels = labels.dim() > 1 and labels.shape[1] == seq_len if has_labels else False
        seq_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(batch_size, seq_len)

        # Flatten and select valid positions using boolean indexing
        valid_mask = attention_mask.bool()  # [B, L]
        packed_input_ids = input_ids[valid_mask]  # [num_valid_tokens]
        packed_attention_mask = seq_indices[valid_mask]  # [num_valid_tokens] with sequence IDs
        packed_labels = (labels[valid_mask] if is_token_level_labels else labels) if has_labels else None

        # Pad to token_budget if specified
        if self.token_budget > 0:
            current_length = len(packed_input_ids)
            if current_length < self.token_budget:
                pad_length = self.token_budget - current_length

                # Pad input_ids
                padding = torch.full((pad_length,), self.pad_token_id, dtype=torch.long, device=input_ids.device)
                packed_input_ids = torch.cat([packed_input_ids, padding])

                # Pad attention_mask with -1 (padding marker)
                padding_mask = torch.full((pad_length,), -1, dtype=torch.long, device=input_ids.device)
                packed_attention_mask = torch.cat([packed_attention_mask, padding_mask])

                # Pad token-level labels with -100 (ignore index)
                if has_labels and is_token_level_labels:
                    padding_labels = torch.full((pad_length,), -100, dtype=torch.long, device=input_ids.device)
                    packed_labels = torch.cat([packed_labels, padding_labels])

        # Build output dict with batch dimension (batch_size=1 for packed format)
        output = {
            "input_ids": packed_input_ids.unsqueeze(0),
            "attention_mask": packed_attention_mask.unsqueeze(0),
        }

        # Add labels based on type
        if has_labels:
            if is_token_level_labels:
                # Token-level: add batch dimension [1, num_tokens]
                output["labels"] = packed_labels.unsqueeze(0)
            else:
                # Sequence-level: no batch dimension, keep [num_sequences] or [num_sequences, C]
                output["labels"] = packed_labels

        return output


__all__ = [
    "is_packed_batch",
    "PackingBatchSampler",
    "PackingIterableDataset",
    "PackingCollatorWrapper",
]
