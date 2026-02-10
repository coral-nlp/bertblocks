"""Sequence packing using batch sampling and a wrapper around collators to produce full flat-packed batches."""

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, Sampler


def _pack_sequences(
    indices: list[int],
    lengths: list[int],
    token_budget: int,
) -> list[list[int]]:
    """Pack sequences greedily into batches that fit within token budget.

    Args:
        indices: Dataset indices to pack.
        lengths: Sequence lengths for all dataset samples.
        token_budget: Maximum tokens per packed batch.

    Returns:
        List of packed batches (each batch is a list of dataset indices).
    """
    batches: list[list[int]] = []
    ptr = 0
    while ptr < len(indices):
        batch: list[int] = []
        tokens = 0
        while ptr < len(indices):
            seq_len = lengths[indices[ptr]]
            if not batch or tokens + seq_len <= token_budget:
                batch.append(indices[ptr])
                tokens += seq_len
                ptr += 1
            else:
                break
        if batch:
            batches.append(batch)
    return batches


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
    """Distributed batch sampler that packs variable-length sequences into token-budget batches.

    This sampler follows PyTorch's DistributedSampler pattern: it packs ALL sequences globally
    into batches, truncates to ensure even distribution across ranks, then assigns batches
    round-robin to each rank. All ranks independently compute the same global packing using
    deterministic shuffling, ensuring no distributed communication is needed.

    Args:
        lengths: Sequence lengths for all dataset samples.
        token_budget: Maximum tokens per packed batch.
        world_size: Number of processes participating in distributed training.
            If None, uses world_size from current distributed group.
        rank: Rank of current process within num_replicas.
            If None, uses rank from current distributed group.
        shuffle: Whether to shuffle indices before packing. Default: True.
        seed: Random seed for shuffling. Should be identical across all processes. Default: 0.
        drop_last: Drop the last batch, which might not be fully packed. Default: False.

    Example:

        >>> from datasets import load_dataset
        >>> from torch.utils.data import DataLoader
        >>> dataset = load_dataset("dataset_name", split="train")
        >>> batch_sampler = PackingBatchSampler(list(datasets["length"]), token_budget=4096)
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
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]")

        self.lengths = lengths
        self.token_budget = token_budget
        self.num_replicas = world_size
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Compute batches once to determine num_samples
        self._batches = self._compute_batches()
        self.num_samples = len(self._batches)

    def _compute_batches(self) -> list[list[int]]:
        """Compute this rank's batches for the current epoch."""
        total_size = len(self.lengths)

        # Deterministically shuffle based on epoch and seed (same across all ranks)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(total_size, generator=g).tolist()
        else:
            indices = list(range(total_size))

        # Pack all sequences globally
        all_batches = _pack_sequences(indices, self.lengths, self.token_budget)

        # Drop last batch if requested (it might not be fully packed)
        if self.drop_last and len(all_batches) > 0:
            all_batches = all_batches[:-1]

        # Truncate to make batch count evenly divisible across ranks
        num_batches = len(all_batches) - (len(all_batches) % self.num_replicas)
        all_batches = all_batches[:num_batches]

        # Distribute batches round-robin to ranks
        rank_batches = all_batches[self.rank :: self.num_replicas]

        return rank_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Return an iterator over the current rank's batches."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._batches)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When shuffle=True, this ensures all replicas use a different random ordering
        for each epoch. Otherwise, the next iteration will yield the same ordering.

        Args:
            epoch: Epoch number.
        """
        self.epoch = epoch
        self._batches = self._compute_batches()
        self.num_samples = len(self._batches)
        print(f"[Rank {self.rank}] Epoch {self.epoch}: sampled {self.num_samples} batches")


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
    "PackingCollatorWrapper",
    "PackingIterableDataset",
]
