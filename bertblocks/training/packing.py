from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizerBase

from collections import deque

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset


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


class DistributedStoppingDataLoader:
    """Wrapper around a DataLoader that synchronizes stopping across distributed ranks.

    In distributed training with sequence packing, different ranks may produce different numbers
    of batches depending on how sequences pack together. This wrapper ensures all ranks stop
    together when any rank exhausts its data, preventing DDP hangs.

    The synchronization happens in the main process after receiving each batch from workers,
    so it's compatible with ``num_workers > 0``.

    Args:
        dataloader (DataLoader): The underlying DataLoader to wrap.
        device (torch.device | str): Device for synchronization tensors. Defaults to "cuda".
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: "torch.device | str" = "cuda",
    ) -> None:
        self.dataloader = dataloader
        self.device = device
        # Pre-allocate the sync tensor
        self.has_data_tensor = torch.tensor([0], dtype=torch.int32, device=self.device)

    def __iter__(self) -> "Iterator":
        """Iterate with cross-rank synchronization.

        After receiving each batch (or exhausting data), performs an all_reduce
        to check if all ranks still have data. Stops all ranks when any rank is done.
        """
        iterator = iter(self.dataloader)

        while True:
            try:
                batch = next(iterator)
                has_data = 1
            except StopIteration:
                has_data = 0
                batch = None

            # Set the tensor to current local state
            self.has_data_tensor.fill_(has_data)

            # Perform sync across all ranks
            dist.all_reduce(self.has_data_tensor, op=dist.ReduceOp.MIN)

            # If any rank has no data, stop all ranks together
            if self.has_data_tensor.item() == 0:
                # Synchronize all ranks before exiting to prevent deadlocks when starting the next epoch
                dist.barrier()
                return

            # All ranks still have data, continue yielding
            yield batch

    def __len__(self) -> int:
        """Return length of underlying dataloader if it has one."""
        return len(self.dataloader)


class PackedDataset(IterableDataset):
    """An iterable dataset that tokenizes and packs samples into flat tensors.

    Each yielded item is a dict with flat tensors of shape (token_budget,):
    - ``input_ids``: concatenated token IDs from all packed sequences
    - ``attention_mask``: sequence indices (0, 1, 2, ...) indicating which sequence each token belongs to.
      Padding tokens are marked with -1.

    Packing is done on-the-fly from a buffer. Tokenization happens here so the
    downstream collator must run in pre-tokenized mode.

    Note:
        For distributed training, wrap the DataLoader with :class:`DistributedStoppingDataLoader` to ensure
        all ranks stop together when any rank exhausts its data. This prevents DDP hangs from uneven batch counts.

    Args:
        dataset (Dataset): The underlying dataset to wrap (map-style or iterable).
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer for tokenizing raw text.
        max_length (int): Maximum sequence length (for truncation).
        token_budget (int): Maximum total tokens per packed group.
        text_column (str): Column name containing raw text.
        pretokenized (bool): Whether input data is already tokenized.
        buffer_size (int): Number of samples to buffer for packing.
        lookahead (int): How many samples to look into the buffer to find a matching sample before returning the batch.
        pad_to_budget (bool): No longer used. All batches are now padded to ``token_budget`` with padding tokens
            marked as -1 in the attention_mask for efficient filtering. Kept for backward compatibility.
    """

    def __init__(
        self,
        dataset: "Dataset",
        tokenizer: "PreTrainedTokenizerBase",
        max_length: int,
        token_budget: int,
        text_column: str = "text",
        pretokenized: bool = False,
        buffer_size: int = 4096,
        lookahead: int = 0,
        pad_to_budget: bool = False,
    ) -> None:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multiple worker processes; shard so each receives a unique subset of data
            self.dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
        else:
            # Single worker in main thread; no sharding necessary
            self.dataset = dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_budget = token_budget
        self.text_column = text_column
        self.pretokenized = pretokenized
        self.buffer_size = buffer_size
        self.lookahead = lookahead
        self.pad_to_budget = pad_to_budget

    def __iter__(self) -> "Iterator[list[dict[str, Any]]]":
        """Buffer-based packing iterator that yields flat packed tensors."""
        dataset_iter = iter(self.dataset)
        buffer: deque[dict[str, torch.Tensor]] = deque()
        data_exhausted = False

        def fill_buffer() -> None:
            nonlocal data_exhausted
            while len(buffer) < self.buffer_size and not data_exhausted:
                try:
                    sample = next(dataset_iter)
                except StopIteration:
                    data_exhausted = True
                    return
                tokenized = self._tokenize(sample)
                buffer.append(tokenized)

        while True:
            fill_buffer()
            if not buffer:
                break
            yield self._batch_from_buffer(buffer)

    def _tokenize(self, sample: "dict[str, str | torch.Tensor]") -> "dict[str, torch.Tensor]":
        """Tokenize a single sample, returning input_ids and attention_mask."""
        if self.pretokenized:
            input_ids = sample["input_ids"]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = sample.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            elif not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        encoded = self.tokenizer(
            sample[self.text_column],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def _batch_from_buffer(self, buffer: deque) -> "list[dict[str, torch.Tensor]]":
        """Fill one batch from the tokenized buffer, greedily picking items that fit within the token budget.

        Returns a list containing a single dict with flattened tensors:
        - input_ids: [token_budget] concatenated token IDs from all packed sequences
        - attention_mask: [token_budget] sequence indices (0, 1, 2, ...) indicating which sequence each token belongs
            to. Padding tokens are marked with -1.
        """
        selected: list[dict[str, torch.Tensor]] = []
        deferred: list[dict[str, torch.Tensor]] = []
        total_seq_len = 0

        while buffer:
            candidate = buffer.popleft()
            candidate_len = len(candidate["input_ids"])
            if total_seq_len + candidate_len > self.token_budget:
                # This ain't it chief, back to the queue
                deferred.append(candidate)
                # Keep searching in the lookahead, or return
                if len(deferred) < self.lookahead:
                    continue
                else:
                    break
            else:
                selected.append(candidate)
                total_seq_len += candidate_len

        # Deferred sequences go back to the front to start the next batch
        buffer.extendleft(deferred)

        # Concatenate all sequences into flat tensors with sequence-indexed attention mask
        input_ids_list = []
        attention_mask_list = []

        for seq_idx, item in enumerate(selected):
            input_ids_list.append(item["input_ids"])
            # Use sequence index instead of 0/1 mask
            attention_mask_list.append(torch.full((len(item["input_ids"]),), seq_idx, dtype=torch.long))

        # Pad to token_budget if needed
        pad_id = self.tokenizer.pad_token_id or 0
        if total_seq_len < self.token_budget:
            fill_len = self.token_budget - total_seq_len
            input_ids_list.append(torch.full((fill_len,), pad_id, dtype=torch.long))
            # Mark padding tokens with -1 in the attention mask
            attention_mask_list.append(torch.full((fill_len,), -1, dtype=torch.long))

        # Return in list and add batch dimension (batch_size=1) for compatibility with collator logic
        return [
            {
                "input_ids": torch.cat(input_ids_list).unsqueeze(0),
                "attention_mask": torch.cat(attention_mask_list).unsqueeze(0),
            }
        ]


__all__ = ["is_packed_batch", "DistributedStoppingDataLoader", "PackedDataset"]
