from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizerBase

from collections import deque

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset


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

    def __iter__(self) -> "Iterator":
        """Iterate with cross-rank synchronization.

        After receiving each batch (or exhausting data), performs an all_reduce
        to check if all ranks still have data. Stops all ranks when any rank is done.
        """
        iterator = iter(self.dataloader)

        while True:
            # Try to get next batch
            try:
                batch = next(iterator)
                has_data = True
            except StopIteration:
                has_data = False
                batch = None

            # Sync: all ranks continue only if ALL have data
            has_data_tensor = torch.tensor([has_data], dtype=torch.int32, device=self.device)
            dist.all_reduce(has_data_tensor, op=dist.ReduceOp.MIN)
            if has_data_tensor.item() == 0:
                return

            yield batch

    def __len__(self) -> int:
        """Return the length of the underlying dataloader if available."""
        return len(self.dataloader)


class PackedDataset(IterableDataset):
    """An iterable dataset that tokenizes and packs samples into groups.

    Each yielded item is a list of pretokenized sequences (``input_ids`` and ``attention_mask``) whose combined token
    count fits within the token budget. Packing is done on-the-fly from a buffer. Tokenization happens here so the
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
        pad_to_budget (bool): When True, append a dummy padding sequence to each packed batch so that the total
            token count equals exactly ``token_budget``. This produces a fixed unpadded length every step,
            eliminating ``torch.compile`` recompilation from dynamic shapes. The dummy tokens use the tokenizer's
            pad token ID with ``attention_mask=1`` so they survive unpadding; they receive ``-100`` labels from the
            collator and do not contribute to the loss. Defaults to False.
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
        """Buffer-based packing iterator."""
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
        """Fill one batch from the tokenized buffer, greedily picking items that fit within the token budget."""
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

        if self.pad_to_budget and total_seq_len < self.token_budget:
            fill_len = self.token_budget - total_seq_len
            pad_id = self.tokenizer.pad_token_id or 0
            selected.append(
                {
                    "input_ids": torch.full((fill_len,), pad_id, dtype=torch.long),
                    "attention_mask": torch.ones(fill_len, dtype=torch.long),
                }
            )

        return selected
