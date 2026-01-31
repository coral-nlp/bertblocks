from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def _unpad_batch(
    batch: dict[str, torch.Tensor],
    pad_token_id: int,
    ignore_index: int = -100,
) -> list[dict[str, torch.Tensor]]:
    """Unpad a collated batch into individual variable-length sequences.

    Uses attention_mask if present, otherwise infers lengths from pad_token_id.
    """
    input_ids = batch["input_ids"]
    batch_size = input_ids.shape[0]

    if "attention_mask" in batch:
        lengths = batch["attention_mask"].sum(dim=1).tolist()
    else:
        lengths = [(row != pad_token_id).sum().item() for row in input_ids]

    examples = []
    for i in range(batch_size):
        seq_len = int(lengths[i])
        if seq_len == 0:
            continue
        example: dict[str, torch.Tensor] = {"input_ids": input_ids[i, :seq_len]}
        if "labels" in batch and batch["labels"] is not None:
            example["labels"] = batch["labels"][i, :seq_len]
        examples.append(example)
    return examples


def _pad_and_stack(
    examples: list[dict[str, torch.Tensor]],
    pad_token_id: int,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Pad variable-length examples to the longest in the group and stack into a batch."""
    input_ids = pad_sequence([ex["input_ids"] for ex in examples], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(
        [torch.ones(len(ex["input_ids"]), dtype=torch.long) for ex in examples],
        batch_first=True,
        padding_value=0,
    )
    result: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "labels" in examples[0]:
        result["labels"] = pad_sequence([ex["labels"] for ex in examples], batch_first=True, padding_value=ignore_index)
    return result


class SequencePacker(ABC):
    """Base class for sequence packers that rebatch a DataLoader by token budget.

    The inner DataLoader (with its collator) handles tokenization, masking, and
    label computation. This wrapper unpads those batches into individual sequences,
    then reassembles them into variable-size batches where total tokens ≈ token_budget.

    The model's internal unpadding (attention_mask → cu_seqlens → flash attention)
    handles the rest — no explicit cu_seqlens are produced here.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        token_budget: int,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        buffer_size: int = 4096,
    ) -> None:
        self.dataloader = dataloader
        self.token_budget = token_budget
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over the unpacked batches from the inner DataLoader, re-assembling them into packed batches."""
        buffer: deque[dict[str, torch.Tensor]] = deque()
        src_iter = iter(self.dataloader)
        src_exhausted = False

        def fill_buffer() -> None:
            nonlocal src_exhausted
            while len(buffer) < self.buffer_size and not src_exhausted:
                try:
                    batch = next(src_iter)
                except StopIteration:
                    src_exhausted = True
                    return
                buffer.extend(_unpad_batch(batch, self.pad_token_id, self.ignore_index))

        while True:
            fill_buffer()
            if not buffer:
                break
            selected = self._select_batch(buffer)
            if not selected:
                break
            yield _pad_and_stack(selected, self.pad_token_id, self.ignore_index)

    @abstractmethod
    def _select_batch(self, buffer: deque[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Select and remove sequences from the buffer to form one batch.

        The total tokens in the selected sequences (accounting for padding to the
        longest) should be as close to self.token_budget as possible without exceeding it.

        Returns an empty list if no valid batch can be formed.
        """
        ...


class NoOpSequencePacker(SequencePacker):
    """Passes batches through from the inner DataLoader unchanged."""

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Pass through batches from the inner DataLoader unchanged."""
        yield from self.dataloader

    def _select_batch(self, buffer: deque[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """No-op. Never actually gets called."""
        return []


class GreedySequencePacker(SequencePacker):
    """Accumulates sequences from the buffer, until total sequence length exceeds token_budget.

    Takes into account the next incoming sequences to maybe fill a batch, up until lookahead is reached.
    """

    def __init__(self, lookahead: int = 1024, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lookahead = lookahead

    def _select_batch(self, buffer: deque[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Re-assemble sequences from the buffer, until total sequence length exceeds token budget."""
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
        return selected


def get_sequence_packer_cls(strategy: str) -> type[SequencePacker]:
    """Get the sequence packer class for the given strategy.

    Args:
        strategy: Packing strategy. Available options: "none", "greedy".

    Raises:
        ValueError: If the strategy is unknown.

    Returns:
        The corresponding SequencePacker subclass.
    """
    match strategy:
        case "none":
            return NoOpSequencePacker
        case "greedy":
            return GreedySequencePacker
        case _:
            raise ValueError(f"Unknown sequence packing strategy: {strategy}")
