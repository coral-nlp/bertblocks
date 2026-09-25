from unittest.mock import patch

import datasets
import pytest
import torch

from bertblocks.training.packing import PackingIterableDataset

TOKEN_BUDGET = 8


def _map_style(rows=16):
    return datasets.Dataset.from_dict({
        "input_ids": [[i] * 4 for i in range(rows)],
        "length": [4] * rows,
        "idx": list(range(rows)),
    })


def _ids_seen(packed, num_workers, worker_id):
    info = None
    if num_workers is not None:
        info = torch.utils.data._utils.worker.WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=0, dataset=packed
        )
    with patch("torch.utils.data.get_worker_info", return_value=info):
        return [sample["idx"] for batch in packed for sample in batch]


class TestWorkerSharding:

    def test_workers_get_disjoint_samples_covering_the_dataset(self):
        packed = PackingIterableDataset(_map_style(16), token_budget=TOKEN_BUDGET)
        seen = [_ids_seen(packed, num_workers=4, worker_id=w) for w in range(4)]

        flat = [idx for worker in seen for idx in worker]
        assert sorted(flat) == list(range(16)), "samples lost or duplicated across workers"
        assert all(len(worker) == 4 for worker in seen), seen

    def test_single_worker_and_no_worker_sees_everything(self):
        packed = PackingIterableDataset(_map_style(16), token_budget=TOKEN_BUDGET)
        assert sorted(_ids_seen(packed, num_workers=None, worker_id=0)) == list(range(16))
        assert sorted(_ids_seen(packed, num_workers=1, worker_id=0)) == list(range(16))

    def test_unshardable_dataset_warns_instead_of_duplicating_silently(self):
        class PlainIterable(torch.utils.data.IterableDataset):
            def __iter__(self):
                yield from ({"idx": i, "length": 4} for i in range(8))

        packed = PackingIterableDataset(PlainIterable(), token_budget=TOKEN_BUDGET)
        with pytest.warns(UserWarning, match="cannot be sharded"):
            _ids_seen(packed, num_workers=4, worker_id=0)

    def test_streaming_dataset_is_left_to_shard_itself(self):
        streaming = _map_style(16).to_iterable_dataset()
        packed = PackingIterableDataset(streaming, token_budget=TOKEN_BUDGET)

        info = torch.utils.data._utils.worker.WorkerInfo(id=0, num_workers=4, seed=0, dataset=packed)
        with patch("torch.utils.data.get_worker_info", return_value=info):
            assert packed._worker_dataset() is streaming
