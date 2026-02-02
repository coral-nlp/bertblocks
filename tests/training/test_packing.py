import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from bertblocks.training.packing import PackedDataset


@pytest.fixture(scope="module")
def tokenizer():
    """Test tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def iterable_dataset():
    """Test dataset."""
    return load_dataset("roneneldan/TinyStories", split="train", streaming=True)


class TestPackedDataset:
    """Tests for PackedDataset."""

    @pytest.mark.dependency()
    @pytest.mark.parametrize("token_budget", [128, 256, 512])
    def test_yields_pretokenized_groups(self, tokenizer, iterable_dataset, token_budget):
        """Tests if the returned batches contain tokenized data."""
        packed = PackedDataset(
            dataset=iterable_dataset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=token_budget,
            buffer_size=64,
        )
        groups = []
        for i, group in enumerate(packed):
            groups.append(group)
            if i >= 4:
                break
        assert len(groups) > 0
        assert "input_ids" in groups[0][0]
        assert isinstance(groups[0][0]["input_ids"], torch.Tensor)

    @pytest.mark.dependency(depends=["TestPackedDatasetIterable::test_yields_pretokenized_groups[128]"])
    @pytest.mark.parametrize("token_budget", [128, 256, 512])
    def test_respects_budget(self, tokenizer, iterable_dataset, token_budget):
        """Test if every returned batch is below the token budget."""
        packed = PackedDataset(
            dataset=iterable_dataset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=token_budget,
            buffer_size=64,
        )
        for i, group in enumerate(packed):
            total_tokens = sum(len(s["input_ids"]) for s in group)
            if len(group) > 1:
                assert total_tokens <= token_budget
            if i >= 9:
                break

    @pytest.mark.dependency(depends=["TestPackedDatasetIterable::test_yields_pretokenized_groups[128]"])
    @pytest.mark.parametrize("buffer_size", [16, 64, 256])
    def test_all_samples_emitted(self, tokenizer, iterable_dataset, buffer_size):
        """Test if changing the buffer parameter still returns the whole  dataset."""
        n_source = 50
        subset = iterable_dataset.take(n_source)
        packed = PackedDataset(
            dataset=subset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=256,
            buffer_size=buffer_size,
        )
        total_samples = sum(len(group) for group in packed)
        assert total_samples == n_source

    @pytest.mark.dependency(depends=["TestPackedDatasetIterable::test_yields_pretokenized_groups[128]"])
    @pytest.mark.parametrize("lookahead", [0, 5, 20])
    def test_lookahead(self, tokenizer, iterable_dataset, lookahead):
        """Test if changing the lookahead parameter still returns the whole  dataset."""
        subset = iterable_dataset.take(50)
        packed = PackedDataset(
            dataset=subset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=256,
            buffer_size=64,
            lookahead=lookahead,
        )
        groups = list(packed)
        total_samples = sum(len(g) for g in groups)
        assert total_samples == 50
