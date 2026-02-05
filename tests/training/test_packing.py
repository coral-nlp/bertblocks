import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from bertblocks.training.packing import PackedDataset, is_packed_batch


@pytest.fixture(scope="module")
def tokenizer():
    """Test tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def iterable_dataset():
    """Test dataset."""
    return load_dataset("roneneldan/TinyStories", split="train", streaming=True)


class TestIsPackedBatch:
    """Tests for the is_packed_batch detection function."""

    def test_standard(self):
        """Test that standard binary (0/1) attention masks are not detected as packed."""
        assert not is_packed_batch(torch.tensor([[1, 1, 1, 0, 0, 0]])), "Binary mask should not be detected as packed"
        assert not is_packed_batch(torch.tensor([[1, 1, 1, 1, 1]])), "All-ones mask should not be detected as packed"
        assert not is_packed_batch(torch.tensor([[0, 0, 0, 0]])), "All-zeros mask should not be detected as packed"

    def test_packed(self):
        """Test that packed format with sequence indices is correctly detected."""
        assert is_packed_batch(torch.tensor([[0, 0, 1, 1, 2, 2]])), "Sequence-indexed mask should be detected as packed"
        assert is_packed_batch(torch.tensor([[5, 5, 6, 6, 9, 9]])), "Sequence-indexed mask should be detected as packed"

    def test_packed_with_padding(self):
        """Test that packed format with -1 padding markers is correctly detected."""
        assert is_packed_batch(torch.tensor([[0, 0, 1, -1, -1]])), "Mask with -1 padding should be detected as packed"
        assert is_packed_batch(torch.tensor([[1, 0, 0, -1, -1]])), "Mask with -1 padding should be detected as packed"

    def test_none(self):
        """Test that None is handled correctly (not packed)."""
        assert not is_packed_batch(None), "None should not be detected as packed"

    def test_single_sequence(self):
        """Test edge cases with single sequences."""
        assert not is_packed_batch(torch.tensor([[0, 0, 0, 0]])), "Single sequence with all zeros is standard"
        assert not is_packed_batch(torch.tensor([[1, 1, 1, 0, 0]])), "Single sequence 0/1 is standard"

    def test_batch_dimension(self):
        """Test that detection works across batch dimensions."""
        batch_standard = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        assert not is_packed_batch(batch_standard), "Batch of standard masks should not be packed"

        batch_packed = torch.tensor([[0, 0, 1, 1, 2], [0, 0, 0, 1, 1]])
        assert is_packed_batch(batch_packed), "Batch with sequence indices should be packed"

        batch_packed_padded = torch.tensor([[1, 1, 1, 0, 0, -1, -1], [1, 1, 0, 0, 0, -1, -1]])
        assert is_packed_batch(batch_packed_padded), "Batch with padding indicators should be packed"


class TestPackedDataset:
    """Tests for PackedDataset."""

    @pytest.mark.dependency()
    @pytest.mark.parametrize("token_budget", [128, 256, 512])
    def test_return_format(self, tokenizer, iterable_dataset, token_budget):
        """Tests if the returned batches contain tokenized data in packed format."""
        packed = PackedDataset(
            dataset=iterable_dataset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=token_budget,
            buffer_size=64,
        )
        n_batches = 0
        for i, batch in enumerate(packed):
            n_batches += 1
            for ex in batch:
                assert "input_ids" in ex, "Sample should contain input_ids"
                assert "attention_mask" in ex, "Sample should contain attention_mask"
                assert isinstance(ex["input_ids"], torch.Tensor), "Input ids should be a tensor"
                assert isinstance(ex["attention_mask"], torch.Tensor), "Attention mask should be a tensor"

            if i >= 5:
                break  # We don't want this to run forever

        assert n_batches > 0, "Should return at least 1 batch"

    @pytest.mark.dependency(depends=["TestPackedDataset::test_return_format[128]"])
    @pytest.mark.parametrize("token_budget", [128, 256, 512])
    def test_token_budget(self, tokenizer, iterable_dataset, token_budget):
        """Test if every returned batch is at or below the token budget."""
        packed = PackedDataset(
            dataset=iterable_dataset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=token_budget,
            buffer_size=64,
        )
        for i, batch in enumerate(packed):
            for ex in batch:
                # Count valid tokens (attention_mask >= 0)
                attention_mask = ex["attention_mask"]
                total_tokens = (attention_mask >= 0).sum().item()
                assert total_tokens <= token_budget, "Total tokens should be less or equal to budget"
                assert ex["input_ids"].shape[0] == token_budget, "Total sequence length should be equal to budget"
            if i >= 5:
                break  # We don't want this to run forever

    @pytest.mark.dependency(depends=["TestPackedDataset::test_return_format[128]"])
    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    @pytest.mark.parametrize("buffer_size", [16, 64, 256])
    def test_buffer(self, tokenizer, iterable_dataset, n_samples, buffer_size):
        """Test if changing the buffer parameter still emits the whole dataset."""
        packed = PackedDataset(
            dataset=iterable_dataset.take(n_samples),
            tokenizer=tokenizer,
            max_length=128,
            token_budget=256,
            buffer_size=buffer_size,
        )
        batches = list(packed)
        # Count unique sequences across all batches
        total_samples = 0
        for batch in batches:
            for ex in batch:
                attention_mask = ex["attention_mask"]
                # Count unique sequence indices (excluding -1 padding)
                valid_mask = attention_mask >= 0
                if valid_mask.any():
                    unique_seqs = torch.unique(attention_mask[valid_mask])
                    total_samples += len(unique_seqs)

        assert total_samples == n_samples, "Packing should return full input dataset"

    @pytest.mark.dependency(
        depends=["TestPackedDataset::test_return_format[128]", "TestPackedDataset::test_buffer[16-10]"]
    )
    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    @pytest.mark.parametrize("lookahead", [0, 5, 20])
    def test_lookahead(self, tokenizer, iterable_dataset, n_samples, lookahead):
        """Test if changing the lookahead parameter still emits the whole dataset."""
        packed = PackedDataset(
            dataset=iterable_dataset.take(n_samples),
            tokenizer=tokenizer,
            max_length=128,
            token_budget=256,
            buffer_size=64,
            lookahead=lookahead,
        )
        batches = list(packed)
        total_samples = 0
        for batch in batches:
            for ex in batch:
                attention_mask = ex["attention_mask"]  # Remove batch dim
                valid_mask = attention_mask >= 0
                if valid_mask.any():
                    unique_seqs = torch.unique(attention_mask[valid_mask])
                    total_samples += len(unique_seqs)
        assert total_samples == n_samples, "Packing should return full input dataset"

    @pytest.mark.dependency(depends=["TestPackedDataset::test_return_format[128]"])
    def test_attention_mask(self, tokenizer, iterable_dataset):
        """Test that packed format uses sequence-indexed attention masks."""
        packed = PackedDataset(
            dataset=iterable_dataset,
            tokenizer=tokenizer,
            max_length=128,
            token_budget=320,  # not divisible by max_length to force padding
            buffer_size=64,
        )
        batch = next(iter(packed))
        for ex in batch:
            valid_mask = ex["attention_mask"] >= 0
            unique_sequences = torch.unique(ex["attention_mask"][valid_mask])
            expected_indices = torch.arange(len(unique_sequences), dtype=unique_sequences.dtype)

            assert (ex["attention_mask"] == -1).any(), "Should have padding tokens marked with -1"
            assert unique_sequences[0] == 0, "Sequences should start from index 0"
            assert len(unique_sequences) > 1, "Should have multiple sequences packed together"
            assert torch.equal(unique_sequences, expected_indices), "Sequence indices should be consecutive"

    @pytest.mark.dependency(depends=["TestPackedDataset::test_return_format[128]"])
    def test_padding(self, tokenizer, iterable_dataset):
        """Test that packed format doesn't insert padding when not needed."""
        packed = PackedDataset(
            dataset=iterable_dataset,
            tokenizer=tokenizer,
            max_length=64,  # Small max_length, all sequences in dataset exceed it
            token_budget=256,  # Divisible by max_length, no padding necessary
            buffer_size=64,
        )
        batch = next(iter(packed))
        for ex in batch:
            valid_indices = torch.nonzero(ex["attention_mask"] >= 0, as_tuple=False).flatten()
            expected_valid_indices = torch.arange(len(valid_indices), dtype=valid_indices.dtype)

            assert ex["input_ids"].shape[0] == 256, "Tensor size should be equal to token budget"
            assert torch.equal(valid_indices, expected_valid_indices), "Valid tokens should be densely packed, no gaps"
