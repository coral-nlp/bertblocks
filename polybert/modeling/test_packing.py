import pytest
import torch
from torch.autograd import gradcheck

from polybert.modeling.packing import (
    index_first_axis,
    index_put_first_axis,
    pack_seq,
    unpack_seq,
)


class TestIndexFirstAxis:
    """Test the IndexFirstAxis autograd function."""
    
    def test_forward(self):
        """Test forward pass."""
        input_tensor = torch.randn(4, 6, dtype=torch.float32)
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        
        result = index_first_axis(input_tensor, indices)
        expected = input_tensor[indices]
        
        assert result.shape == (3, 6)
        assert torch.allclose(result, expected)

    def test_gradient(self):
        """Test gradient computation."""
        input_tensor = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        
        # Test gradcheck
        def func(x):
            return index_first_axis(x, indices)
        
        assert gradcheck(func, input_tensor, eps=1e-6, atol=1e-4)
    
    def test_empty_indices(self):
        """Test with empty indices tensor."""
        input_tensor = torch.randn(4, 6, dtype=torch.float32)
        indices = torch.tensor([], dtype=torch.long)
        
        result = index_first_axis(input_tensor, indices)
        assert result.shape == (0, 6)
    
    def test_single_index(self):
        """Test with single index."""
        input_tensor = torch.randn(4, 6, dtype=torch.float32)
        indices = torch.tensor([2], dtype=torch.long)
        
        result = index_first_axis(input_tensor, indices)
        expected = input_tensor[2:3]
        
        assert result.shape == (1, 6)
        assert torch.allclose(result, expected)


class TestIndexPutFirstAxis:
    """Test the IndexPutFirstAxis autograd function."""
    
    def test_forward(self):
        """Test forward pass."""
        values = torch.randn(3, 6, dtype=torch.float32)
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        first_axis_dim = 5
        
        result = index_put_first_axis(values, indices, first_axis_dim)
        
        assert result.shape == (5, 6)
        assert torch.allclose(result[0], values[0])
        assert torch.allclose(result[2], values[1])
        assert torch.allclose(result[3], values[2])
        assert torch.allclose(result[1], torch.zeros(6))
        assert torch.allclose(result[4], torch.zeros(6))
    
    def test_gradient(self):
        """Test gradient computation."""
        values = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([0, 2, 3], dtype=torch.long)
        first_axis_dim = 5
        
        def func(x):
            return index_put_first_axis(x, indices, first_axis_dim)
        
        assert gradcheck(func, values, eps=1e-6, atol=1e-4)
    
    def test_empty_values(self):
        """Test with empty values tensor."""
        values = torch.randn(0, 6, dtype=torch.float32)
        indices = torch.tensor([], dtype=torch.long)
        first_axis_dim = 5
        
        result = index_put_first_axis(values, indices, first_axis_dim)
        assert result.shape == (5, 6)
        assert torch.allclose(result, torch.zeros(5, 6))
    
    def test_single_value(self):
        """Test with single value."""
        values = torch.randn(1, 6, dtype=torch.float32)
        indices = torch.tensor([2], dtype=torch.long)
        first_axis_dim = 5
        
        result = index_put_first_axis(values, indices, first_axis_dim)
        
        assert result.shape == (5, 6)
        assert torch.allclose(result[2], values[0])
        for i in [0, 1, 3, 4]:
            assert torch.allclose(result[i], torch.zeros(6))


class TestPackSeqInput:
    """Test the pack_seq function."""
    
    def test_basic_packing(self):
        """Test basic packing functionality."""
        batch_size, seq_len, hidden_size = 2, 5, 4
        
        # Create test data
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],  # First sequence has length 3
            [1, 1, 1, 1, 0],  # Second sequence has length 4
        ], dtype=torch.bool)
        
        unpacked, indices, cu_seqlens, max_seq_len = pack_seq(hidden_states, attention_mask)
        
        # Check shapes
        assert unpacked.shape == (7, hidden_size)  # 3 + 4 = 7 valid tokens
        assert indices.shape == (7,)
        assert cu_seqlens.shape == (3,)  # batch_size + 1
        assert max_seq_len == 4
        
        # Check cumulative sequence lengths
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 3
        assert cu_seqlens[2] == 7
        
        # Check that packed tokens match original valid tokens
        expected_indices = torch.tensor([0, 1, 2, 5, 6, 7, 8], dtype=torch.long)
        assert torch.equal(indices, expected_indices)
    
    def test_single_sequence(self):
        """Test with single sequence."""
        batch_size, seq_len, hidden_size = 1, 4, 3
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        
        unpacked, indices, cu_seqlens, max_seq_len = pack_seq(hidden_states, attention_mask)
        
        assert unpacked.shape == (2, hidden_size)
        assert indices.shape == (2,)
        assert cu_seqlens.shape == (2,)
        assert max_seq_len == 2
        
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 2
    
    def test_all_padded(self):
        """Test with all padded sequences."""
        batch_size, seq_len, hidden_size = 2, 3, 4
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        unpacked, indices, cu_seqlens, max_seq_len = pack_seq(hidden_states, attention_mask)
        
        assert unpacked.shape == (0, hidden_size)
        assert indices.shape == (0,)
        assert cu_seqlens.shape == (3,)
        assert max_seq_len == 0
        
        assert torch.equal(cu_seqlens, torch.tensor([0, 0, 0], dtype=torch.int32))
    
    def test_no_padding(self):
        """Test with no padding."""
        batch_size, seq_len, hidden_size = 2, 3, 4
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        packed, indices, cu_seqlens, max_seq_len = pack_seq(hidden_states, attention_mask)
        
        assert packed.shape == (6, hidden_size)
        assert indices.shape == (6,)
        assert cu_seqlens.shape == (3,)
        assert max_seq_len == 3
        
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 3
        assert cu_seqlens[2] == 6
    
    def test_different_dtypes(self):
        """Test with different data types."""
        batch_size, seq_len, hidden_size = 2, 4, 3
        
        # Test with float32
        hidden_states_f32 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        attention_mask = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ], dtype=torch.bool)
        
        packed_f32, indices, cu_seqlens, max_seq_len = pack_seq(hidden_states_f32, attention_mask)
        assert packed_f32.dtype == torch.float32
        
        # Test with float64
        hidden_states_f64 = hidden_states_f32.double()
        packed_f64, _, _, _ = pack_seq(hidden_states_f64, attention_mask)
        assert packed_f64.dtype == torch.float64


class TestUnpackSeq:
    """Test the unpack_seq function."""
    
    def test_basic_padding(self):
        """Test basic padding functionality."""
        batch_size, seq_len, hidden_size = 2, 5, 4
        total_tokens = 7
        
        # Create test data
        hidden_states = torch.randn(total_tokens, hidden_size)
        indices = torch.tensor([0, 1, 2, 5, 6, 7, 8], dtype=torch.long)
        
        padded = unpack_seq(hidden_states, indices, batch_size, seq_len)
        
        # Check shape
        assert padded.shape == (batch_size, seq_len, hidden_size)
        
        # Check that valid tokens are correctly placed
        assert torch.allclose(padded[0, 0], hidden_states[0])
        assert torch.allclose(padded[0, 1], hidden_states[1])
        assert torch.allclose(padded[0, 2], hidden_states[2])
        assert torch.allclose(padded[1, 0], hidden_states[3])
        assert torch.allclose(padded[1, 1], hidden_states[4])
        assert torch.allclose(padded[1, 2], hidden_states[5])
        assert torch.allclose(padded[1, 3], hidden_states[6])
        
        # Check that padded positions are zero
        assert torch.allclose(padded[0, 3], torch.zeros(hidden_size))
        assert torch.allclose(padded[0, 4], torch.zeros(hidden_size))
        assert torch.allclose(padded[1, 4], torch.zeros(hidden_size))
    
    def test_single_sequence(self):
        """Test with single sequence."""
        batch_size, seq_len, hidden_size = 1, 4, 3
        total_tokens = 2
        
        hidden_states = torch.randn(total_tokens, hidden_size)
        indices = torch.tensor([0, 1], dtype=torch.long)
        
        padded = unpack_seq(hidden_states, indices, batch_size, seq_len)
        
        assert padded.shape == (batch_size, seq_len, hidden_size)
        assert torch.allclose(padded[0, 0], hidden_states[0])
        assert torch.allclose(padded[0, 1], hidden_states[1])
        assert torch.allclose(padded[0, 2], torch.zeros(hidden_size))
        assert torch.allclose(padded[0, 3], torch.zeros(hidden_size))
    
    def test_no_tokens(self):
        """Test with no tokens."""
        batch_size, seq_len, hidden_size = 2, 3, 4
        total_tokens = 0
        
        hidden_states = torch.randn(total_tokens, hidden_size)
        indices = torch.tensor([], dtype=torch.long)
        
        padded = unpack_seq(hidden_states, indices, batch_size, seq_len)
        
        assert padded.shape == (batch_size, seq_len, hidden_size)
        assert torch.allclose(padded, torch.zeros(batch_size, seq_len, hidden_size))
    
    def test_full_sequences(self):
        """Test with full sequences (no padding)."""
        batch_size, seq_len, hidden_size = 2, 3, 4
        total_tokens = 6
        
        hidden_states = torch.randn(total_tokens, hidden_size)
        indices = torch.arange(total_tokens, dtype=torch.long)
        
        padded = unpack_seq(hidden_states, indices, batch_size, seq_len)
        
        assert padded.shape == (batch_size, seq_len, hidden_size)
        for i in range(batch_size):
            for j in range(seq_len):
                token_idx = i * seq_len + j
                assert torch.allclose(padded[i, j], hidden_states[token_idx])


class TestPackingRoundTrip:
    """Test that pack_seq and unpack_seq are inverse operations."""
    
    def test_roundtrip(self):
        """Test roundtrip for 2D tensors."""
        batch_size, seq_len, hidden_size = 3, 6, 8
        
        # Create original data
        original_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.bool)
        
        # Unpad
        packed, indices, cu_seqlens, max_seq_len = pack_seq(original_hidden_states, attention_mask)
        
        # Pad back
        reconstructed = unpack_seq(packed, indices, batch_size, seq_len)
        
        # Check that valid tokens are perfectly reconstructed
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask[i, j]:
                    assert torch.allclose(reconstructed[i, j], original_hidden_states[i, j])
                else:
                    assert torch.allclose(reconstructed[i, j], torch.zeros(hidden_size))
    
    def test_roundtrip_no_padding(self):
        """Test roundtrip with no padding."""
        batch_size, seq_len, hidden_size = 2, 3, 4
        
        original_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Unpad
        unpacked, indices, cu_seqlens, max_seq_len = pack_seq(original_hidden_states, attention_mask)
        
        # Pad back
        reconstructed = unpack_seq(unpacked, indices, batch_size, seq_len)
        
        # Should be perfectly reconstructed
        assert torch.allclose(reconstructed, original_hidden_states)
    
    def test_roundtrip_gradient_flow(self):
        """Test that gradients flow correctly through the roundtrip."""
        batch_size, seq_len, hidden_size = 2, 4, 3
        
        # Create data with gradients
        original_hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        attention_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ], dtype=torch.bool)
        
        # Unpad
        unpacked, indices, cu_seqlens, max_seq_len = pack_seq(original_hidden_states, attention_mask)
        
        # Pad back
        reconstructed = unpack_seq(unpacked, indices, batch_size, seq_len)
        
        # Compute loss and backpropagate
        loss = reconstructed.sum()
        loss.backward()
        
        # Check that gradients exist and are correct
        assert original_hidden_states.grad is not None
        
        # Gradients should be 1 for valid tokens, 0 for padded tokens
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask[i, j]:
                    assert torch.allclose(original_hidden_states.grad[i, j], torch.ones(hidden_size))
                else:
                    assert torch.allclose(original_hidden_states.grad[i, j], torch.zeros(hidden_size))
