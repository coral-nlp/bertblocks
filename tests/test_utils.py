"""Utility functions used for testing."""

import torch


def get_boolean_attention_mask_mockup(batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
    """Create a boolean attention mask mockup of specified size."""
    masking = torch.randint(low=1, high=seq_length, size=(batch_size,)).unsqueeze(-1).to(device)
    attention_mask = (
        torch.linspace(0, seq_length, steps=seq_length).repeat(batch_size).reshape(batch_size, seq_length).to(device)
    ) < masking
    # Assert shape is (b, s)
    assert attention_mask.shape == torch.Size([batch_size, seq_length])
    # Assert masking mockup works correctly
    assert torch.all(attention_mask.sum(-1) == masking.squeeze())
    # Assert correct device
    assert attention_mask.device == device
    return attention_mask


def get_float_attention_mask_mockup(
    batch_size: int, num_heads: int, seq_length: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a float attention mask mockup of specified size."""
    attention_mask = torch.rand(batch_size, num_heads, seq_length, seq_length).to(device).type(dtype)
    # Assert shape is (b, h, s, s)
    assert attention_mask.shape == torch.Size([batch_size, num_heads, seq_length, seq_length])
    # Assert correct device
    assert attention_mask.device == device
    return attention_mask
