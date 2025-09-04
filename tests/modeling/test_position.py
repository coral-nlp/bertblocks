import pytest
import torch

from bertblocks.modeling.block import convert_to_4d_attention_mask
from bertblocks.modeling.position import AlibiPositionalEncoding

from ..test_utils import get_boolean_attention_mask_mockup, get_float_attention_mask_mockup


class TestAlibiPositionalEncoding:
    """Test AlibiPositionalEncoding implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("num_heads", [4, 8, 12, 16, 20, 24])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_get_slopes(self, num_heads: int, dtype: torch.dtype) -> None:
        """Test get_slopes with different number of heads and dtypes."""
        slopes = AlibiPositionalEncoding.get_slopes(num_heads, device=self.device, dtype=dtype)

        # Slopes should have the correct shape
        assert slopes.shape == (num_heads,)
        # Slopes should be allocated to the correct device
        assert slopes.device == self.device
        # Slopes should be of correct datatype
        assert slopes.dtype == dtype
        # All slopes should be positive or zero
        assert torch.all(slopes >= 0)

    @pytest.mark.parametrize("num_heads", [4, 8, 12, 16, 20, 24])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_init(self, num_heads, dtype) -> None:
        """Test that initialization properly registers slopes buffer."""
        pos_enc = AlibiPositionalEncoding(num_heads, device=self.device)
        # Object should have slopes attribute
        assert hasattr(pos_enc, "slopes")
        # Slopes should be of correct shape
        assert pos_enc.slopes.shape == (num_heads,)
        # Slopes should be registered as buffer
        assert "slopes" in pos_enc._buffers

    @pytest.mark.parametrize("num_heads", [8, 20])  # One power of two, one non power of two
    @pytest.mark.parametrize("batch_size", [2**i for i in range(5, 7)])
    @pytest.mark.parametrize("seq_len", [2**i for i in range(5, 7)])
    def test_forward_float_mask(self, num_heads: int, batch_size: int, seq_len: int) -> None:
        """Test forward with float attention mask."""
        pos_enc = AlibiPositionalEncoding(num_heads, device=self.device)
        attention_mask = torch.randn(batch_size, num_heads, seq_len, seq_len).to(self.device)

        attention_mask_out = pos_enc.forward(attention_mask)

        assert attention_mask_out.shape == attention_mask.shape
        assert attention_mask_out.device == self.device
        # ALiBi should add negative bias, so output should be <= input for positive inputs
        assert torch.all(attention_mask_out <= attention_mask)

    @pytest.mark.depends(
        ["tests::modeling::test_block::TestConvertTo4dAttentionMask::test_convert_to_4d_attention_mask_boolean"]
    )
    @pytest.mark.parametrize("num_heads", [8, 20])  # One power of two, one non power of two
    @pytest.mark.parametrize("batch_size", [2**i for i in range(5, 7)])
    @pytest.mark.parametrize("seq_len", [2**i for i in range(5, 7)])
    def test_forward_boolean_mask(self, num_heads: int, batch_size: int, seq_len: int) -> None:
        """Test forward with boolean attention mask."""
        pos_enc = AlibiPositionalEncoding(num_heads, device=self.device)

        # Create boolean mask
        attention_mask = get_boolean_attention_mask_mockup(batch_size, seq_len, device=self.device)
        attention_mask = convert_to_4d_attention_mask(attention_mask)
        attention_mask_out = pos_enc.forward(attention_mask)

        for h in range(num_heads):  # Test per dimension as original attention mask is not expanded to num_heads
            # Where mask was False, output should be -inf
            assert torch.all(attention_mask_out[:, h, :, :].unsqueeze(1)[~attention_mask] == -float("inf"))

            # Where mask was True, output should be finite (just ALiBi bias)
            assert torch.all(torch.isfinite(attention_mask_out[:, h, :, :].unsqueeze(1)[attention_mask]))

    @pytest.mark.parametrize("num_heads", [8, 20])  # One power of two, one non power of two
    @pytest.mark.parametrize("batch_size", [2**i for i in range(5, 7)])
    @pytest.mark.parametrize("seq_len", [2**i for i in range(5, 7)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_forward_boolean_mask(self, num_heads: int, batch_size: int, seq_len: int, dtype: torch.dtype) -> None:
        """Test forward with boolean attention mask."""
        pos_enc = AlibiPositionalEncoding(num_heads, device=self.device)

        attention_mask = get_float_attention_mask_mockup(
            batch_size, num_heads, seq_len, device=self.device, dtype=dtype
        )
        attention_mask_out = pos_enc.forward(attention_mask)

        # ALiBi will only ever decrease scores
        assert torch.all(attention_mask_out <= attention_mask)
