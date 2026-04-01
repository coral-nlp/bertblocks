import pytest
import torch

from bertblocks.modeling.block import convert_to_4d_attention_mask
from bertblocks.modeling.position import AlibiPositionalEncoding, RotaryPositionalEncoding

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


class TestRotaryPositionalEncoding:
    """Test RotaryPositionalEncoding implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("rope_dim", [32, 64])
    @pytest.mark.parametrize("interleaved", [True, False])
    def test_init(self, rope_dim: int, interleaved: bool) -> None:
        """Test initialization and cache creation."""
        head_dim = 64
        enc = RotaryPositionalEncoding(rope_dim=rope_dim, head_dim=head_dim, interleaved=interleaved)
        assert enc._cos_cached is not None
        assert enc._sin_cached is not None
        assert enc._cos_cached.shape[0] == 512  # default max_seq_len
        assert enc._cos_cached.shape[1] == rope_dim // 2

    @pytest.mark.parametrize("rope_dim,head_dim", [(32, 64), (64, 64)])
    @pytest.mark.parametrize("interleaved", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    def test_forward_padded(
        self, rope_dim: int, head_dim: int, interleaved: bool, batch_size: int, seq_len: int
    ) -> None:
        """Test forward pass with padded input (attention_mask path)."""
        num_heads = 4
        enc = RotaryPositionalEncoding(
            rope_dim=rope_dim, head_dim=head_dim, interleaved=interleaved, max_seq_len=seq_len
        ).to(self.device)

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)

        q_out, k_out = enc(q, k, attention_mask=attention_mask)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert q_out.device == q.device
        assert k_out.device == k.device

    @pytest.mark.parametrize("rope_dim,head_dim", [(32, 64), (64, 64)])
    @pytest.mark.parametrize("interleaved", [True, False])
    def test_forward_unpadded(self, rope_dim: int, head_dim: int, interleaved: bool) -> None:
        """Test forward pass with unpadded input (cu_seqlens path)."""
        num_heads = 4
        seq_lens = [8, 12, 6]
        total_len = sum(seq_lens)
        max_seqlen = max(seq_lens)
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0)), device=self.device).to(
            torch.int32
        )

        enc = RotaryPositionalEncoding(
            rope_dim=rope_dim, head_dim=head_dim, interleaved=interleaved, max_seq_len=max_seqlen
        ).to(self.device)

        q = torch.randn(total_len, num_heads, head_dim, device=self.device)
        k = torch.randn(total_len, num_heads, head_dim, device=self.device)

        q_out, k_out = enc(q, k, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert q_out.device == q.device

    def test_error_no_mask_no_cu_seqlens(self) -> None:
        """Test that ValueError is raised when neither cu_seqlens nor attention_mask is provided."""
        enc = RotaryPositionalEncoding(rope_dim=32, head_dim=64).to(self.device)
        q = torch.randn(2, 16, 4, 64, device=self.device)
        k = torch.randn(2, 16, 4, 64, device=self.device)

        with pytest.raises(ValueError, match="Neither cu_seqlens nor attention_mask"):
            enc(q, k)

    def test_cache_device_update(self) -> None:
        """Test that cos/sin cache is updated to match input device and dtype."""
        enc = RotaryPositionalEncoding(rope_dim=32, head_dim=64, max_seq_len=32)
        # Cache starts on CPU
        assert enc._cos_cached.device == torch.device("cpu")

        enc = enc.to(self.device)
        q = torch.randn(2, 16, 4, 64, device=self.device)
        k = torch.randn(2, 16, 4, 64, device=self.device)
        mask = torch.ones(2, 16, device=self.device)

        q_out, _ = enc(q, k, attention_mask=mask)
        # Cache should now be on the same device as input
        assert enc._cos_cached.device == self.device
        assert q_out.device == self.device
