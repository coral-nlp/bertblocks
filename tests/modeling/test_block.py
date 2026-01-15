import pytest
import torch

from bertblocks.modeling.block import convert_to_4d_attention_mask
from tests.test_utils import get_boolean_attention_mask_mockup, get_float_attention_mask_mockup


class TestConvertTo4dAttentionMask:
    """Test mask conversion to 4d attention mask."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @pytest.mark.dependency
    @pytest.mark.parametrize("batch_size", [4, 16, 512])
    @pytest.mark.parametrize("seq_length", [4, 16, 512])
    def test_convert_to_4d_attention_mask_boolean(self, batch_size, seq_length):
        """Test mask conversion to 4d attention mask."""
        attention_mask = get_boolean_attention_mask_mockup(batch_size, seq_length, device=self.device)
        attention_mask_out = convert_to_4d_attention_mask(attention_mask)
        # Assert output has shape (b, 1, s, s), where 1 is broadcasted head dimension
        assert attention_mask_out.shape == torch.Size([batch_size, 1, seq_length, seq_length])
