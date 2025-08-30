import unittest
import torch

from bertblocks.modeling.position import AlibiPositionalEncoding


class AlibiPositionalEncodingTest(unittest.TestCase):

    def test_forward(self, batch_size: int = 32, num_heads: int = 12, seq_len: int = 64) -> None:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pos_enc = AlibiPositionalEncoding(num_heads)
        attention_mask = torch.randn(batch_size, num_heads, seq_len, seq_len).to(device)

        attention_mask_out = pos_enc.forward(attention_mask)

        self.assertTrue(torch.all(attention_mask_out <= attention_mask).item())

    def test_forward_boolean_mask(self, batch_size: int = 32, num_heads: int = 12, seq_len: int = 64) -> None:

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pos_enc = AlibiPositionalEncoding(num_heads)

        attention_mask = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attention_mask = (attention_mask > attention_mask.mean(dim=None)).to(device)
        attention_mask_out = pos_enc.forward(attention_mask)

        self.assertTrue(torch.all(attention_mask_out <= attention_mask).item())
