"""Tests for the multi-task MLM objective: pooling, bag-of-words, and isotropy losses."""

from types import SimpleNamespace

import torch

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.head import masked_mean_pool
from bertblocks.modeling.loss import BagOfWordsLoss, InBatchSimilarityLoss
from bertblocks.modeling.model import BertBlocksForMultiTaskMaskedLM
from bertblocks.modeling.utils import flatten_and_segment


def _tiny_config(**overrides):
    kwargs = {
        "hidden_size": 32,
        "num_blocks": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 50,
        "max_sequence_length": 16,
        "attn_implementation": "sdpa",
        "block_pos_enc_kind": "alibi",
        "pad_token_id": 0,
    }
    kwargs.update(overrides)
    return BertBlocksConfig(**kwargs)


class TestMaskedMeanPool:
    """Padded and unpadded/packed representations must pool identically."""

    def test_padded_equals_packed(self):
        """Padded and packed inputs produce identical pooled vectors."""
        hidden_size = 4
        doc0 = torch.randn(3, hidden_size)
        doc1 = torch.randn(2, hidden_size)

        padded = torch.zeros(2, 3, hidden_size)
        padded[0, :3] = doc0
        padded[1, :2] = doc1
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        pooled_padded = masked_mean_pool(padded, SimpleNamespace(cu_seqlens=None), attention_mask)

        flat = torch.cat([doc0, doc1], dim=0)
        cu_seqlens = torch.tensor([0, 3, 5])
        pooled_packed = masked_mean_pool(flat, SimpleNamespace(cu_seqlens=cu_seqlens), None)

        assert torch.allclose(pooled_padded, pooled_packed, atol=1e-6)
        assert torch.allclose(pooled_packed[0], doc0.mean(0), atol=1e-6)
        assert torch.allclose(pooled_packed[1], doc1.mean(0), atol=1e-6)


class TestBagOfWordsLoss:
    """Multi-hot target construction and segment splitting."""

    def test_build_multihot_segments_and_ignore(self):
        """Multi-hot target reflects presence per segment and drops ignored ids."""
        cu_seqlens = torch.tensor([0, 3, 5])
        segment_ids, lengths, num_segments, _ = flatten_and_segment(SimpleNamespace(cu_seqlens=cu_seqlens), None)
        assert num_segments == 2
        assert lengths.tolist() == [3, 2]

        token_ids = torch.tensor([5, 6, 6, 7, 2])
        target = BagOfWordsLoss.build_multihot(
            token_ids, segment_ids, num_segments, vocab_size=10, ignore_token_ids=[2]
        )
        assert target.shape == (2, 10)
        assert target[0].nonzero().flatten().tolist() == [5, 6]  # duplicates collapse to presence
        assert target[1].nonzero().flatten().tolist() == [7]  # id 2 is ignored
        assert set(target.unique().tolist()) <= {0.0, 1.0}

    def test_forward_is_bce(self):
        """The loss reduces to binary cross-entropy with logits."""
        loss_fn = BagOfWordsLoss()
        logits = torch.zeros(2, 5)
        target = torch.zeros(2, 5)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        assert torch.allclose(loss_fn(logits, target), expected)


class TestInBatchSimilarityLoss:
    """Isotropy loss on mean-pooled embeddings."""

    def test_identical_embeddings_high(self):
        """Identical embeddings yield maximal (1.0) similarity loss."""
        loss_fn = InBatchSimilarityLoss()
        pooled = torch.ones(4, 8)
        assert torch.allclose(loss_fn(pooled), torch.tensor(1.0), atol=1e-6)

    def test_orthogonal_embeddings_zero(self):
        """Orthogonal embeddings yield ~0 similarity loss."""
        loss_fn = InBatchSimilarityLoss()
        pooled = torch.eye(6)
        assert loss_fn(pooled) < 1e-6

    def test_single_sequence_zero(self):
        """A single sequence has no pairs, so the loss is 0."""
        loss_fn = InBatchSimilarityLoss()
        assert float(loss_fn(torch.randn(1, 8))) == 0.0


class TestMultiTaskModel:
    """End-to-end forward/backward of the multi-task model on the padded (sdpa) path."""

    def test_forward_backward(self):
        """Forward returns finite component losses and all heads receive gradients."""
        torch.manual_seed(0)
        model = BertBlocksForMultiTaskMaskedLM(
            _tiny_config(), bow_loss_weight=0.5, isotropy_loss_weight=0.3, ignore_token_ids=[0, 1, 2]
        )
        model.train()

        batch_size, seq_len = 4, 10
        input_ids = torch.randint(3, 50, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[0, 7:] = 0
        labels = torch.full((batch_size, seq_len), -100)
        labels[0, 2] = int(input_ids[0, 2])
        input_ids[0, 2] = 1  # replaced-with-mask position
        labels[1, 3] = int(input_ids[1, 3])
        input_ids[1, 3] = 1

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        for component in (out.loss, out.mlm_loss, out.bow_loss, out.isotropy_loss):
            assert component is not None
            assert torch.isfinite(component)
        assert out.logits.shape == (batch_size, seq_len, model.vocab_size)

        expected = out.mlm_loss + 0.5 * out.bow_loss + 0.3 * out.isotropy_loss
        assert torch.allclose(out.loss, expected)

        out.loss.backward()
        assert model.bow_decoder.weight.grad.abs().sum() > 0
        assert model.decoder.weight.grad.abs().sum() > 0

    def test_both_heads_tied_to_input_embeddings(self):
        """The MLM decoder and the BOW decoder both share storage with the input embeddings."""
        model = BertBlocksForMultiTaskMaskedLM(_tiny_config())
        input_embeddings = model.get_input_embeddings().weight
        assert model.config.tie_word_embeddings is True
        assert model.decoder.weight.data_ptr() == input_embeddings.data_ptr()
        assert model.bow_decoder.weight.data_ptr() == input_embeddings.data_ptr()

    def test_no_labels_returns_none_loss(self):
        """Without labels the model returns logits but no losses."""
        model = BertBlocksForMultiTaskMaskedLM(_tiny_config())
        model.eval()
        input_ids = torch.randint(3, 50, (2, 6))
        attention_mask = torch.ones(2, 6, dtype=torch.long)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        assert out.loss is None
        assert out.mlm_loss is None
        assert out.logits.shape == (2, 6, model.vocab_size)
