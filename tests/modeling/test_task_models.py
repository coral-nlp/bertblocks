"""Tests for task-specific BertBlocks model variants.

Each variant is tested with both padded (standard) and unpadded (flash-attention-style)
base model outputs by mocking the inner BertBlocksModel.
"""

from unittest.mock import MagicMock

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.model import (
    BertBlocksForEnhancedMaskedLM,
    BertBlocksForMaskedLM,
    BertBlocksForQuestionAnswering,
    BertBlocksForSequenceClassification,
    BertBlocksForTokenClassification,
    UnpaddedBaseModelOutput,
)

# ── Dimensions ──────────────────────────────────────────────────────────────
BATCH = 2
SEQ_LEN = 8
HIDDEN = 32
VOCAB = 100
NUM_CLASSES = 4

# Unpadded: last token of each sequence is treated as padding → 2 positions removed.
TOTAL_TOKENS = BATCH * SEQ_LEN - 2
# Flat indices of non-padding positions in the [BATCH * SEQ_LEN] tensor.
_INDICES = torch.cat(
    [
        torch.arange(SEQ_LEN - 1),  # positions 0-6  (seq 0, length 7)
        torch.arange(SEQ_LEN, BATCH * SEQ_LEN - 1),  # positions 8-14 (seq 1, length 7)
    ]
)  # shape [TOTAL_TOKENS]
# Cumulative sequence lengths for flash-attention style indexing.
# Seq 0: 7 tokens, seq 1: 7 tokens → [0, 7, 14].
_CU_SEQLENS = torch.tensor([0, SEQ_LEN - 1, TOTAL_TOKENS], dtype=torch.int32)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _padded_base_output(hidden_states: bool = False) -> BaseModelOutput:
    hs = tuple(torch.randn(BATCH, SEQ_LEN, HIDDEN) for _ in range(3)) if hidden_states else None
    return BaseModelOutput(
        last_hidden_state=torch.randn(BATCH, SEQ_LEN, HIDDEN),
        hidden_states=hs,
        attentions=None,
    )


def _unpadded_base_output(hidden_states: bool = False) -> UnpaddedBaseModelOutput:
    hs = tuple(torch.randn(TOTAL_TOKENS, HIDDEN) for _ in range(3)) if hidden_states else None
    return UnpaddedBaseModelOutput(
        last_hidden_state=torch.randn(TOTAL_TOKENS, HIDDEN),
        indices=_INDICES.clone(),
        seq_len=SEQ_LEN,
        batch_size=BATCH,
        cu_seqlens=_CU_SEQLENS.clone(),
        hidden_states=hs,
        attentions=None,
    )


def _mock_inner_model(task_model, return_value):
    """Patch the inner BertBlocksModel's forward to return a fixed output.

    PyTorch's Module.__setattr__ rejects non-Module values, so we patch
    the forward method directly instead of replacing the module.
    """
    task_model.model.forward = MagicMock(return_value=return_value)


def _consistent_outputs():
    """Return (padded_out, unpadded_out) with identical non-padding hidden states.

    The unpadded tensor is derived by gathering _INDICES from the flat padded tensor,
    so non-padding positions carry exactly the same hidden vectors in both outputs.
    Use this when asserting that padded and unpadded forward paths give the same result.
    """
    padded_hs = torch.randn(BATCH, SEQ_LEN, HIDDEN)
    unpadded_hs = padded_hs.reshape(BATCH * SEQ_LEN, HIDDEN)[_INDICES]  # [TOTAL_TOKENS, HIDDEN]
    padded_out = BaseModelOutput(last_hidden_state=padded_hs)
    unpadded_out = UnpaddedBaseModelOutput(
        last_hidden_state=unpadded_hs,
        indices=_INDICES.clone(),
        seq_len=SEQ_LEN,
        batch_size=BATCH,
        cu_seqlens=_CU_SEQLENS.clone(),
    )
    return padded_out, unpadded_out


@pytest.fixture
def base_config():
    return BertBlocksConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_blocks=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_sequence_length=SEQ_LEN,
        pad_token_id=0,
        mask_token_id=1,
    )


@pytest.fixture
def cls_config():
    return BertBlocksConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_blocks=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_sequence_length=SEQ_LEN,
        pad_token_id=0,
        mask_token_id=1,
        num_classes=NUM_CLASSES,
        problem_type="single_label_classification",
    )


def _input_ids():
    return torch.randint(2, VOCAB, (BATCH, SEQ_LEN))


# ── BertBlocksForMaskedLM ────────────────────────────────────────────────────


class TestBertBlocksForMaskedLM:
    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_logits_shape_no_labels(self, base_config, base_output_fn):
        model = BertBlocksForMaskedLM(base_config)
        _mock_inner_model(model, base_output_fn())
        out = model(_input_ids())
        assert out.loss is None
        assert out.logits.shape == (BATCH, SEQ_LEN, VOCAB)

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_logits_shape_with_labels(self, base_config, base_output_fn):
        model = BertBlocksForMaskedLM(base_config)
        _mock_inner_model(model, base_output_fn())
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        out = model(_input_ids(), labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0  # scalar
        assert out.logits.shape == (BATCH, SEQ_LEN, VOCAB)

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_hidden_states_passthrough(self, base_config, base_output_fn):
        model = BertBlocksForMaskedLM(base_config)
        _mock_inner_model(model, base_output_fn(hidden_states=True))
        out = model(_input_ids(), output_hidden_states=True)
        assert out.hidden_states is not None
        assert all(h.shape == (BATCH, SEQ_LEN, HIDDEN) for h in out.hidden_states)

    def test_loss_equivalence(self, base_config):
        """Padded and unpadded paths must give the same loss.

        In the padded path, cross-entropy ignores positions whose label is -100
        (PyTorch default ignore_index). Setting labels to -100 at the two padding
        positions makes the effective label set identical to the unpadded path.
        """
        padded_out, unpadded_out = _consistent_outputs()
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        # Mask out padding positions so the padded path skips them in the loss.
        padded_labels = labels.clone()
        padded_labels.view(-1)[[SEQ_LEN - 1, BATCH * SEQ_LEN - 1]] = -100

        model = BertBlocksForMaskedLM(base_config)

        _mock_inner_model(model, padded_out)
        padded_loss = model(_input_ids(), labels=padded_labels).loss

        _mock_inner_model(model, unpadded_out)
        unpadded_loss = model(_input_ids(), labels=labels).loss

        assert torch.allclose(padded_loss, unpadded_loss)


# ── BertBlocksForEnhancedMaskedLM ────────────────────────────────────────────


class TestBertBlocksForEnhancedMaskedLM:
    """EnhancedMaskedLM always operates on padded tensors (the extra block needs [B,S,H])."""

    def test_logits_shape_no_labels(self, base_config):
        model = BertBlocksForEnhancedMaskedLM(base_config)
        _mock_inner_model(model, _padded_base_output())
        # Also mock the extra block so it returns the expected padded shape.
        hidden = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        model.enhanced_masking_block.forward = MagicMock(return_value=(hidden, None))
        out = model(_input_ids())
        assert out.loss is None
        assert out.logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_logits_shape_with_labels(self, base_config):
        model = BertBlocksForEnhancedMaskedLM(base_config)
        _mock_inner_model(model, _padded_base_output())
        hidden = torch.randn(BATCH, SEQ_LEN, HIDDEN)
        model.enhanced_masking_block.forward = MagicMock(return_value=(hidden, None))
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        out = model(_input_ids(), labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert out.logits.shape == (BATCH, SEQ_LEN, VOCAB)


# ── BertBlocksForSequenceClassification ──────────────────────────────────────


class TestBertBlocksForSequenceClassification:
    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_logits_shape_no_labels(self, cls_config, base_output_fn):
        model = BertBlocksForSequenceClassification(cls_config)
        _mock_inner_model(model, base_output_fn())
        out = model(_input_ids())
        assert out.loss is None
        assert out.logits.shape == (BATCH, NUM_CLASSES)

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_logits_shape_with_labels(self, cls_config, base_output_fn):
        model = BertBlocksForSequenceClassification(cls_config)
        _mock_inner_model(model, base_output_fn())
        labels = torch.randint(0, NUM_CLASSES, (BATCH,))
        out = model(_input_ids(), labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert out.logits.shape == (BATCH, NUM_CLASSES)

    def test_loss_equivalence(self, cls_config):
        """Padded and unpadded paths must give the same loss.

        Both paths extract the CLS vector from position 0 of each sequence, which
        maps to the same hidden state regardless of packing strategy, so no label
        masking is needed.
        """
        padded_out, unpadded_out = _consistent_outputs()
        labels = torch.randint(0, NUM_CLASSES, (BATCH,))

        model = BertBlocksForSequenceClassification(cls_config)

        _mock_inner_model(model, padded_out)
        padded_loss = model(_input_ids(), labels=labels).loss

        _mock_inner_model(model, unpadded_out)
        unpadded_loss = model(_input_ids(), labels=labels).loss

        assert torch.allclose(padded_loss, unpadded_loss)

    def test_cls_from_cu_seqlens(self, cls_config):
        """CLS features must come from the first token of each sequence (cu_seqlens[:-1])."""
        model = BertBlocksForSequenceClassification(cls_config)
        base_out = _unpadded_base_output()
        _mock_inner_model(model, base_out)

        out = model(_input_ids())

        # Manually replicate what the forward should do and compare logits.
        cls_indices = _CU_SEQLENS[:-1].long()
        expected_cls = base_out.last_hidden_state[cls_indices]
        with torch.no_grad():
            expected_logits = model.classifier(model.head(expected_cls))

        assert torch.allclose(out.logits, expected_logits)

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_hidden_states_passthrough(self, cls_config, base_output_fn):
        model = BertBlocksForSequenceClassification(cls_config)
        _mock_inner_model(model, base_output_fn(hidden_states=True))
        out = model(_input_ids(), output_hidden_states=True)
        assert out.hidden_states is not None
        assert all(h.shape == (BATCH, SEQ_LEN, HIDDEN) for h in out.hidden_states)


# ── BertBlocksForTokenClassification ─────────────────────────────────────────


class TestBertBlocksForTokenClassification:
    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_logits_shape_no_labels(self, cls_config, base_output_fn):
        model = BertBlocksForTokenClassification(cls_config)
        _mock_inner_model(model, base_output_fn())
        out = model(_input_ids())
        assert out.loss is None
        assert out.logits.shape == (BATCH, SEQ_LEN, NUM_CLASSES)

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_logits_shape_with_labels(self, cls_config, base_output_fn):
        model = BertBlocksForTokenClassification(cls_config)
        _mock_inner_model(model, base_output_fn())
        labels = torch.randint(0, NUM_CLASSES, (BATCH, SEQ_LEN))
        out = model(_input_ids(), labels=labels)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert out.logits.shape == (BATCH, SEQ_LEN, NUM_CLASSES)

    def test_loss_equivalence(self, cls_config):
        """Padded and unpadded paths must give the same loss.

        Setting labels to -100 at padding positions in the padded path makes the
        effective label set identical to the unpadded path (which gathers only
        non-padding positions before computing the loss).
        """
        padded_out, unpadded_out = _consistent_outputs()
        labels = torch.randint(0, NUM_CLASSES, (BATCH, SEQ_LEN))
        padded_labels = labels.clone()
        padded_labels.view(-1)[[SEQ_LEN - 1, BATCH * SEQ_LEN - 1]] = -100

        model = BertBlocksForTokenClassification(cls_config)

        _mock_inner_model(model, padded_out)
        padded_loss = model(_input_ids(), labels=padded_labels).loss

        _mock_inner_model(model, unpadded_out)
        unpadded_loss = model(_input_ids(), labels=labels).loss

        assert torch.allclose(padded_loss, unpadded_loss)

    def test_padding_positions_zeroed(self, cls_config):
        """Logit values at padding positions should be 0 in unpadded mode."""
        model = BertBlocksForTokenClassification(cls_config)
        _mock_inner_model(model, _unpadded_base_output())
        out = model(_input_ids())
        # Positions 7 and 15 are the padding positions (see _INDICES definition).
        padding_positions = [SEQ_LEN - 1, BATCH * SEQ_LEN - 1]
        for flat_pos in padding_positions:
            b, s = flat_pos // SEQ_LEN, flat_pos % SEQ_LEN
            assert (out.logits[b, s] == 0).all()

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_hidden_states_passthrough(self, cls_config, base_output_fn):
        model = BertBlocksForTokenClassification(cls_config)
        _mock_inner_model(model, base_output_fn(hidden_states=True))
        out = model(_input_ids(), output_hidden_states=True)
        assert out.hidden_states is not None
        assert all(h.shape == (BATCH, SEQ_LEN, HIDDEN) for h in out.hidden_states)


# ── BertBlocksForQuestionAnswering ────────────────────────────────────────────


class TestBertBlocksForQuestionAnswering:
    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_span_logits_shape_no_positions(self, cls_config, base_output_fn):
        model = BertBlocksForQuestionAnswering(cls_config)
        _mock_inner_model(model, base_output_fn())
        out = model(_input_ids())
        assert out.loss is None
        assert out.start_logits.shape == (BATCH, SEQ_LEN)
        assert out.end_logits.shape == (BATCH, SEQ_LEN)

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_span_logits_shape_with_positions(self, cls_config, base_output_fn):
        model = BertBlocksForQuestionAnswering(cls_config)
        _mock_inner_model(model, base_output_fn())
        start_positions = torch.randint(0, SEQ_LEN, (BATCH,))
        end_positions = torch.randint(0, SEQ_LEN, (BATCH,))
        out = model(_input_ids(), start_positions=start_positions, end_positions=end_positions)
        assert out.loss is not None
        assert out.loss.ndim == 0
        assert out.start_logits.shape == (BATCH, SEQ_LEN)
        assert out.end_logits.shape == (BATCH, SEQ_LEN)

    def test_logits_equivalence_at_valid_positions(self, cls_config):
        """At non-padding positions, padded and unpadded paths must produce the same logits.

        Full loss equivalence cannot be tested here: the QA loss uses CE over all
        positions (softmax denominator includes padding slots), and the padded path
        does not zero out padding logits while the unpadded path fills them with 0
        via pad_output.  Comparing logits only at _INDICES positions sidesteps this
        and still verifies that the underlying representations are consistent.
        """
        padded_out, unpadded_out = _consistent_outputs()

        model = BertBlocksForQuestionAnswering(cls_config)

        _mock_inner_model(model, padded_out)
        padded_result = model(_input_ids())
        padded_start = padded_result.start_logits.reshape(-1)[_INDICES]
        padded_end = padded_result.end_logits.reshape(-1)[_INDICES]

        _mock_inner_model(model, unpadded_out)
        unpadded_result = model(_input_ids())
        unpadded_start = unpadded_result.start_logits.reshape(-1)[_INDICES]
        unpadded_end = unpadded_result.end_logits.reshape(-1)[_INDICES]

        assert torch.allclose(padded_start, unpadded_start)
        assert torch.allclose(padded_end, unpadded_end)

    def test_padding_positions_zeroed(self, cls_config):
        """Logit values at padding positions should be 0 in unpadded mode."""
        model = BertBlocksForQuestionAnswering(cls_config)
        _mock_inner_model(model, _unpadded_base_output())
        out = model(_input_ids())
        padding_positions = [SEQ_LEN - 1, BATCH * SEQ_LEN - 1]
        for flat_pos in padding_positions:
            b, s = flat_pos // SEQ_LEN, flat_pos % SEQ_LEN
            assert out.start_logits[b, s].item() == 0.0
            assert out.end_logits[b, s].item() == 0.0

    @pytest.mark.parametrize(
        "base_output_fn",
        [_padded_base_output, _unpadded_base_output],
        ids=["padded", "unpadded"],
    )
    def test_hidden_states_passthrough(self, cls_config, base_output_fn):
        model = BertBlocksForQuestionAnswering(cls_config)
        _mock_inner_model(model, base_output_fn(hidden_states=True))
        out = model(_input_ids(), output_hidden_states=True)
        assert out.hidden_states is not None
        assert all(h.shape == (BATCH, SEQ_LEN, HIDDEN) for h in out.hidden_states)
