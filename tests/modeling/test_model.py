"""Tests for the ``BertBlocksModel`` backbone forward pass.

Covers the three input regimes of :meth:`BertBlocksModel.forward`:

    1. padded (standard ``[batch, seq]`` + binary mask),
    2. internally unpadded (flash-attention mode strips/restores padding itself), and
    3. externally unpadded (HuggingFace/ModernBERT *varlen* convention) — the path used by
       seltz-neural's ``UnpaddingBackbone`` wrapper, where the caller supplies flat inputs plus
       ``cu_seq_lens_q`` / ``max_length_q`` and the model must neither unpad nor re-pad.
"""

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import is_flash_attn_2_available

from bertblocks.config import BertBlocksConfig
from bertblocks.modeling.model import BertBlocksModel

HIDDEN = 16
VOCAB = 100


def _config(attn_implementation: str) -> BertBlocksConfig:
    return BertBlocksConfig(
        vocab_size=VOCAB,
        max_sequence_length=32,
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_blocks=2,
        intermediate_size=32,
        block_pos_enc_kind="rope",
        attn_implementation=attn_implementation,
    )


def test_forward_padded_accepts_varlen_kwargs() -> None:
    """The padded path is unaffected and tolerates the new (ignored) HF varlen kwargs."""
    model = BertBlocksModel(_config("sdpa")).eval()
    input_ids = torch.randint(0, VOCAB, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    attention_mask[0, 3:] = 0

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=torch.arange(5).expand(2, 5),  # accepted, ignored
            unused_flash_kwarg="ignored",  # absorbed by **kwargs
        )

    assert isinstance(out, BaseModelOutput)
    assert out.last_hidden_state.shape == (2, 5, HIDDEN)


def test_forward_external_unpadding_arg_prep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supplying ``cu_seq_lens_q`` selects the external-varlen path.

    That path flattens ``input_ids``, forwards the caller's ``cu_seqlens`` (cast to int32) and
    ``max_seq_len``, never calls the internal ``unpad_input``, and returns a flat
    ``[1, total_tokens, hidden]`` ``last_hidden_state`` (no re-padding). ``_forward`` is stubbed so
    the test needs no flash kernels.
    """
    # Build an sdpa model (flash config can't be constructed on CPU) and simulate flash/unpadding
    # mode: if the external branch wrongly fell through to `elif self.unpadding`, the patched
    # `unpad_input` below would fire and fail the test.
    model = BertBlocksModel(_config("sdpa")).eval()
    model.unpadding = True

    def _fail_unpad(*_args, **_kwargs):
        raise AssertionError("internal unpad_input must not run in the external-unpadding path")

    monkeypatch.setattr(model, "unpad_input", _fail_unpad)

    captured: dict = {}

    def _fake_forward(input_ids, attention_mask, cu_seqlens, max_seq_len, token_type_ids, oa, ohs):
        captured.update(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
        )
        return torch.randn(input_ids.shape[0], HIDDEN), None, None, None

    monkeypatch.setattr(model, "_forward", _fake_forward)

    total = 5
    input_ids = torch.randint(0, VOCAB, (1, total))
    cu = torch.tensor([0, 2, total], dtype=torch.int64)  # int64 on purpose: must be cast to int32

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=torch.arange(total).unsqueeze(0),
            cu_seq_lens_q=cu,
            cu_seq_lens_k=cu,
            max_length_q=3,
            max_length_k=3,
        )

    assert captured["input_ids"].shape == (total,)  # flattened to 1D
    assert captured["attention_mask"] is None  # attention comes from cu_seqlens
    assert captured["cu_seqlens"].dtype == torch.int32
    assert captured["max_seq_len"] == 3
    assert isinstance(out, BaseModelOutput)
    assert out.last_hidden_state.shape == (1, total, HIDDEN)  # leading batch axis of 1 for caller.squeeze(0)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_flash_attn_2_available()),
    reason="external-unpadding path uses the flash varlen kernel (requires CUDA + flash-attn)",
)
def test_forward_external_unpadding_matches_padded() -> None:
    """Externally-unpadded forward agrees with the padded forward on the valid (non-pad) tokens."""
    torch.manual_seed(0)
    model = BertBlocksModel(_config("flash_attention_2")).cuda().eval().to(torch.bfloat16)

    batch, seq = 3, 6
    input_ids = torch.randint(0, VOCAB, (batch, seq), device="cuda")
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    attention_mask[0, 4:] = 0
    attention_mask[1, 5:] = 0

    # Reference: internal unpadding strips + restores padding, giving a padded [batch, seq, hidden].
    with torch.no_grad():
        padded = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    # Build the caller-side flat inputs + varlen metadata exactly as UnpaddingBackbone would.
    flat_ids, indices, cu_seqlens, max_seq_len = model.unpad_input(input_ids, attention_mask)
    with torch.no_grad():
        external = model(
            input_ids=flat_ids.reshape(1, -1),
            attention_mask=None,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seq_len,
            max_length_k=max_seq_len,
        ).last_hidden_state.squeeze(0)  # [total_tokens, hidden]

    padded_valid = padded.reshape(batch * seq, HIDDEN)[indices]
    torch.testing.assert_close(external.float(), padded_valid.float(), rtol=2e-2, atol=2e-2)
