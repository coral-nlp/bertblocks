import pytest
import torch
from transformers import AutoTokenizer, BertTokenizer

from bertblocks.training.objectives import (
    EnhancedMaskedLanguageModelingCollator,
    MaskedDiffusionCollator,
    MaskedLanguageModelingCollator,
)

# Parametrization
FORMAT_TYPES = ["standard", "packed"]
PRETOKENIZED = [True, False]
COLLATOR_CLS = [MaskedLanguageModelingCollator, EnhancedMaskedLanguageModelingCollator, MaskedDiffusionCollator]


@pytest.fixture(scope="module")
def tokenizer():
    """Test tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture(scope="module")
def seq():
    """Provide standard, untokenized sequence."""
    return [{"text": "The cat sat on the mat."}, {"text": "The bat had a hat."}]


@pytest.fixture(scope="module")
def seq_standard_tokenized():
    """Provide standard, non-packed, tokenized sequence."""
    return [
        {
            "input_ids": torch.LongTensor([101, 1996, 4937, 2938, 2006, 1996, 13523, 1012, 102, 0, 0, 0]),
            "token_type_ids": torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "attention_mask": torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
        },
        {
            "input_ids": torch.LongTensor([101, 1996, 7151, 2018, 1037, 6045, 1012, 102, 0, 0, 0, 0]),
            "token_type_ids": torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            "attention_mask": torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        },
    ]


@pytest.fixture(scope="module")
def seq_packed_tokenized():
    """Provide standard, packed, tokenized sequence."""
    return [
        {
            "input_ids": torch.LongTensor(
                [
                    [
                        101,
                        1996,
                        4937,
                        2938,
                        2006,
                        1996,
                        13523,
                        1012,
                        102,
                        101,
                        1996,
                        7151,
                        2018,
                        1037,
                        6045,
                        1012,
                        102,
                        0,
                        0,
                        0,
                    ]
                ]
            ),
            "token_type_ids": torch.LongTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            "attention_mask": torch.LongTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1]]),
        }
    ]


class TestPretrainingCollators:
    """Tests for pretraining (masked modeling) with both standard and packed formats."""

    @pytest.mark.parametrize("format_type", FORMAT_TYPES)
    @pytest.mark.parametrize("pretokenized", PRETOKENIZED)
    @pytest.mark.parametrize("collator_cls", COLLATOR_CLS)
    def test_output(
        self, tokenizer, seq, seq_standard_tokenized, seq_packed_tokenized, format_type, pretokenized, collator_cls
    ):
        """Test that collator masking yields correct output formats."""
        collator = collator_cls(tokenizer=tokenizer, max_sequence_length=128, pretokenized=pretokenized)
        if format_type == "packed":
            batch = seq_packed_tokenized
        elif format_type == "standard" and pretokenized:
            batch = seq_standard_tokenized
        else:
            batch = seq

        if format_type == "packed" and not pretokenized:
            with pytest.raises(ValueError, match="Expected raw string input when not running in pretokenized mode."):
                _ = collator(batch)
            return
        else:
            result = collator(batch)
            special_token_positions = torch.isin(
                result["labels"], torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id])
            )

            assert "input_ids" in result, "Collator should return input_ids"
            assert "attention_mask" in result, "Collator should return attention_mask"
            assert "labels" in result, "Collator should return labels"

    @pytest.mark.parametrize("format_type", FORMAT_TYPES)
    @pytest.mark.parametrize("pretokenized", PRETOKENIZED)
    @pytest.mark.parametrize("collator_cls", COLLATOR_CLS)
    def test_masking(
        self, tokenizer, seq, seq_standard_tokenized, seq_packed_tokenized, format_type, pretokenized, collator_cls
    ):
        """Test that collators don't mask padding tokens."""
        collator = MaskedLanguageModelingCollator(
            tokenizer=tokenizer,
            max_sequence_length=128,
            pretokenized=pretokenized,
            mlm_probability=1.0,  # Mask all
        )
        if format_type == "packed":
            batch = seq_packed_tokenized
        elif format_type == "standard" and pretokenized:
            batch = seq_standard_tokenized
        else:
            batch = seq

        if format_type == "packed" and not pretokenized:
            with pytest.raises(ValueError, match="Expected raw string input when not running in pretokenized mode."):
                _ = collator(batch)
            return
        else:
            res = collator(batch)
        padding_positions = res["attention_mask"][0] == 0 if format_type == "standard" else res["attention_mask"][0] < 0

        assert (res["labels"][0][padding_positions] == -100).all(), "Padding tokens should have -100 label"
