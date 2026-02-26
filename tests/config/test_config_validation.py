"""Tests for BertBlocksConfig validation."""

import pytest

from bertblocks.config import BertBlocksConfig


class TestPositionalEncodingValidation:
    """Test validation of positional encoding configuration parameters."""

    def test_valid_block_pos_enc_kinds(self):
        """Test that all valid block positional encoding kinds are accepted."""
        valid_kinds = ["alibi", "rope", "learned", "learned_alibi"]
        for kind in valid_kinds:
            config = BertBlocksConfig(block_pos_enc_kind=kind)
            assert config.block_pos_enc_kind == kind

        # None is also valid
        config = BertBlocksConfig(block_pos_enc_kind=None)
        assert config.block_pos_enc_kind is None

    def test_valid_emb_pos_enc_kinds(self):
        """Test that all valid embedding positional encoding kinds are accepted."""
        valid_kinds = ["sinusoidal", "learned"]
        for kind in valid_kinds:
            config = BertBlocksConfig(emb_pos_enc_kind=kind)
            assert config.emb_pos_enc_kind == kind

        # None is also valid
        config = BertBlocksConfig(emb_pos_enc_kind=None)
        assert config.emb_pos_enc_kind is None

    def test_invalid_block_pos_enc_kind(self):
        """Test that invalid block positional encoding kinds raise ValueError."""
        invalid_kinds = [
            "",
            "invalid",
            "sinusoidal",
            "relative",
            "none",
            "none ",
        ]  # Note: "sinusoidal" and "none" not valid for block
        for kind in invalid_kinds:
            with pytest.raises(ValueError, match="invalid block_pos_enc_kind"):
                BertBlocksConfig(block_pos_enc_kind=kind)

    def test_invalid_emb_pos_enc_kind(self):
        """Test that invalid embedding positional encoding kinds raise ValueError."""
        invalid_kinds = [
            "",
            "invalid",
            "alibi",
            "rope",
            "none",
            "learned ",
        ]  # Note: "alibi"/"rope"/"none" not valid for emb
        for kind in invalid_kinds:
            with pytest.raises(ValueError, match="invalid emb_pos_enc_kind"):
                BertBlocksConfig(emb_pos_enc_kind=kind)

    def test_empty_string_block_pos_enc_kind(self):
        """Test that empty string for block_pos_enc_kind raises ValueError."""
        with pytest.raises(ValueError, match="invalid block_pos_enc_kind"):
            BertBlocksConfig(block_pos_enc_kind="")

    def test_empty_string_emb_pos_enc_kind(self):
        """Test that empty string for emb_pos_enc_kind raises ValueError."""
        with pytest.raises(ValueError, match="invalid emb_pos_enc_kind"):
            BertBlocksConfig(emb_pos_enc_kind="")

    def test_default_block_pos_enc_kind(self):
        """Test that default block_pos_enc_kind is 'alibi'."""
        config = BertBlocksConfig()
        assert config.block_pos_enc_kind == "alibi"

    def test_default_emb_pos_enc_kind(self):
        """Test that default emb_pos_enc_kind is None."""
        config = BertBlocksConfig()
        assert config.emb_pos_enc_kind is None

    def test_none_block_pos_enc_kind_accepted(self):
        """Test that None value for block_pos_enc_kind is accepted."""
        config = BertBlocksConfig(block_pos_enc_kind=None)
        assert config.block_pos_enc_kind is None

    def test_none_emb_pos_enc_kind_accepted(self):
        """Test that None value for emb_pos_enc_kind is accepted."""
        config = BertBlocksConfig(emb_pos_enc_kind=None)
        assert config.emb_pos_enc_kind is None

    def test_corrupted_config_from_json(self):
        """Test that corrupted configs loaded from JSON are caught."""
        import json
        import tempfile
        from pathlib import Path

        config_dict = {
            "model_type": "bertblocks",
            "vocab_size": 30522,
            "max_sequence_length": 1024,
            "hidden_size": 768,
            "num_blocks": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "block_pos_enc_kind": "",  # Empty string, like the bug report
            "block_pos_enc_kwargs": {"max_seq_len": 1024, "rope_dim": 64},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            with pytest.raises(ValueError, match="invalid block_pos_enc_kind"):
                BertBlocksConfig.from_pretrained(tmpdir)
