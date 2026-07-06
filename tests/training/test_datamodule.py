"""Tests for the pretraining DataModule, focused on held-out validation slicing."""

import json

from bertblocks.training.modules import BertBlocksPretrainingDataModule


def _write_jsonl(directory, num_docs):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "train.jsonl"
    with path.open("w") as handle:
        for i in range(num_docs):
            handle.write(json.dumps({"text": f"document number {i} with some filler tokens"}) + "\n")
    return str(directory)


class TestHeldOutValidation:
    """`held_out_val_size` carves a disjoint validation slice from the training stream."""

    def test_streaming_held_out_is_disjoint(self, tmp_path):
        """Validation is the first N examples; training skips them, so they do not overlap."""
        data_dir = _write_jsonl(tmp_path / "data", num_docs=64)
        held_out = 8
        dm = BertBlocksPretrainingDataModule(
            train_dataset_name_or_path=data_dir,
            pretrained_tokenizer_name_or_path="bert-base-uncased",
            objective="multitask_mlm",
            file_format="json",
            text_column="text",
            max_sequence_length=32,
            train_batch_size=4,
            streaming=True,
            packing=False,
            held_out_val_size=held_out,
        )
        dm.prepare_data()
        dm.setup("fit")

        val_texts = [ex["text"] for ex in dm.val_dataset[0]]
        train_texts = [ex["text"] for _, ex in zip(range(30), dm.train_dataset, strict=False)]

        assert len(val_texts) == held_out
        assert val_texts == [f"document number {i} with some filler tokens" for i in range(held_out)]
        assert set(val_texts).isdisjoint(train_texts)
        # Training resumes right after the held-out slice.
        assert train_texts[0] == f"document number {held_out} with some filler tokens"
