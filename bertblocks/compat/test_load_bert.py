import pytest
import torch
from transformers import AutoTokenizer, BertModel

from bertblocks.compat.load_bert import from_bert_model


@pytest.mark.skip(reason="Under development")
class TestFromBertModel:
    """Test that Huggingface BERT and loaded BertBlocks implementations are equivalent in weights and output."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = "bert-base-uncased"

    @pytest.fixture(scope="class")
    def hf_model(self):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        bert_model = BertModel.from_pretrained(self.baseline_model, add_pooling_layer=False).to(self.device)
        bert_model.eval()
        yield bert_model
        del bert_model

    @pytest.fixture(scope="class")
    def bb_model(self):  # type: ignore
        """Instantiate BertBlocks model as fixture."""
        bb_model = from_bert_model(self.baseline_model, add_pooling_layer=False).to(self.device)
        bb_model.eval()
        yield bb_model
        del bb_model

    @pytest.fixture(scope="class")
    def seq(self):  # type: ignore
        """Create sample sequence data."""
        tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        yield tokenizer(
            [
                "The cat sat on the mat.",
                "He didn't know why she left.",
                "Can you believe it's already August?",
                "Running late, she skipped breakfast.",
                "Wow, that's an incredibly fast response!",
            ],
            return_tensors="pt",
            padding="max_length",
        ).to(self.device)
        del tokenizer

    @pytest.mark.dependency
    def test_weights(self, subtests, hf_model, bb_model):  # type: ignore
        """Test if weight copying worked."""
        with subtests.test(msg="layer_embedding"):
            torch.testing.assert_close(bb_model.embd.embd.weight, hf_model.embeddings.word_embeddings.weight)
            torch.testing.assert_close(bb_model.embd.pose.embd.weight, hf_model.embeddings.position_embeddings.weight)
            torch.testing.assert_close(bb_model.embd.norm.weight, hf_model.embeddings.LayerNorm.weight)
            torch.testing.assert_close(bb_model.embd.norm.bias, hf_model.embeddings.LayerNorm.bias)
            torch.testing.assert_close(bb_model.embd.tokt.embd.weight, hf_model.embeddings.token_type_embeddings.weight)

        assert len(bb_model.encd.blocks) == len(hf_model.encoder.layer)
        for layer_idx in range(len(hf_model.encoder.layer)):
            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_projection"):
                qw, kw, vw = bb_model.encd.blocks[layer_idx].attn.proj.weight.chunk(3, dim=0)
                torch.testing.assert_close(qw, hf_model.encoder.layer[layer_idx].attention.self.query.weight)
                torch.testing.assert_close(kw, hf_model.encoder.layer[layer_idx].attention.self.key.weight)
                torch.testing.assert_close(vw, hf_model.encoder.layer[layer_idx].attention.self.value.weight)

            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_bias"):
                qb, kb, vb = bb_model.encd.blocks[layer_idx].attn.proj.bias.chunk(3, dim=0)
                torch.testing.assert_close(qb, hf_model.encoder.layer[layer_idx].attention.self.query.bias)
                torch.testing.assert_close(kb, hf_model.encoder.layer[layer_idx].attention.self.key.bias)
                torch.testing.assert_close(vb, hf_model.encoder.layer[layer_idx].attention.self.value.bias)

            with subtests.test(f"layer_encoder_block_{layer_idx}_output_projection"):
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].attn.ffwd.weight,
                    hf_model.encoder.layer[layer_idx].attention.output.dense.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].attn.ffwd.bias,
                    hf_model.encoder.layer[layer_idx].attention.output.dense.bias,
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_ffwd"):
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.Uprj.weight,
                    hf_model.encoder.layer[layer_idx].intermediate.dense.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.Uprj.bias,
                    hf_model.encoder.layer[layer_idx].intermediate.dense.bias,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.Dprj.weight,
                    hf_model.encoder.layer[layer_idx].output.dense.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.Dprj.bias,
                    hf_model.encoder.layer[layer_idx].output.dense.bias,
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_norms"):
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].post_norm_attn.weight,
                    hf_model.encoder.layer[layer_idx].attention.output.LayerNorm.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].post_norm_attn.bias,
                    hf_model.encoder.layer[layer_idx].attention.output.LayerNorm.bias,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].post_norm_ffwd.weight,
                    hf_model.encoder.layer[layer_idx].output.LayerNorm.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].post_norm_ffwd.bias,
                    hf_model.encoder.layer[layer_idx].output.LayerNorm.bias,
                )

    @pytest.mark.dependency(depends=["TestFromBertModel::test_weights"])
    def test_embedding(self, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the embedding layer."""
        position_ids = torch.arange(0, seq["input_ids"].shape[1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with subtests.test("word_embeddings"):
                torch.testing.assert_close(
                    hf_model.embeddings.word_embeddings(seq["input_ids"]),
                    bb_model.embd.embd(seq["input_ids"]),
                )
            with subtests.test("token_type_embeddings"):
                torch.testing.assert_close(
                    hf_model.embeddings.token_type_embeddings(seq["token_type_ids"]),
                    bb_model.embd.tokt.embd(seq["token_type_ids"]),
                )
            with subtests.test("position_embeddings"):
                torch.testing.assert_close(
                    hf_model.embeddings.position_embeddings(position_ids),
                    bb_model.embd.pose.embd(position_ids),
                )

            with subtests.test("norm"):
                torch.testing.assert_close(
                    hf_model.embeddings.LayerNorm(hf_model.embeddings.word_embeddings(seq["input_ids"])),
                    bb_model.embd.norm(bb_model.embd.embd(seq["input_ids"])),
                )

            with subtests.test("end_to_end"):
                from bertblocks.modeling.padding import pad_output, unpad_input

                B, S = seq["input_ids"].shape
                input_ids, indices, cu_seqlens, max_seq_len = unpad_input(seq["input_ids"], seq["attention_mask"])
                bb_out = bb_model.embd(input_ids, cu_seqlens=cu_seqlens)
                bb_out = pad_output(bb_out, indices, B, S)
                # We have to set attention masked values to 0, since the unpadding inserts zeros
                hf_out = hf_model.embeddings(seq["input_ids"])
                hf_out = hf_out * seq["attention_mask"].unsqueeze(-1)

                torch.testing.assert_close(hf_out, bb_out)

    @pytest.mark.dependency(depends=["TestFromBertModel::test_weights"])
    def test_ffwd(self, subtests, hf_model, bb_model):  # type: ignore
        """Test the feed-forward layers individually."""
        inp = torch.rand((1, 8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(hf_model.encoder.layer)):
                with subtests.test(f"layer_{layer_idx}"):
                    hf_out = hf_model.encoder.layer[layer_idx].intermediate(inp)
                    hf_out = hf_model.encoder.layer[layer_idx].output(hf_out, inp)

                    bb_out = bb_model.encd.blocks[layer_idx].ffwd(inp)
                    bb_out = bb_model.encd.blocks[layer_idx].post_norm_ffwd(bb_out + inp)
                    torch.testing.assert_close(hf_out, bb_out)

    def test_attn(self, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the attention mechanism individually."""
        from bertblocks.modeling.padding import pad_output, unpad_input

        B, S = seq["input_ids"].shape
        input_ids, indices, cu_seqlens, max_seq_len = unpad_input(seq["input_ids"], seq["attention_mask"])
        bb_emb = bb_model.embd(input_ids, cu_seqlens=cu_seqlens)
        hf_emb = hf_model.embeddings(seq["input_ids"], seq["attention_mask"])

        with torch.no_grad():
            for layer_idx in range(len(hf_model.encoder.layer)):
                with subtests.test(f"layer_{layer_idx}"):
                    hf_out = hf_model.encoder.layer[layer_idx].attention(
                        hf_emb, attention_mask=seq["attention_mask"].bool()
                    )
                    hf_out = hf_out[0] * seq["attention_mask"].unsqueeze(-1)

                    bb_out = bb_model.encd.blocks[layer_idx].attn(
                        bb_emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
                    )[0]
                    bb_out = bb_model.encd.blocks[layer_idx].post_norm_attn(bb_out)
                    bb_out = pad_output(bb_out, indices, B, S)
                    torch.testing.assert_close(hf_out, bb_out)

    @pytest.mark.dependency(depends=["TestFromBertModel::test_ffwd", "TestFromBertModel::test_attn"])
    def test_blocks_individual(self, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the encoder blocks individually."""
        from bertblocks.modeling.padding import pad_output, unpad_input

        B, S = seq["input_ids"].shape
        input_ids, indices, cu_seqlens, max_seq_len = unpad_input(seq["input_ids"], seq["attention_mask"])
        bb_emb = bb_model.embd(input_ids, cu_seqlens=cu_seqlens)

        hf_emb = hf_model.embeddings(input_ids, seq["attention_mask"])

        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    hf_out = hf_model.layer[layer_idx](hf_emb)
                    hf_out = hf_out * seq["attention_mask"].unsqueeze(-1)
                    bb_out = bb_model.encd.blocks[layer_idx](bb_emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)[0]
                    bb_out = pad_output(bb_out, indices, B, S)
                    torch.testing.assert_close(hf_out, bb_out)

    @pytest.mark.dependency(depends=["TestFromBertModel::test_weights"])
    def test_encoder(self, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the encoder stacks."""
        with torch.no_grad():
            bert_hidden = hf_model(
                seq["input_ids"], seq["attention_mask"].bool(), output_hidden_states=True
            ).hidden_states
            berkit_hidden = bb_model(
                seq["input_ids"], seq["attention_mask"].bool(), output_hidden_states=True
            ).hidden_states

            for layer_idx, (bhs, phs) in enumerate(zip(bert_hidden, berkit_hidden, strict=False)):
                with subtests.test(f"layer_encoder_block_{layer_idx}"):
                    torch.testing.assert_close(bhs * seq["attention_mask"].unsqueeze(-1), phs)
