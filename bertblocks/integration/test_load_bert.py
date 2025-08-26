import pytest
import torch
from pytest_dependency import depends
from transformers import AutoTokenizer, BertModel

from bertblocks.integration import from_bert_model

TEST_MODELS = ["bert-base-uncased", "bert-base-cased", "bert-large-uncased"]


@pytest.mark.parametrize("baseline_model", TEST_MODELS, scope="class")
class TestFromBertModel:
    """Test that Huggingface BERT and loaded BertBlocks implementations are equivalent in weights and output."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="class")
    def baseline_model_name(self, baseline_model):  # type: ignore
        """Return the current baseline model name for dynamic test dependencies."""
        yield baseline_model

    @pytest.fixture(scope="class")
    def hf_model(self, baseline_model):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        bert_model = BertModel.from_pretrained(baseline_model, add_pooling_layer=False).to(self.device)
        bert_model.eval()
        yield bert_model
        del bert_model

    @pytest.fixture(scope="class")
    def bb_model(self, baseline_model):  # type: ignore
        """Instantiate BertBlocks model as fixture."""
        bb_model = from_bert_model(baseline_model, add_pooling_layer=False).to(self.device)
        bb_model.eval()
        yield bb_model
        del bb_model

    @pytest.fixture(scope="class")
    def seq(self, baseline_model):  # type: ignore
        """Create sample sequence data."""
        tokenizer = AutoTokenizer.from_pretrained(baseline_model)
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
                    bb_model.encd.blocks[layer_idx].ffwd.uprj.weight,
                    hf_model.encoder.layer[layer_idx].intermediate.dense.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.uprj.bias,
                    hf_model.encoder.layer[layer_idx].intermediate.dense.bias,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.dprj.weight,
                    hf_model.encoder.layer[layer_idx].output.dense.weight,
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.dprj.bias,
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

    @pytest.mark.dependency
    def test_embedding(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the embedding layer."""
        depends(request, [f"TestFromBertModel::test_weights[{baseline_model_name}]"])

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
                torch.testing.assert_close(hf_model.embeddings(seq["input_ids"]), bb_model.embd(seq["input_ids"]))

    @pytest.mark.dependency
    def test_ffwd(self, request, baseline_model_name, subtests, hf_model, bb_model):  # type: ignore
        """Test the feed-forward layers individually."""
        depends(request, [f"TestFromBertModel::test_weights[{baseline_model_name}]"])

        inp = torch.rand((1, hf_model.config.max_position_embeddings, hf_model.config.hidden_size)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(hf_model.encoder.layer)):
                with subtests.test(f"layer_{layer_idx}"):
                    hf_out = hf_model.encoder.layer[layer_idx].intermediate(inp)
                    hf_out = hf_model.encoder.layer[layer_idx].output(hf_out, inp)

                    bb_out = bb_model.encd.blocks[layer_idx].ffwd(inp)
                    bb_out = bb_model.encd.blocks[layer_idx].post_norm_ffwd(bb_out + inp)

                    torch.testing.assert_close(hf_out, bb_out)

    @pytest.mark.dependency
    def test_attn(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the attention mechanism individually."""
        # depends(request, [f"TestFromBertModel::test_embedding[{baseline_model_name}]"])

        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

        bb_emb = bb_model.embd(seq["input_ids"])
        hf_emb = hf_model.embeddings(seq["input_ids"])
        torch.testing.assert_close(bb_emb, hf_emb)
        hf_msk = _prepare_4d_attention_mask_for_sdpa(
            seq["attention_mask"], dtype=hf_emb.dtype, tgt_len=seq["input_ids"].shape[1]
        )

        with torch.no_grad():
            for layer_idx in range(len(hf_model.encoder.layer)):
                with subtests.test(f"layer_{layer_idx}"):
                    hf_out = hf_model.encoder.layer[layer_idx].attention(hf_emb, attention_mask=hf_msk)[0]
                    # HF does residual and norm inside the attention, so we need to manually add it here, too
                    bb_out = bb_model.encd.blocks[layer_idx].attn(bb_emb, attention_mask=seq["attention_mask"])[0]
                    bb_out = bb_model.encd.blocks[layer_idx].post_norm_attn(bb_out + bb_emb)
                    torch.testing.assert_close(hf_out, bb_out)

    @pytest.mark.dependency
    def test_blocks(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the encoder blocks individually."""
        depends(
            request,
            [
                f"TestFromBertModel::test_ffwd[{baseline_model_name}]",
                f"TestFromBertModel::test_attn[{baseline_model_name}]",
            ],
        )

        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

        bb_emb = bb_model.embd(seq["input_ids"])
        hf_emb = hf_model.embeddings(seq["input_ids"])
        hf_msk = _prepare_4d_attention_mask_for_sdpa(
            seq["attention_mask"], dtype=hf_emb.dtype, tgt_len=seq["input_ids"].shape[1]
        )

        with torch.no_grad():
            for layer_idx in range(len(hf_model.encoder.layer)):
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(
                        hf_model.encoder.layer[layer_idx](hf_emb, attention_mask=hf_msk)[0],
                        bb_model.encd.blocks[layer_idx](bb_emb, attention_mask=seq["attention_mask"])[0],
                    )

    @pytest.mark.dependency
    def test_model(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the model end-to-end."""
        depends(
            request,
            [
                f"TestFromBertModel::test_weights[{baseline_model_name}]",
                f"TestFromBertModel::test_embedding[{baseline_model_name}]",
                f"TestFromBertModel::test_ffwd[{baseline_model_name}]",
                f"TestFromBertModel::test_attn[{baseline_model_name}]",
                f"TestFromBertModel::test_blocks[{baseline_model_name}]",
            ],
        )

        with torch.no_grad():
            hf_hidden = hf_model(seq["input_ids"], seq["attention_mask"], output_hidden_states=True).hidden_states
            bb_hidden = bb_model(seq["input_ids"], seq["attention_mask"], output_hidden_states=True).hidden_states

            for layer_idx, (hf_hidden_layer, bb_hidden_layer) in enumerate(zip(hf_hidden, bb_hidden, strict=False)):
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(hf_hidden_layer, bb_hidden_layer)
