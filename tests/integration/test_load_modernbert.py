import pytest
import torch
from pytest_dependency import depends
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel
from transformers.masking_utils import create_bidirectional_mask

from bertblocks.integration import from_huggingface
from bertblocks.modeling.padding import pad_output

TEST_MODELS = ["answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"]


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available but flash-attn depends on it")
@pytest.mark.parametrize("baseline_model", TEST_MODELS, scope="class")
class TestFromModernBertModel:
    """Test equivalency of Huggingface ModernBERT and loaded BertBlocks implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="class")
    def baseline_model_name(self, baseline_model):  # type: ignore
        """Return the current baseline model name for dynamic test dependencies."""
        yield baseline_model

    @pytest.fixture(scope="class")
    def hf_model(self, baseline_model):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        config = ModernBertConfig.from_pretrained(baseline_model)
        config.deterministic_flash_attn = True
        config.reference_compile = False
        hf_model = ModernBertModel.from_pretrained(baseline_model, config=config).to(self.device)
        hf_model.eval()
        yield hf_model
        del hf_model

    @pytest.fixture(scope="class")
    def bb_model(self, baseline_model):  # type: ignore
        """Instantiate bertblocks model as fixture."""
        bertblocks_model = from_huggingface(baseline_model, add_pooling_layer=False).to(self.device)
        bertblocks_model.eval()
        yield bertblocks_model
        del bertblocks_model

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
            padding=True,
        ).to(self.device)
        del tokenizer

    @pytest.mark.dependency
    def test_weights(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test if weight copying worked."""
        with subtests.test(msg="layer_embedding"):
            torch.testing.assert_close(bb_model.embd.embd.weight, hf_model.embeddings.tok_embeddings.weight)

        assert len(bb_model.encd.blocks) == len(hf_model.layers)
        for layer_idx in range(len(hf_model.layers)):
            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_projection"):
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].attn.proj.weight,
                    hf_model.layers[layer_idx].attn.Wqkv.weight,
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_output_projection"):
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].attn.ffwd.weight, hf_model.layers[layer_idx].attn.Wo.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_ffwd"):
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.uprj.weight, hf_model.layers[layer_idx].mlp.Wi.weight
                )
                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].ffwd.dprj.weight, hf_model.layers[layer_idx].mlp.Wo.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_norms"):
                if layer_idx == 0:
                    # If first layer, the norm is in the embedding (pre-norm)
                    torch.testing.assert_close(
                        bb_model.encd.blocks[layer_idx].pre_norm_attn.weight, hf_model.embeddings.norm.weight
                    )
                else:
                    torch.testing.assert_close(
                        bb_model.encd.blocks[layer_idx].pre_norm_attn.weight,
                        hf_model.layers[layer_idx].attn_norm.weight,
                    )

                torch.testing.assert_close(
                    bb_model.encd.blocks[layer_idx].pre_norm_ffwd.weight,
                    hf_model.layers[layer_idx].mlp_norm.weight,
                )

        with subtests.test("final_norm"):
            torch.testing.assert_close(bb_model.norm.weight, hf_model.final_norm.weight)

    @pytest.mark.dependency
    def test_embedding(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the embedding layer."""
        depends(request, [f"TestFromModernBertModel::test_weights[{baseline_model_name}]"])

        from bertblocks.modeling.padding import unpad_input

        with torch.no_grad():
            input_ids, _, _, _ = unpad_input(seq["input_ids"], seq["attention_mask"])
            # The norm is not part of the embedding module in BertBlocks
            # So we have to manually apply it to get equivalent output
            torch.testing.assert_close(
                hf_model.embeddings(input_ids),
                bb_model.encd.blocks[0].pre_norm_attn(bb_model.embd(input_ids)),
            )

    @pytest.mark.dependency
    def test_ffwd(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the feed-forward layers individually."""
        depends(
            request,
            [
                f"TestFromModernBertModel::test_weights[{baseline_model_name}]",
                f"TestFromModernBertModel::test_embedding[{baseline_model_name}]",
            ],
        )

        from bertblocks.modeling.padding import unpad_input

        with torch.no_grad():
            input_ids, _, _, _ = unpad_input(seq["input_ids"], seq["attention_mask"])

            hf_emb = hf_model.embeddings(input_ids)
            bb_emb = bb_model.encd.blocks[0].pre_norm_attn(bb_model.embd(input_ids))

            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].mlp(hf_emb), bb_model.encd.blocks[layer_idx].ffwd(bb_emb)
                    )

    @pytest.mark.dependency
    def test_attn(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the attention mechanism individually."""
        depends(
            request,
            [
                f"TestFromModernBertModel::test_weights[{baseline_model_name}]",
                f"TestFromModernBertModel::test_embedding[{baseline_model_name}]",
            ],
        )

        with torch.no_grad():
            input_ids = seq["input_ids"]
            attention_mask = seq["attention_mask"]

            hf_emb = hf_model.embeddings(input_ids)
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            hf_attention_mask = create_bidirectional_mask(
                config=hf_model.config, inputs_embeds=hf_emb, attention_mask=seq["attention_mask"]
            )
            bb_emb = bb_model.encd.blocks[0].pre_norm_attn(bb_model.embd(input_ids))
            bb_attention_mask = create_bidirectional_mask(
                config=bb_model.config, inputs_embeds=bb_emb, attention_mask=seq["attention_mask"]
            )

            for layer_idx in range(len(hf_model.layers)):
                position_embeddings = hf_model.rotary_emb(hf_emb, position_ids, hf_model.config.layer_types[layer_idx])
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].attn(hf_emb, position_embeddings, hf_attention_mask)[0][
                            attention_mask.bool()
                        ],
                        bb_model.encd.blocks[layer_idx].attn(bb_emb, attention_mask=bb_attention_mask)[0][
                            attention_mask.bool()
                        ],
                    )

    @pytest.mark.dependency
    def test_norms(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the normalization individually."""
        depends(
            request,
            [
                f"TestFromModernBertModel::test_weights[{baseline_model_name}]",
                f"TestFromModernBertModel::test_embedding[{baseline_model_name}]",
            ],
        )

        from bertblocks.modeling.padding import unpad_input

        with torch.no_grad():
            input_ids, _, _, _ = unpad_input(seq["input_ids"], seq["attention_mask"])

            hf_emb = hf_model.embeddings(input_ids)
            bb_emb = bb_model.encd.blocks[0].pre_norm_attn(bb_model.embd(input_ids))

            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    # Attention norm
                    if layer_idx == 0:
                        torch.testing.assert_close(
                            hf_model.embeddings.norm(hf_emb), bb_model.encd.blocks[layer_idx].pre_norm_attn(bb_emb)
                        )
                    else:
                        torch.testing.assert_close(
                            hf_model.layers[layer_idx].attn_norm(hf_emb),
                            bb_model.encd.blocks[layer_idx].pre_norm_attn(bb_emb),
                        )
                    # MLP norm
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].mlp_norm(hf_emb),
                        bb_model.encd.blocks[layer_idx].pre_norm_ffwd(bb_emb),
                    )

            with subtests.test("final_norm"):
                torch.testing.assert_close(hf_model.final_norm(hf_emb), bb_model.norm(bb_emb))

    @pytest.mark.dependency
    def test_blocks(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the encoder blocks individually."""
        depends(
            request,
            [
                f"TestFromModernBertModel::test_weights[{baseline_model_name}]",
                f"TestFromModernBertModel::test_embedding[{baseline_model_name}]",
                f"TestFromModernBertModel::test_ffwd[{baseline_model_name}]",
                f"TestFromModernBertModel::test_norms[{baseline_model_name}]",
                f"TestFromModernBertModel::test_attn[{baseline_model_name}]",
            ],
        )

        with torch.no_grad():
            input_ids, attention_mask = seq["input_ids"], seq["attention_mask"]

            hf_emb = hf_model.embeddings(input_ids)
            bb_emb = bb_model.encd.blocks[0].pre_norm_attn(bb_model.embd(input_ids))

            hf_emb = hf_model.embeddings(input_ids)
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            hf_attention_mask = create_bidirectional_mask(
                config=hf_model.config, inputs_embeds=hf_emb, attention_mask=attention_mask
            )
            bb_emb = bb_model.encd.blocks[0].pre_norm_attn(bb_model.embd(input_ids))
            bb_attention_mask = create_bidirectional_mask(
                config=bb_model.config, inputs_embeds=bb_emb, attention_mask=attention_mask
            )

            for layer_idx in range(len(hf_model.layers)):
                position_embeddings = hf_model.rotary_emb(hf_emb, position_ids, hf_model.config.layer_types[layer_idx])
                with subtests.test(f"layer_{layer_idx}"):
                    if layer_idx == 0:
                        torch.testing.assert_close(
                            hf_model.layers[layer_idx](
                                hf_model.embeddings.norm(hf_emb),
                                attention_mask=hf_attention_mask,
                                position_embeddings=position_embeddings,
                            )[attention_mask.bool()],
                            bb_model.encd.blocks[layer_idx](bb_emb, attention_mask=bb_attention_mask)[0][
                                attention_mask.bool()
                            ],
                        )
                    else:
                        torch.testing.assert_close(
                            hf_model.layers[layer_idx](
                                hf_emb,
                                attention_mask=hf_attention_mask,
                                position_embeddings=position_embeddings,
                            )[attention_mask.bool()],
                            bb_model.encd.blocks[layer_idx](bb_emb, attention_mask=bb_attention_mask)[0][
                                attention_mask.bool()
                            ],
                        )

    @pytest.mark.dependency
    def test_model(self, request, baseline_model_name, subtests, seq, hf_model, bb_model):  # type: ignore
        """Test the entire model end-to-end."""
        depends(
            request,
            [
                f"TestFromModernBertModel::test_ffwd[{baseline_model_name}]",
                f"TestFromModernBertModel::test_attn[{baseline_model_name}]",
            ],
        )
        with torch.no_grad():
            hf_out = hf_model.forward(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state
            bb_out = bb_model.forward(seq["input_ids"], attention_mask=seq["attention_mask"])
            torch.testing.assert_close(hf_out[seq["attention_mask"].bool()], bb_out[0][seq["attention_mask"].bool()])
