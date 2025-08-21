import pytest
import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from bertblocks.compat import from_modernbert_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available but flash-attn depends on it")
class TestFromModernBertModel:
    """Test equivalency of Huggingface ModernBERT and loaded BertBlocks implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = "answerdotai/ModernBERT-base"

    @pytest.fixture(scope="class")
    def hf_model(self):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        config = ModernBertConfig.from_pretrained(self.baseline_model)
        config.deterministic_flash_attn = True
        config.reference_compile = False
        hf_model = ModernBertModel.from_pretrained(self.baseline_model, config=config)
        hf_model = hf_model.to(self.device)
        hf_model.eval()
        yield hf_model
        del hf_model

    @pytest.fixture(scope="class")
    def bertblocks_model(self):  # type: ignore
        """Instantiate bertblocks model as fixture."""
        bertblocks_model = from_modernbert_model(self.baseline_model, add_pooling_layer=False).to(self.device)
        bertblocks_model.eval()
        yield bertblocks_model
        del bertblocks_model

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
    def test_weights(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test if weight copying worked."""
        with subtests.test(msg="layer_embedding"):
            torch.testing.assert_close(bertblocks_model.embd.embd.weight, hf_model.embeddings.tok_embeddings.weight)

        assert len(bertblocks_model.encd.blocks) == len(hf_model.layers)
        for layer_idx in range(len(hf_model.layers)):
            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_projection"):
                torch.testing.assert_close(
                    bertblocks_model.encd.blocks[layer_idx].attn.proj.weight,
                    hf_model.layers[layer_idx].attn.Wqkv.weight,
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_output_projection"):
                torch.testing.assert_close(
                    bertblocks_model.encd.blocks[layer_idx].attn.ffwd.weight, hf_model.layers[layer_idx].attn.Wo.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_ffwd"):
                torch.testing.assert_close(
                    bertblocks_model.encd.blocks[layer_idx].ffwd.uprj.weight, hf_model.layers[layer_idx].mlp.Wi.weight
                )
                torch.testing.assert_close(
                    bertblocks_model.encd.blocks[layer_idx].ffwd.dprj.weight, hf_model.layers[layer_idx].mlp.Wo.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_norms"):
                if layer_idx == 0:
                    # If first layer, the norm is in the embedding (pre-norm)
                    torch.testing.assert_close(
                        bertblocks_model.encd.blocks[layer_idx].pre_norm_attn.weight, hf_model.embeddings.norm.weight
                    )
                else:
                    torch.testing.assert_close(
                        bertblocks_model.encd.blocks[layer_idx].pre_norm_attn.weight,
                        hf_model.layers[layer_idx].attn_norm.weight,
                    )

                torch.testing.assert_close(
                    bertblocks_model.encd.blocks[layer_idx].pre_norm_ffwd.weight,
                    hf_model.layers[layer_idx].mlp_norm.weight,
                )

        with subtests.test("final_norm"):
            torch.testing.assert_close(bertblocks_model.norm.weight, hf_model.final_norm.weight)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_embedding(self, seq, hf_model, bertblocks_model):  # type: ignore
        """Test the embedding layer."""
        from bertblocks.modeling.padding import unpad_input

        with torch.no_grad():
            input_ids, _, _, _ = unpad_input(seq["input_ids"], seq["attention_mask"])
            # The norm is not part of the embedding module in BertBlocks
            # So we have to manually apply it to get equivalent output
            torch.testing.assert_close(
                hf_model.embeddings(input_ids),
                bertblocks_model.encd.blocks[0].pre_norm_attn(bertblocks_model.embd(input_ids)),
            )

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_ffwd(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test the feed-forward layers individually."""
        inp = torch.rand((1, 8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].mlp(inp), bertblocks_model.encd.blocks[layer_idx].ffwd(inp)
                    )

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_rotary(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test the rotary positional encoding individually."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        qkv = torch.rand((8192, 3, 12, 64)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].attn.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len),
                        bertblocks_model.encd.blocks[layer_idx].attn.rotary_emb(
                            qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
                        ),
                    )

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights", "TestFromModernBertModel::test_rotary"])
    def test_attn(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test the attention mechanism individually."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        emb = torch.rand((8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].attn(emb, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)[0],
                        bertblocks_model.encd.blocks[layer_idx].attn(
                            emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
                        )[0],
                    )

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_norms(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test the normalization individually."""
        emb = torch.rand((8192, 768)).to(self.device)
        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    # Attention norm
                    if layer_idx == 0:
                        torch.testing.assert_close(
                            hf_model.embeddings.norm(emb), bertblocks_model.encd.blocks[layer_idx].pre_norm_attn(emb)
                        )
                    else:
                        torch.testing.assert_close(
                            hf_model.layers[layer_idx].attn_norm(emb),
                            bertblocks_model.encd.blocks[layer_idx].pre_norm_attn(emb),
                        )
                    # MLP norm
                    torch.testing.assert_close(
                        hf_model.layers[layer_idx].mlp_norm(emb),
                        bertblocks_model.encd.blocks[layer_idx].pre_norm_ffwd(emb),
                    )

            with subtests.test("final_norm"):
                torch.testing.assert_close(hf_model.final_norm(emb), bertblocks_model.norm(emb))

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_ffwd", "TestFromModernBertModel::test_attn"])
    def test_blocks_individual(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test the encoder blocks individually."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        emb = torch.rand((8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    if layer_idx == 0:
                        torch.testing.assert_close(
                            hf_model.layers[layer_idx](
                                hf_model.embeddings.norm(emb), cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
                            )[0],
                            bertblocks_model.encd.blocks[layer_idx](
                                emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
                            )[0],
                        )
                    else:
                        torch.testing.assert_close(
                            hf_model.layers[layer_idx](emb, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)[0],
                            bertblocks_model.encd.blocks[layer_idx](
                                emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
                            )[0],
                        )

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_blocks_individual"])
    def test_blocks_sequential(self, subtests, hf_model, bertblocks_model):  # type: ignore
        """Test the encoder blocks sequentially."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        hf_out = torch.rand((8192, 768)).to(self.device)
        bertblocks_out = hf_out.clone()

        with torch.no_grad():
            for layer_idx in range(len(hf_model.layers)):
                if layer_idx == 0:
                    hf_out = hf_model.embeddings.norm(hf_out)
                hf_out = hf_model.layers[layer_idx](hf_out, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)[0]
                bertblocks_out = bertblocks_model.encd.blocks[layer_idx](
                    bertblocks_out, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
                )[0]
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(hf_out, bertblocks_out)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available but flash-attn depends on it")
    @pytest.mark.dependency(
        depends=["TestFromModernBertModel::test_embedding", "TestFromModernBertModel::test_blocks_sequential"]
    )
    def test_model(self, seq, hf_model, bertblocks_model):  # type: ignore
        """Test the entire model end-to-end."""
        with torch.no_grad():
            torch.testing.assert_close(
                hf_model.forward(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state,
                bertblocks_model.forward(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state,
            )
