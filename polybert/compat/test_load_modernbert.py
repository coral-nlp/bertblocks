import pytest
import torch
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

from polybert.compat.load_modernbert import from_modernbert_model


class TestFromModernBertModel:
    """Test equivalency of Huggingface ModernBERT and loaded PolyBERT implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = "answerdotai/ModernBERT-base"

    @pytest.fixture(scope="class")
    def bert_model(self):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        config = ModernBertConfig.from_pretrained(self.baseline_model)
        config.deterministic_flash_attn = True
        config.reference_compile = False
        bert_model = ModernBertModel.from_pretrained(self.baseline_model, config=config)
        bert_model = bert_model.to(self.device)
        bert_model.eval()
        yield bert_model
        del bert_model

    @pytest.fixture(scope="class")
    def poly_model(self):  # type: ignore
        """Instantiate PolyBERT model as fixture."""
        poly_model = from_modernbert_model(self.baseline_model, add_pooling_layer=False).to(self.device)
        poly_model.eval()
        yield poly_model
        del poly_model

    @pytest.fixture(scope="class")
    def tokenizer(self):  # type: ignore
        """Instantiate Huggingface BERT tokenizer as fixture."""
        tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        yield tokenizer
        del tokenizer

    @pytest.mark.dependency
    def test_weights(self, subtests, bert_model, poly_model):  # type: ignore
        """Test if weight copying worked."""
        with subtests.test(msg="layer_embedding"):
            torch.testing.assert_close(poly_model.embd.embd.weight, bert_model.embeddings.tok_embeddings.weight)

        assert len(poly_model.encd.blocks) == len(bert_model.layers)
        for layer_idx in range(len(bert_model.layers)):
            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_projection"):
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].attn.proj.weight, bert_model.layers[layer_idx].attn.Wqkv.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_output_projection"):
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].attn.ffwd.weight, bert_model.layers[layer_idx].attn.Wo.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_ffwd"):
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].ffwd.Uprj.weight, bert_model.layers[layer_idx].mlp.Wi.weight
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].ffwd.Dprj.weight, bert_model.layers[layer_idx].mlp.Wo.weight
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_norms"):
                if layer_idx == 0:
                    # If first layer, the norm is in the embedding (pre-norm)
                    torch.testing.assert_close(
                        poly_model.encd.blocks[layer_idx].pre_norm_attn.weight, bert_model.embeddings.norm.weight
                    )
                else:
                    torch.testing.assert_close(
                        poly_model.encd.blocks[layer_idx].pre_norm_attn.weight,
                        bert_model.layers[layer_idx].attn_norm.weight,
                    )

                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].pre_norm_ffwd.weight,
                    bert_model.layers[layer_idx].mlp_norm.weight,
                )

        with subtests.test("final_norm"):
            torch.testing.assert_close(poly_model.norm.weight, bert_model.final_norm.weight)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_embedding(self, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the embedding layer."""
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length").to(self.device)

        from polybert.modeling.padding import unpad_input

        with torch.no_grad():
            input_ids, _, _, _ = unpad_input(seq["input_ids"], seq["attention_mask"])
            emb_bert = bert_model.embeddings(input_ids)
            emb_poly = poly_model.embd(input_ids)
            # The norm is not part of the embedding module in polybert
            # So we have to manually apply it to get equivalent output
            emb_poly = poly_model.encd.blocks[0].pre_norm_attn(emb_poly)

        torch.testing.assert_close(emb_bert, emb_poly)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_ffwd(self, subtests, bert_model, poly_model):  # type: ignore
        """Test the feed-forward layers individually."""
        inp = torch.rand((1, 8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(bert_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    bert_out = bert_model.layers[layer_idx].mlp(inp)
                    poly_out = poly_model.encd.blocks[layer_idx].ffwd(inp)
                    torch.testing.assert_close(bert_out, poly_out, msg=f"Layer {layer_idx} ffwd not matching.")

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_rotary(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the rotary positional encoding individually."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        qkv = torch.rand((8192, 3, 12, 64)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(bert_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    bert_out = bert_model.layers[layer_idx].attn.rotary_emb(
                        qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
                    )
                    poly_out = poly_model.encd.blocks[layer_idx].attn.rotary_emb(
                        qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
                    )
                    torch.testing.assert_close(bert_out, poly_out)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights", "TestFromModernBertModel::test_rotary"])
    def test_attn(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the attention mechanism individually."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        emb = torch.rand((8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(bert_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    bert_out = bert_model.layers[layer_idx].attn(emb, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
                    poly_out = poly_model.encd.blocks[layer_idx].attn(
                        emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len
                    )
                    torch.testing.assert_close(bert_out[0], poly_out[0])

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_norms(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the normalization individually."""
        emb = torch.rand((8192, 768)).to(self.device)
        with torch.no_grad():
            for layer_idx in range(len(bert_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    # Attention norm
                    if layer_idx == 0:
                        bert_out = bert_model.embeddings.norm(emb)
                    else:
                        bert_out = bert_model.layers[layer_idx].attn_norm(emb)
                    poly_out = poly_model.encd.blocks[layer_idx].pre_norm_attn(emb)
                    torch.testing.assert_close(bert_out, poly_out)
                    # MLP norm
                    bert_out = bert_model.layers[layer_idx].mlp_norm(emb)
                    poly_out = poly_model.encd.blocks[layer_idx].pre_norm_ffwd(emb)
                    torch.testing.assert_close(bert_out, poly_out)

            with subtests.test("final_norm"):
                bert_out = bert_model.final_norm(emb)
                poly_out = poly_model.norm(emb)
                torch.testing.assert_close(bert_out, poly_out)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_ffwd", "TestFromModernBertModel::test_attn"])
    def test_blocks_individual(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the encoder blocks individually."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        emb = torch.rand((8192, 768)).to(self.device)

        with torch.no_grad():
            for layer_idx in range(len(bert_model.layers)):
                with subtests.test(f"layer_{layer_idx}"):
                    if layer_idx == 0:
                        bert_out = bert_model.embeddings.norm(emb)
                        bert_out = bert_model.layers[layer_idx](
                            bert_out, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
                        )[0]
                    else:
                        bert_out = bert_model.layers[layer_idx](emb, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)[0]

                    poly_out = poly_model.encd.blocks[layer_idx](emb, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)[0]
                    torch.testing.assert_close(bert_out, poly_out)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_blocks_individual"])
    def test_blocks_sequential(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the encoder blocks sequentially."""
        cu_seqlens = torch.Tensor([0, 8192]).to(self.device, dtype=torch.int32)
        max_seq_len = 8192
        bert_out = torch.rand((8192, 768)).to(self.device)
        poly_out = bert_out.clone()

        with torch.no_grad():
            for layer_idx in range(len(bert_model.layers)):
                if layer_idx == 0:
                    bert_out = bert_model.embeddings.norm(bert_out)
                bert_out = bert_model.layers[layer_idx](bert_out, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)[0]
                poly_out = poly_model.encd.blocks[layer_idx](poly_out, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)[
                    0
                ]
                with subtests.test(f"layer_{layer_idx}"):
                    torch.testing.assert_close(bert_out, poly_out)

    def test_encoder(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the encoder blocks sequentially."""
        seq = tokenizer(
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

        from polybert.modeling.padding import unpad_input

        with torch.no_grad():
            input_ids, _, cu_seqlens, max_seq_len = unpad_input(seq["input_ids"], seq["attention_mask"])
            bert_out = bert_model.embeddings(input_ids)
            poly_out = poly_model.embd(input_ids)

            for layer_idx in range(len(bert_model.layers)):
                bert_out = bert_model.layers[layer_idx](bert_out, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)[0]

            poly_out = poly_model.encd(poly_out, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)[0]

            bert_out = bert_model.final_norm(poly_out)
            poly_out = poly_model.norm(poly_out)

        torch.testing.assert_close(bert_out, poly_out)

    @pytest.mark.dependency(
        depends=["TestFromModernBertModel::test_embedding", "TestFromModernBertModel::test_blocks_sequential"]
    )
    def test_model(self, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the entire model end-to-end."""
        seq = tokenizer(
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

        with torch.no_grad():
            hidden_bert = bert_model.forward(seq["input_ids"], attention_mask=seq["attention_mask"])
            hidden_poly = poly_model.forward(seq["input_ids"], attention_mask=seq["attention_mask"])

        torch.testing.assert_close(hidden_bert.last_hidden_state, hidden_poly.last_hidden_state)
