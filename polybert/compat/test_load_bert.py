import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from polybert.compat.load_bert import from_bert_model


class TestFromBertModel:
    """Test that Huggingface BERT and loaded polybert BERT implementations are equivalent in weights and output."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = "bert-base-uncased"

    @pytest.fixture(scope="class")
    def bert_model(self):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        config = BertConfig.from_pretrained(self.baseline_model)
        bert_model = BertModel(config, add_pooling_layer=False).to(self.device)
        bert_model.eval()
        yield bert_model
        del bert_model

    @pytest.fixture(scope="class")
    def poly_model(self):  # type: ignore
        """Instantiate PolyBERT model as fixture."""
        poly_model = from_bert_model(self.baseline_model, add_pooling_layer=False).to(self.device)
        poly_model.eval()
        yield poly_model
        del poly_model

    @pytest.fixture(scope="class")
    def tokenizer(self):  # type: ignore
        """Instantiate Huggingface BERT tokenizer as fixture."""
        tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        yield tokenizer

    @pytest.mark.dependency
    def test_weights(self, subtests, bert_model, poly_model):  # type: ignore
        """Test if weight copying worked."""
        with subtests.test(msg="layer_embedding"):
            torch.testing.assert_close(poly_model.embd.embd.weight, bert_model.embeddings.word_embeddings.weight)
            torch.testing.assert_close(
                poly_model.embd.pose.embd.weight, bert_model.embeddings.position_embeddings.weight
            )
            torch.testing.assert_close(poly_model.embd.norm.weight, bert_model.embeddings.LayerNorm.weight)
            torch.testing.assert_close(poly_model.embd.norm.bias, bert_model.embeddings.LayerNorm.bias)
            torch.testing.assert_close(
                poly_model.embd.tokt.embd.weight, bert_model.embeddings.token_type_embeddings.weight
            )

        assert len(poly_model.encd.blocks) == len(bert_model.encoder.layer)
        for layer_idx in range(len(bert_model.encoder.layer)):
            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_projection"):
                qw, kw, vw = poly_model.encd.blocks[layer_idx].attn.proj.weight.chunk(3, dim=0)
                torch.testing.assert_close(qw, bert_model.encoder.layer[layer_idx].attention.self.query.weight)
                torch.testing.assert_close(kw, bert_model.encoder.layer[layer_idx].attention.self.key.weight)
                torch.testing.assert_close(vw, bert_model.encoder.layer[layer_idx].attention.self.value.weight)

            with subtests.test(f"layer_encoder_block_{layer_idx}_qkv_bias"):
                qb, kb, vb = poly_model.encd.blocks[layer_idx].attn.proj.bias.chunk(3, dim=0)
                torch.testing.assert_close(qb, bert_model.encoder.layer[layer_idx].attention.self.query.bias)
                torch.testing.assert_close(kb, bert_model.encoder.layer[layer_idx].attention.self.key.bias)
                torch.testing.assert_close(vb, bert_model.encoder.layer[layer_idx].attention.self.value.bias)

            with subtests.test(f"layer_encoder_block_{layer_idx}_output_projection"):
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].attn.ffwd.weight,
                    bert_model.encoder.layer[layer_idx].attention.output.dense.weight,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].attn.ffwd.bias,
                    bert_model.encoder.layer[layer_idx].attention.output.dense.bias,
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_ffwd"):
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].ffwd.Uprj.weight,
                    bert_model.encoder.layer[layer_idx].intermediate.dense.weight,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].ffwd.Uprj.bias,
                    bert_model.encoder.layer[layer_idx].intermediate.dense.bias,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].ffwd.Dprj.weight,
                    bert_model.encoder.layer[layer_idx].output.dense.weight,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].ffwd.Dprj.bias,
                    bert_model.encoder.layer[layer_idx].output.dense.bias,
                )

            with subtests.test(f"layer_encoder_block_{layer_idx}_norms"):
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].post_norm_attn.weight,
                    bert_model.encoder.layer[layer_idx].attention.output.LayerNorm.weight,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].post_norm_attn.bias,
                    bert_model.encoder.layer[layer_idx].attention.output.LayerNorm.bias,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].post_norm_ffwd.weight,
                    bert_model.encoder.layer[layer_idx].output.LayerNorm.weight,
                )
                torch.testing.assert_close(
                    poly_model.encd.blocks[layer_idx].post_norm_ffwd.bias,
                    bert_model.encoder.layer[layer_idx].output.LayerNorm.bias,
                )

    @pytest.mark.dependency(depends=["TestFromBertModel::test_weights"])
    def test_embedding_layer(self, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the embedding layer."""
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            emb_bert = bert_model.embeddings(seq["input_ids"])
            emb_poly = poly_model.embd(seq["input_ids"])

        torch.testing.assert_close(emb_bert, emb_poly)

    @pytest.mark.dependency(depends=["TestFromBertModel::test_weights"])
    def test_encoder(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the encoder stack."""
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            # Use original BERT embedding to be able to test polybert embedding independently
            hidden = bert_model.embeddings(seq["input_ids"])
            bert_hidden, poly_hidden = hidden.clone(), hidden.clone()
            # Test the encoder stacks
            bert_hidden_states = bert_model.encoder(
                bert_hidden, seq["attention_mask"].bool(), output_hidden_states=True
            ).hidden_states
            _, poly_hidden_states, _ = poly_model.encd(
                poly_hidden, seq["attention_mask"].bool(), output_hidden_states=True
            )

            assert len(bert_hidden_states) == len(
                poly_hidden_states
            ), f"Unequal number of encoder blocks: {len(poly_hidden_states)} vs. {len(bert_hidden_states)}"
            for layer_idx, (bhs, phs) in enumerate(zip(bert_hidden_states, poly_hidden_states, strict=False)):
                with subtests.test(f"layer_encoder_block_{layer_idx}"):
                    torch.testing.assert_close(bhs, phs, atol=4e-2, rtol=4e-2)

    @pytest.mark.dependency(depends=["TestFromBertModel::test_embedding_layer", "TestFromBertModel::test_encoder"])
    def test_complete_single_sequence(self, tokenizer, bert_model, poly_model):  # type: ignore
        """Test complete model pass for a single given string."""
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length").to(self.device)

        with torch.no_grad():
            seq1_bert = bert_model(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state[:, 0, :]
            seq1_poly = poly_model(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state[:, 0, :]

        # Assert that both produced same output
        torch.testing.assert_close(seq1_bert, seq1_poly)

    @pytest.mark.dependency(depends=["TestFromBertModel::test_complete_single_sequence"])
    def test_complete_multiple_sequences(self, tokenizer, bert_model, poly_model):  # type: ignore
        """Test complete model pass with more than one given string."""
        seq = tokenizer(["I like cats.", "Cats are the ultimate pet."], return_tensors="pt", padding="max_length").to(
            self.device
        )

        with torch.no_grad():
            seq1_bert, seq2_bert = bert_model(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state[
                :, 0, :
            ]
            seq1_poly, seq2_poly = poly_model(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state[
                :, 0, :
            ]

        # Assert that both produced same output
        torch.testing.assert_close(seq1_bert, seq1_poly)
        torch.testing.assert_close(seq2_bert, seq2_poly)
