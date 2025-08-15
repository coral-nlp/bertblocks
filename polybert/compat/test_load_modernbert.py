import pytest
import torch
from transformers import AutoTokenizer, ModernBertModel

from polybert.compat.load_modernbert import from_modernbert_model


class TestFromModernBertModel:
    """Test equivalency of Huggingface ModernBERT and loaded polybert BERT implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = "answerdotai/ModernBERT-base"

    @pytest.fixture(scope="class")
    def bert_model(self):  # type: ignore
        """Instantiate Huggingface BERT model as fixture."""
        bert_model = ModernBertModel.from_pretrained(self.baseline_model)
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
                    poly_model.encd.blocks[layer_idx].ffwd.Dprj.weight, bert_model.layers[layer_idx].mlp.Wo.weight.data
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
                    bert_model.layers[layer_idx].mlp_norm.weight.data,
                )

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_weights"])
    def test_embedding_layer(self, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the embedding layer."""
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad():
            emb_bert = bert_model.embeddings(seq["input_ids"])
            emb_poly = poly_model.embd(seq["input_ids"])
            # The norm is not part of the embedding module in polybert, so we have to manually apply it
            emb_poly = poly_model.encd.blocks[0].pre_norm_attn(emb_poly)

        torch.testing.assert_close(emb_bert, emb_poly)

    @pytest.mark.dependency(depends=["TestFromModernBertModel::test_embedding_layer"])
    def test_encoder(self, subtests, tokenizer, bert_model, poly_model):  # type: ignore
        """Test the encoder stack."""
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length").to(self.device)

        with torch.no_grad():
            hidden_bert = bert_model(
                seq["input_ids"], attention_mask=seq["attention_mask"], output_hidden_states=True
            ).hidden_states
            hidden_poly = poly_model(
                seq["input_ids"], attention_mask=seq["attention_mask"], output_hidden_states=True
            ).hidden_states

        for layer_idx, (bhs, phs) in enumerate(zip(hidden_bert, hidden_poly, strict=False)):
            with subtests.test(f"layer_encoder_block_{layer_idx}"):
                torch.testing.assert_close(bhs, phs)
