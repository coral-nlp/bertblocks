import torch

from polybert.modeling.load import from_bert_model


class TestFromBertModel:
    """Test that huggingface BERT and loaded polybert BERT implementations return same output."""

    def test_from_bert_model_single_sequence(self) -> None:
        """Test for a single given string."""
        from transformers import AutoModel, AutoTokenizer

        poly_model = from_bert_model("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        poly_model.eval()
        bert_model.eval()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        seq = tokenizer("I like cats.", return_tensors="pt", padding="max_length")

        with torch.no_grad():
            seq1_bert = bert_model(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state[:, 0, :]
            seq1_poly = poly_model(seq["input_ids"], attention_mask=seq["attention_mask"]).last_hidden_state[:, 0, :]

        # Assert that both produced same output
        torch.testing.assert_close(seq1_bert, seq1_poly)

    def test_from_bert_model_multiple_sequences(self) -> None:
        """Test with more than one given string."""
        from transformers import AutoModel, AutoTokenizer

        poly_model = from_bert_model("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        poly_model.eval()
        bert_model.eval()

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        seq = tokenizer(["I like cats.", "Cats are the ultimate pet."], return_tensors="pt", padding="max_length")

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
