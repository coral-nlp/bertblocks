import torch

from polybert.modeling.config import PolyBertConfig
from polybert.modeling.model import PolyBertModel


def from_bert_model(pretrained_model_name_or_path: str) -> "PolyBertModel":
    """Instantiate an equivalent PolyBERT model from BERT weights and config."""
    from transformers import BertConfig, BertModel

    def _bert_config_to_polybert_config(pretrained_model_name_or_path: str) -> "PolyBertConfig":
        """Construct an equivalent PolyBERT config from a given BERT config."""
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        return PolyBertConfig(
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            hidden_size=config.hidden_size,
            num_blocks=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            pos_emb_kind="learned" if config.position_embedding_type == "absolute" else "relative",
            add_token_type_emb=True,
            type_vocab_size=config.type_vocab_size,
            mlp_type="mlp",
            mlp_in_bias=True,
            mlp_out_bias=True,
            attn_proj_bias=True,
            attn_out_bias=True,
            initializer_kind="trunc_normal",
            initializer_range=config.initializer_range,
            initializer_cutoff_factor=4.0,
            initializer_gain=1.0,
            actv_fn=config.hidden_act,
            norm_kind="post",
            norm_fn="layer",
            norm_eps=config.layer_norm_eps,
            emb_dropout_prob=config.hidden_dropout_prob or 0.0,
            attn_dropout_prob=config.attention_probs_dropout_prob or 0.0,
            hidden_dropout_prob=config.hidden_dropout_prob or 0.0,
            classifier_dropout_prob=config.classifier_dropout or 0.0,
        )

    config = _bert_config_to_polybert_config(pretrained_model_name_or_path)
    model = PolyBertModel(config, add_pooling_layer=False)
    bert_model = BertModel.from_pretrained(pretrained_model_name_or_path)

    # Embedding layer
    model.embd.embd = bert_model.embeddings.word_embeddings
    model.embd.pose.embd = bert_model.embeddings.position_embeddings
    model.embd.norm.weight = bert_model.embeddings.LayerNorm.weight
    model.embd.norm.bias = bert_model.embeddings.LayerNorm.bias
    model.embd.tokt.embd = bert_model.embeddings.token_type_embeddings

    for i in range(len(model.encd.blocks)):
        # QKV Projection
        model.encd.blocks[i].attn.proj.weight = torch.nn.Parameter(
            torch.cat(
                (
                    bert_model.encoder.layer[i].attention.self.query.weight,
                    bert_model.encoder.layer[i].attention.self.key.weight,
                    bert_model.encoder.layer[i].attention.self.value.weight,
                ),
                0,
            )
        )
        model.encd.blocks[i].attn.proj.bias = torch.nn.Parameter(
            torch.cat(
                (
                    bert_model.encoder.layer[i].attention.self.query.bias,
                    bert_model.encoder.layer[i].attention.self.key.bias,
                    bert_model.encoder.layer[i].attention.self.value.bias,
                ),
                0,
            )
        )
        # Attention output projection
        model.encd.blocks[i].attn.ffwd.weight = bert_model.encoder.layer[i].attention.output.dense.weight
        model.encd.blocks[i].attn.ffwd.bias = bert_model.encoder.layer[i].attention.output.dense.bias
        # Feed-forward up and down projection
        model.encd.blocks[i].ffwd.Uprj.weight = bert_model.encoder.layer[i].intermediate.dense.weight
        model.encd.blocks[i].ffwd.Uprj.bias = bert_model.encoder.layer[i].intermediate.dense.bias
        model.encd.blocks[i].ffwd.Dprj.weight = bert_model.encoder.layer[i].output.dense.weight
        model.encd.blocks[i].ffwd.Dprj.bias = bert_model.encoder.layer[i].output.dense.bias
        # Norms
        model.encd.blocks[i].post_norm_attn.weight = bert_model.encoder.layer[i].attention.output.LayerNorm.weight
        model.encd.blocks[i].post_norm_attn.bias = bert_model.encoder.layer[i].attention.output.LayerNorm.bias
        model.encd.blocks[i].post_norm_ffwd.weight = bert_model.encoder.layer[i].output.LayerNorm.weight
        model.encd.blocks[i].post_norm_ffwd.bias = bert_model.encoder.layer[i].output.LayerNorm.bias

    return model


if __name__ == "__main__":
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
