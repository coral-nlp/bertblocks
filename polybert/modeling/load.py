import warnings

import torch

from polybert.modeling.config import PolyBertConfig
from polybert.modeling.model import PolyBertModel


def from_bert_model(pretrained_model_name_or_path: str, add_pooling_layer: bool = False) -> "PolyBertModel":
    """Instantiate an equivalent PolyBERT model from BERT weights and config."""
    from transformers import BertConfig, BertModel

    orig_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
    poly_config = PolyBertConfig(
        vocab_size=orig_config.vocab_size,
        max_sequence_length=orig_config.max_position_embeddings,
        pad_token_id=orig_config.pad_token_id,
        hidden_size=orig_config.hidden_size,
        num_blocks=orig_config.num_hidden_layers,
        intermediate_size=orig_config.intermediate_size,
        num_attention_heads=orig_config.num_attention_heads,
        pos_emb_kind="learned" if orig_config.position_embedding_type == "absolute" else "relative",
        add_token_type_emb=True,
        type_vocab_size=orig_config.type_vocab_size,
        mlp_type="mlp",
        mlp_in_bias=True,
        mlp_out_bias=True,
        attn_proj_bias=True,
        attn_out_bias=True,
        initializer_kind="trunc_normal",
        initializer_range=orig_config.initializer_range,
        initializer_cutoff_factor=4.0,
        initializer_gain=1.0,
        actv_fn=orig_config.hidden_act,
        norm_kind="post",
        norm_fn="layer",
        norm_eps=orig_config.layer_norm_eps,
        emb_dropout_prob=orig_config.hidden_dropout_prob or 0.0,
        attn_dropout_prob=orig_config.attention_probs_dropout_prob or 0.0,
        hidden_dropout_prob=orig_config.hidden_dropout_prob or 0.0,
        classifier_dropout_prob=orig_config.classifier_dropout or 0.0,
    )
    poly_model = PolyBertModel(poly_config, add_pooling_layer=add_pooling_layer)
    orig_model = BertModel(orig_config, add_pooling_layer=add_pooling_layer)

    # Embedding layer
    poly_model.embd.embd = orig_model.embeddings.word_embeddings
    poly_model.embd.pose.embd = orig_model.embeddings.position_embeddings
    poly_model.embd.norm.weight = orig_model.embeddings.LayerNorm.weight
    poly_model.embd.norm.bias = orig_model.embeddings.LayerNorm.bias
    poly_model.embd.tokt.embd = orig_model.embeddings.token_type_embeddings

    for i in range(len(poly_model.encd.blocks)):
        # QKV Projection
        poly_model.encd.blocks[i].attn.proj.weight = torch.nn.Parameter(
            torch.cat(
                (
                    orig_model.encoder.layer[i].attention.self.query.weight,
                    orig_model.encoder.layer[i].attention.self.key.weight,
                    orig_model.encoder.layer[i].attention.self.value.weight,
                ),
                0,
            )
        )
        poly_model.encd.blocks[i].attn.proj.bias = torch.nn.Parameter(
            torch.cat(
                (
                    orig_model.encoder.layer[i].attention.self.query.bias,
                    orig_model.encoder.layer[i].attention.self.key.bias,
                    orig_model.encoder.layer[i].attention.self.value.bias,
                ),
                0,
            )
        )
        # Attention output projection
        poly_model.encd.blocks[i].attn.ffwd.weight = orig_model.encoder.layer[i].attention.output.dense.weight
        poly_model.encd.blocks[i].attn.ffwd.bias = orig_model.encoder.layer[i].attention.output.dense.bias
        # Feed-forward up and down projection
        poly_model.encd.blocks[i].ffwd.Uprj.weight = orig_model.encoder.layer[i].intermediate.dense.weight
        poly_model.encd.blocks[i].ffwd.Uprj.bias = orig_model.encoder.layer[i].intermediate.dense.bias
        poly_model.encd.blocks[i].ffwd.Dprj.weight = orig_model.encoder.layer[i].output.dense.weight
        poly_model.encd.blocks[i].ffwd.Dprj.bias = orig_model.encoder.layer[i].output.dense.bias
        # Norms
        poly_model.encd.blocks[i].post_norm_attn.weight = orig_model.encoder.layer[i].attention.output.LayerNorm.weight
        poly_model.encd.blocks[i].post_norm_attn.bias = orig_model.encoder.layer[i].attention.output.LayerNorm.bias
        poly_model.encd.blocks[i].post_norm_ffwd.weight = orig_model.encoder.layer[i].output.LayerNorm.weight
        poly_model.encd.blocks[i].post_norm_ffwd.bias = orig_model.encoder.layer[i].output.LayerNorm.bias

    return poly_model


def from_modernbert_model(pretrained_model_name_or_path: str, add_pooling_layer: bool = False) -> "PolyBertModel":
    """Instantiate an equivalent PolyBERT model from ModernBERT weights and config."""
    from transformers import ModernBertConfig, ModernBertModel

    orig_config = ModernBertConfig.from_pretrained(pretrained_model_name_or_path)
    if orig_config.global_attn_every_n_layers != 1:
        warnings.warn(
            "Local attention is currently unsupported, falling back to global attention at every layer.", stacklevel=2
        )
        orig_config.global_attn_every_n_layers = 1
    poly_config = PolyBertConfig(
        vocab_size=orig_config.vocab_size,
        max_sequence_length=orig_config.max_position_embeddings,
        pad_token_id=orig_config.pad_token_id,
        hidden_size=orig_config.hidden_size,
        num_blocks=orig_config.num_hidden_layers,
        intermediate_size=orig_config.intermediate_size,
        num_attention_heads=orig_config.num_attention_heads,
        pos_emb_kind="rope",
        pos_emb_kwargs={"base": orig_config.global_rope_theta},
        add_token_type_emb=False,
        mlp_type="glu",
        mlp_in_bias=orig_config.mlp_bias,
        mlp_out_bias=orig_config.mlp_bias,
        attn_proj_bias=orig_config.attention_bias,
        attn_out_bias=orig_config.attention_bias,
        initializer_kind="trunc_normal",
        initializer_range=orig_config.initializer_range,
        initializer_cutoff_factor=4.0,
        initializer_gain=1.0,
        actv_fn=orig_config.hidden_activation or "gelu",
        norm_kind="pre",
        norm_fn="layer",
        norm_eps=orig_config.layer_norm_eps,
        emb_dropout_prob=orig_config.embedding_dropout or 0.0,
        attn_dropout_prob=orig_config.attention_dropout or 0.0,
        hidden_dropout_prob=orig_config.mlp_dropout or 0.0,
        classifier_dropout_prob=orig_config.classifier_dropout or 0.0,
    )

    orig_model = ModernBertModel(orig_config)
    poly_model = PolyBertModel(poly_config)

    # Embedding layer
    poly_model.embd.embd = orig_model.embeddings.tok_embeddings
    poly_model.embd.norm.weight = orig_model.embeddings.norm.weight
    poly_model.embd.norm.bias = orig_model.embeddings.norm.bias

    for i in range(len(poly_model.encd.blocks)):
        # QKV Projection
        poly_model.encd.blocks[i].attn.proj.weight = orig_model.layers[i].attn.Wqkv.weight
        if poly_config.attn_proj_bias:
            poly_model.encd.blocks[i].attn.proj.bias = orig_model.layers[i].attn.Wqkv.bias
        # Attention output projection
        poly_model.encd.blocks[i].attn.ffwd.weight = orig_model.layers[i].attn.Wo.weight
        if poly_config.attn_out_bias:
            poly_model.encd.blocks[i].attn.ffwd.bias = orig_model.layers[i].attn.Wo.bias
        # Feed-forward up and down projection
        poly_model.encd.blocks[i].ffwd.Uprj.weight = orig_model.layers[i].mlp.Wi.weight
        if poly_config.mlp_in_bias:
            poly_model.encd.blocks[i].ffwd.Uprj.bias = orig_model.layers[i].mlp.Wi.bias
        poly_model.encd.blocks[i].ffwd.Dprj.weight = orig_model.layers[i].mlp.Wo.weight
        if poly_config.mlp_out_bias:
            poly_model.encd.blocks[i].ffwd.Dprj.bias = orig_model.layers[i].mlp.Wo.bias
        # Norms
        if i == 0:
            # If first layer, the norm is in the embedding (pre-norm)
            poly_model.encd.blocks[i].pre_norm_attn.weight = orig_model.embeddings.norm.weight
            poly_model.encd.blocks[i].pre_norm_attn.bias = orig_model.embeddings.norm.bias
        else:
            poly_model.encd.blocks[i].pre_norm_attn.weight = orig_model.layers[i].attn_norm.weight
            poly_model.encd.blocks[i].pre_norm_attn.bias = orig_model.layers[i].attn_norm.bias

        poly_model.encd.blocks[i].pre_norm_ffwd.weight = orig_model.layers[i].mlp_norm.weight
        poly_model.encd.blocks[i].pre_norm_ffwd.bias = orig_model.layers[i].mlp_norm.bias

    return poly_model
