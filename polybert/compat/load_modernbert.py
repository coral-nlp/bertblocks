from polybert.modeling.config import PolyBertConfig
from polybert.modeling.model import PolyBertModel


def from_modernbert_model(pretrained_model_name_or_path: str, add_pooling_layer: bool = False) -> "PolyBertModel":
    """Instantiate an equivalent PolyBERT model from ModernBERT weights and config."""
    from transformers import ModernBertConfig, ModernBertModel

    orig_config = ModernBertConfig.from_pretrained(pretrained_model_name_or_path)

    poly_config = PolyBertConfig(
        vocab_size=orig_config.vocab_size,
        max_sequence_length=orig_config.max_position_embeddings,
        pad_token_id=orig_config.pad_token_id,
        hidden_size=orig_config.hidden_size,
        num_blocks=orig_config.num_hidden_layers,
        intermediate_size=orig_config.intermediate_size,
        num_attention_heads=orig_config.num_attention_heads,
        pos_emb_kind="rope",
        pos_emb_kwargs={"base": orig_config.global_rope_theta, "scale": 1},
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
        norm_bias=orig_config.norm_bias,
        emb_dropout_prob=orig_config.embedding_dropout or 0.0,
        attn_dropout_prob=orig_config.attention_dropout or 0.0,
        hidden_dropout_prob=orig_config.mlp_dropout or 0.0,
        classifier_dropout_prob=orig_config.classifier_dropout or 0.0,
    )

    orig_model = ModernBertModel.from_pretrained(pretrained_model_name_or_path)
    poly_model = PolyBertModel(poly_config, add_pooling_layer=add_pooling_layer)

    # Embedding layer
    poly_model.embd.embd.weight.data.copy_(orig_model.embeddings.tok_embeddings.weight.data)

    for i in range(len(poly_model.encd.blocks)):
        # QKV Projection
        poly_model.encd.blocks[i].attn.proj.weight.data.copy_(orig_model.layers[i].attn.Wqkv.weight.data)
        if poly_config.attn_proj_bias:
            poly_model.encd.blocks[i].attn.proj.bias.data.copy_(orig_model.layers[i].attn.Wqkv.bias.data)
        # Attention output projection
        poly_model.encd.blocks[i].attn.ffwd.weight.data.copy_(orig_model.layers[i].attn.Wo.weight.data)
        if poly_config.attn_out_bias:
            poly_model.encd.blocks[i].attn.ffwd.bias.data.copy_(orig_model.layers[i].attn.Wo.bias.data)
        # Feed-forward up and down projection
        poly_model.encd.blocks[i].ffwd.Uprj.weight.data.copy_(orig_model.layers[i].mlp.Wi.weight.data)
        if poly_config.mlp_in_bias:
            poly_model.encd.blocks[i].ffwd.Uprj.bias.data.copy_(orig_model.layers[i].mlp.Wi.bias.data)
        poly_model.encd.blocks[i].ffwd.Dprj.weight.data.copy_(orig_model.layers[i].mlp.Wo.weight.data)
        if poly_config.mlp_out_bias:
            poly_model.encd.blocks[i].ffwd.Dprj.bias.data.copy_(orig_model.layers[i].mlp.Wo.bias.data)
        # Norms
        if i == 0:
            # If first layer, the norm is in the embedding (pre-norm)
            poly_model.encd.blocks[i].pre_norm_attn.weight.data.copy_(orig_model.embeddings.norm.weight.data)
            if poly_config.norm_bias:
                poly_model.encd.blocks[i].pre_norm_attn.bias.data.copy_(orig_model.embeddings.norm.bias.data)
        else:
            poly_model.encd.blocks[i].pre_norm_attn.weight.data.copy_(orig_model.layers[i].attn_norm.weight.data)
            if poly_config.norm_bias:
                poly_model.encd.blocks[i].pre_norm_attn.bias.data.copy_(orig_model.layers[i].attn_norm.bias.data)

        poly_model.encd.blocks[i].pre_norm_ffwd.weight.data.copy_(orig_model.layers[i].mlp_norm.weight.data)
        if poly_config.norm_bias:
            poly_model.encd.blocks[i].pre_norm_ffwd.bias.data.copy_(orig_model.layers[i].mlp_norm.bias.data)

    return poly_model
