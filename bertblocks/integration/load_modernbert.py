from bertblocks.config import ModernBertConfig
from bertblocks.modeling.model import BertBlocksModel


def from_modernbert_model(
    pretrained_model_name_or_path: str, load_weights: bool = True, add_pooling_layer: bool = False
) -> "BertBlocksModel":
    """Instantiate an equivalent BertBlocks model from ModernBERT weights and config."""
    from transformers import ModernBertModel

    bertblocks_config = ModernBertConfig.from_huggingface(pretrained_model_name_or_path)
    bertblocks_model = BertBlocksModel(bertblocks_config, add_pooling_layer=add_pooling_layer)

    if load_weights:
        orig_model = ModernBertModel.from_pretrained(pretrained_model_name_or_path)
        # Embedding layer
        bertblocks_model.embd.embd.weight.data.copy_(orig_model.embeddings.tok_embeddings.weight.data)

        for i in range(len(bertblocks_model.encd.blocks)):
            # QKV Projection
            bertblocks_model.encd.blocks[i].attn.proj.weight.data.copy_(orig_model.layers[i].attn.Wqkv.weight.data)
            if bertblocks_config.attn_proj_bias:
                bertblocks_model.encd.blocks[i].attn.proj.bias.data.copy_(orig_model.layers[i].attn.Wqkv.bias.data)
            # Attention output projection
            bertblocks_model.encd.blocks[i].attn.ffwd.weight.data.copy_(orig_model.layers[i].attn.Wo.weight.data)
            if bertblocks_config.attn_out_bias:
                bertblocks_model.encd.blocks[i].attn.ffwd.bias.data.copy_(orig_model.layers[i].attn.Wo.bias.data)
            # Feed-forward up and down projection
            bertblocks_model.encd.blocks[i].ffwd.uprj.weight.data.copy_(orig_model.layers[i].mlp.Wi.weight.data)
            if bertblocks_config.mlp_in_bias:
                bertblocks_model.encd.blocks[i].ffwd.uprj.bias.data.copy_(orig_model.layers[i].mlp.Wi.bias.data)
            bertblocks_model.encd.blocks[i].ffwd.dprj.weight.data.copy_(orig_model.layers[i].mlp.Wo.weight.data)
            if bertblocks_config.mlp_out_bias:
                bertblocks_model.encd.blocks[i].ffwd.dprj.bias.data.copy_(orig_model.layers[i].mlp.Wo.bias.data)
            # Norms
            if i == 0:
                # If first layer, the norm is in the embedding (pre-norm)
                bertblocks_model.encd.blocks[i].pre_norm_attn.weight.data.copy_(orig_model.embeddings.norm.weight.data)
                if bertblocks_config.norm_bias:
                    bertblocks_model.encd.blocks[i].pre_norm_attn.bias.data.copy_(orig_model.embeddings.norm.bias.data)
            else:
                bertblocks_model.encd.blocks[i].pre_norm_attn.weight.data.copy_(
                    orig_model.layers[i].attn_norm.weight.data
                )
                if bertblocks_config.norm_bias:
                    bertblocks_model.encd.blocks[i].pre_norm_attn.bias.data.copy_(
                        orig_model.layers[i].attn_norm.bias.data
                    )

            bertblocks_model.encd.blocks[i].pre_norm_ffwd.weight.data.copy_(orig_model.layers[i].mlp_norm.weight.data)
            if bertblocks_config.norm_bias:
                bertblocks_model.encd.blocks[i].pre_norm_ffwd.bias.data.copy_(orig_model.layers[i].mlp_norm.bias.data)

            bertblocks_model.norm.weight.data.copy_(orig_model.final_norm.weight.data)
            if bertblocks_config.norm_bias:
                bertblocks_model.norm.bias.data.copy_(orig_model.final_norm.bias.data)

    return bertblocks_model
