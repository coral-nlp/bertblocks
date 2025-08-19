import torch

from bertblocks.modeling.config import BertBlocksConfig
from bertblocks.modeling.model import BertBlocksModel


def from_bert_model(
    pretrained_model_name_or_path: str, load_weights: bool = True, add_pooling_layer: bool = False
) -> "BertBlocksModel":
    """Instantiate an equivalent BertBlocks model from BERT weights and config."""
    from transformers import BertConfig, BertModel

    orig_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
    bertblocks_config = BertBlocksConfig(
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
        norm_bias=True,
        emb_dropout_prob=orig_config.hidden_dropout_prob or 0.0,
        attn_dropout_prob=orig_config.attention_probs_dropout_prob or 0.0,
        hidden_dropout_prob=orig_config.hidden_dropout_prob or 0.0,
        classifier_dropout_prob=orig_config.classifier_dropout or 0.0,
    )
    bertblocks_model = BertBlocksModel(bertblocks_config, add_pooling_layer=add_pooling_layer)

    if load_weights:
        orig_model = BertModel.from_pretrained(pretrained_model_name_or_path, add_pooling_layer=add_pooling_layer)

        # Embedding layer
        bertblocks_model.embd.embd.weight.data.copy_(orig_model.embeddings.word_embeddings.weight.data)
        bertblocks_model.embd.pose.embd.weight.data.copy_(orig_model.embeddings.position_embeddings.weight.data)
        bertblocks_model.embd.norm.weight.data.copy_(orig_model.embeddings.LayerNorm.weight.data)
        bertblocks_model.embd.norm.bias.data.copy_(orig_model.embeddings.LayerNorm.bias.data)
        bertblocks_model.embd.tokt.embd.weight.data.copy_(orig_model.embeddings.token_type_embeddings.weight.data)  # type: ignore

        for i in range(len(bertblocks_model.encd.blocks)):
            # QKV Projection
            qkv_weight = torch.cat(
                [
                    orig_model.encoder.layer[i].attention.self.query.weight,
                    orig_model.encoder.layer[i].attention.self.key.weight,
                    orig_model.encoder.layer[i].attention.self.value.weight,
                ],
                dim=0,
            )
            qkv_bias = torch.cat(
                [
                    orig_model.encoder.layer[i].attention.self.query.bias,
                    orig_model.encoder.layer[i].attention.self.key.bias,
                    orig_model.encoder.layer[i].attention.self.value.bias,
                ],
                dim=0,
            )

            bertblocks_model.encd.blocks[i].attn.proj.weight.data.copy_(qkv_weight.data)
            bertblocks_model.encd.blocks[i].attn.proj.bias.data.copy_(qkv_bias.data)

            # Attention output projection
            bertblocks_model.encd.blocks[i].attn.ffwd.weight.data.copy_(
                orig_model.encoder.layer[i].attention.output.dense.weight.data
            )
            bertblocks_model.encd.blocks[i].attn.ffwd.bias.data.copy_(
                orig_model.encoder.layer[i].attention.output.dense.bias.data
            )

            # Feed-forward layers
            bertblocks_model.encd.blocks[i].ffwd.uprj.weight.data.copy_(
                orig_model.encoder.layer[i].intermediate.dense.weight.data
            )
            bertblocks_model.encd.blocks[i].ffwd.uprj.bias.data.copy_(
                orig_model.encoder.layer[i].intermediate.dense.bias.data
            )
            bertblocks_model.encd.blocks[i].ffwd.dprj.weight.data.copy_(
                orig_model.encoder.layer[i].output.dense.weight.data
            )
            bertblocks_model.encd.blocks[i].ffwd.dprj.bias.data.copy_(
                orig_model.encoder.layer[i].output.dense.bias.data
            )

            # Layer norms
            bertblocks_model.encd.blocks[i].post_norm_attn.weight.data.copy_(
                orig_model.encoder.layer[i].attention.output.LayerNorm.weight.data
            )
            bertblocks_model.encd.blocks[i].post_norm_attn.bias.data.copy_(
                orig_model.encoder.layer[i].attention.output.LayerNorm.bias.data
            )
            bertblocks_model.encd.blocks[i].post_norm_ffwd.weight.data.copy_(
                orig_model.encoder.layer[i].output.LayerNorm.weight.data
            )
            bertblocks_model.encd.blocks[i].post_norm_ffwd.bias.data.copy_(
                orig_model.encoder.layer[i].output.LayerNorm.bias.data
            )

    return bertblocks_model
