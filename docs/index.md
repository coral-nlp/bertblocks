# BertBlocks

**Building blocks for exploring transformer encoders.**

BertBlocks is a unified, clean, and comprehensive collection of components for BERT-like models.
It is highly configurable and designed for easy experimentation with architectural choices
including normalization strategies, attention mechanisms, positional encodings, feed-forward
networks, and more.

```python
import bertblocks as bb

config = bb.BertBlocksConfig(
    vocab_size=30522,
    hidden_size=768,
    num_blocks=12,
    num_attention_heads=12,
    norm_fn="rms",
    block_pos_enc_kind="alibi",
    mlp_type="glu",
    actv_fn="silu",
)
model = bb.BertBlocksForMaskedLM(config)
```

```{toctree}
:maxdepth: 1
:caption: Getting Started

getting_started
```

```{toctree}
:maxdepth: 1
:caption: Conceptual Guides

architecture
configuration
huggingface
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/config
api/modeling
api/training
api/integration
api/benchmarks
```
