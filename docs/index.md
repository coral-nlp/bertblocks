# BertBlocks

**Building blocks for exploring transformer encoders.**

BertBlocks is a unified and comprehensive collection of components for BERT-like models.
It is highly configurable and designed for easy experimentation with architectural choices.

Unlike HuggingFace's approach where each model variant (BERT, RoBERTa, ELECTRA, etc.) has its own implementation, BertBlocks uses a **single flexible architecture** controlled entirely through configuration.
This means you can swap normalization types, attention mechanisms, positional encodings, and other components without modifying code: just update your config.
This is ideal for research, ablation studies, and architectural exploration.

## Features

BertBlocks provides flexible implementations for:

- **Normalization**: Pre/post normalization, RMS Norm, Layer Norm, Group Norm, DeepNorm, DynamicTanhNorm
- **Attention Mechanisms**: Multi-head attention with configurable heads, Grouped Query Attention (GQA), Multi-Query Attention (MQA)
- **Positional Encodings**: ALiBi, Sinusoidal, RoPE, Learned, Learned ALiBi
- **Feed-Forward Networks**: Standard MLP, Gated Linear Units (GLU)
- **Activation Functions**: SiLU, GELU, ReLU, and more
- **Attention Backends**: Flash Attention, SDPA, and eager implementations with support for both padded and unpadded sequences
- **Training**: Integrated PyTorch Lightning support with pre-configured optimizers and training objectives
- **Model Loading**: Load and convert models from HuggingFace
- **Efficiency**: BertBlocks offers full support for Flash Attention, sequence packing, and flattened batches for maximum training throughput, both for single-GPU and distributed training

## Quick Start

### Create and train a model

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

### Load from HuggingFace

You can also load and convert models from HuggingFace:

```python
import bertblocks as bb

# Load a ModernBERT model as BertBlocks
model = bb.from_huggingface("answerdotai/ModernBERT-base", load_weights=True)
```

## Next Steps

- **[Getting Started](getting_started.md)**: Learn the basics and run your first model
- **[Architecture Guide](architecture.md)**: Understand the design and components
- **[Configuration Reference](configuration.md)**: Explore all available configuration options
- **[HuggingFace Integration](huggingface.md)**: Load and convert models from HuggingFace
- **[API Reference](api/config.md)**: Full API documentation

```{toctree}
:hidden:
:maxdepth: 1
:caption: Getting Started

getting_started
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Conceptual Guides

architecture
configuration
huggingface
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API Reference

api/config
api/modeling
api/training
api/integration
api/benchmarks
```
