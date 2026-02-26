<div align="center">
  <img alt="BertBlocks" src="./docs/_static/bertblocks.svg" width="75%" style="max-width: 75%;">
  <br/>
</div>

> ⚠️ **BertBlocks is currently in alpha!** APIs and features may change. We appreciate feedback and contributions as we work towards a stable release.

## Overview

**BertBlocks** provides building blocks for exploring transformer encoders. It aims to be a unified, clean, well-documented, and comprehensive collection of components for BERT-like models.
It is highly configurable and allows for easy experimentation with various architectural components including:

- **Normalization**: Pre/post normalization, RMS Norm, Layer Norm, Group Norm, DeepNorm, DynamicTanhNorm, ...
- **Attention Mechanisms**: Multi-head attention with configurable heads and dropout
- **Positional Encodings**: ALiBi, Sinusoidal, RoPE, Relative, Learned, ...
- **Feed-Forward Networks**: Standard MLP, Gated Linear Units (GLU)...
- **Activation Functions**: SiLU, GELU, ReLU, ...
- **Optimization**: Pre-configured training setup with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/), variety of optimizers, training objectives, ...
- **Attention Backends**: supports flash-, sdpa-, and eager-attention implementations for maximum flexibility, for both padded and unpadded sequences

## Quick Start

### Basic Usage

Train a model with the default configuration:

```bash
uv run -m bertblocks fit --config configs/pretraining.yaml
```

### Configuration

The architecture is configurable through the `BertBlocksConfig` class. Key parameters include:

```python
import bertblocks as bb

config = bb.BertBlocksConfig(
    vocab_size=30522,            # Vocabulary size
    hidden_size=768,             # Model dimension
    num_blocks=12,               # Number of transformer layers
    num_attention_heads=12,      # Number of attention heads
    norm_fn="rms",               # Normalization type
    block_pos_enc_kind="alibi",  # Positional encoding
    mlp_type="glu",              # Feed-forward architecture
    actv_fn="silu"               # Activation function
)

model = bb.BertBlocksForMaskedLM(config)
```

Alternatively, select Huggingface encoder architectures can be reproduced, optionally also loading their weights:

```python
import bertblocks as bb

# Returns an equivalent BertBlocks model
model = bb.from_huggingface("answerdotai/ModernBERT-base", load_weights=True)
```

We are actively working on adding more verified model loaders.

If you want to add one, or make general improvements to `bertblocks`, have a look at our  [contribution guide](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bertblocks,
  title  = {BertBlocks - Building Blocks for Exploring Transformer Encoders},
  author = {CORAL Project Contributors},
  year   = {2025},
  url    = {https://github.com/coral-nlp/bertblocks}
 }
```

---
<div align="center">
  <img alt="BertBlocks" src="./docs/_static/blocks.svg" width="35px" style="max-width: 35px;">
</div>
