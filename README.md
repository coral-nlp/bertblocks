<div align="center">

<img alt="BertBlocks" src="./assets/bertblocks.png" width="800px" style="max-width: 100%;">

<br/>
<br/>
</div>

## Overview

This project implements **BertBlocks**, a highly configurable transformer encoder codebase that allows experimentation with various architectural components including:

- **Normalization**: RMS Norm, Layer Norm, Group Norm, DeepNorm, DynamicTanhNorm, ...
- **Attention Mechanisms**: Multi-head attention with configurable heads and dropout
- **Positional Encodings**: ALiBi, Sinusoidal, RoPE, Relative, Learned, ...
- **Feed-Forward Networks**: Standard MLP and Gated Linear Units (GLU)
- **Activation Functions**: SiLU, GELU, ReLU, ...
- **Optimization**: Pre/post normalization, dropout configurations, ...

## Quick Start

### Basic Usage

Train a model with the default configuration:

```bash
uv run main.py fit --config configs/pretraining.yaml
```

### Configuration

The architecture is highly configurable through the `BertBlocksConfig` class. Key parameters include:

```python
import bertblocks as bb

config = bb.BertBlocksConfig(
    vocab_size=30522,  # Vocabulary size
    hidden_size=768,  # Model dimension
    num_blocks=12,  # Number of transformer layers
    num_attention_heads=12,  # Number of attention heads
    norm_fn="rms",  # Normalization type
    pos_emb_kind="alibi",  # Positional encoding
    mlp_type="glu",  # Feed-forward architecture
    actv_fn="silu"  # Activation function
)

model = bb.BertBlocksForMaskedLM(config)
```

Configuration is typically supplied via lightnings [YAML-based configuration](configs/pretraining.yaml) options, where a dictionary of model config options can be passed to the LightningModule.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bertblocks,
  title={BertBlocks - a comprehensive framework for exploring transformer encoders},
  author={CORAL Project Contributors},
  year={2025},
  url={https://github.com/your-repo/encoder-architecture-search}
}
```
