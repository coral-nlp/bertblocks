# Getting Started

## Installation

Install BertBlocks with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

For development dependencies (linting, testing, docs):

```bash
uv pip install -e ".[dev]"
```

Optional extras:

- **Flash Attention**: `uv pip install -e ".[flash-attn]"` for FlashAttention-2 support
- **Optimizers**: `uv pip install -e ".[optimizers]"` for BitsAndBytes quantized optimizers
- **W&B logging**: `uv pip install -e ".[wandb]"` for Weights & Biases integration

## Quick Start

### Create a model from a configuration

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

### Load a HuggingFace model

BertBlocks can reproduce select HuggingFace encoder architectures and optionally load their weights:

```python
import bertblocks as bb

model = bb.from_huggingface("answerdotai/ModernBERT-base", load_weights=True)
```

See the [HuggingFace Integration](huggingface.md) guide for details.

### Train a model

Train with the default configuration using the CLI:

```bash
uv run -m bertblocks fit --config configs/pretraining.yaml
```

BertBlocks uses [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training.
The {class}`~bertblocks.training.modules.BertBlocksPretrainingModule` wraps the model with
optimizer and scheduler configuration, while
{class}`~bertblocks.training.modules.BertBlocksPretrainingDataModule` handles data loading
with streaming and dynamic masking.

:::{note}
See the [Configuration Guide](configuration.md) for detailed information about all available training options.
:::

:::{tip}
For distributed training across multiple GPUs, set the appropriate `devices` and `num_nodes` in your training config.
:::

:::{warning}
FlashAttention-2 requires a CUDA-capable GPU. For CPU training or unsupported GPUs, install without the flash-attn extra.
:::

## What's Next

- [Architecture Overview](architecture.md) -- understand how components compose
- [Configuration Guide](configuration.md) -- explore all configuration options
- [API Reference](api/config.md) -- full class and function documentation
