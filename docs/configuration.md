# Configuration

All architectural choices in BertBlocks are controlled through a single
{class}`~bertblocks.config.BertBlocksConfig` dataclass. This page explains the key
parameters and how to use preset configurations.

## Creating a Configuration

```python
import bertblocks as bb

config = bb.BertBlocksConfig(
    vocab_size=30522,
    hidden_size=768,
    num_blocks=12,
    num_attention_heads=12,
)
```

All parameters have sensible defaults. See the {class}`~bertblocks.config.BertBlocksConfig`
API reference for the full list.

## Key Parameters

### Model dimensions

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | -- |
| `hidden_size` | Hidden dimension | `768` |
| `intermediate_size` | FFN intermediate dimension | `3072` |
| `num_blocks` | Number of transformer layers | `12` |
| `num_attention_heads` | Number of attention heads | `12` |
| `max_position_embeddings` | Maximum sequence length | `512` |

### Architectural choices

| Parameter | Description | Options |
|-----------|-------------|---------|
| `norm_fn` | Normalization function | `"layer"`, `"rms"`, `"group"`, `"deep"`, `"dynamic_tanh"` |
| `norm_pos` | Where to apply normalization | `"pre"`, `"post"`, `"pre_and_post"`, `"none"` |
| `actv_fn` | Activation function | `"silu"`, `"gelu"`, `"relu"`, `"gelu_new"`, ... |
| `mlp_type` | Feed-forward type | `"mlp"`, `"glu"` |
| `embd_pos_enc_kind` | Embedding positional encoding | `"sinusoidal"`, `"learned"`, `"none"` |
| `block_pos_enc_kind` | Block-level positional encoding | `"alibi"`, `"rope"`, `"none"` |
| `attention_backend` | Attention implementation | `"flash"`, `"sdpa"`, `"eager"` |

### Advanced options

| Parameter | Description |
|-----------|-------------|
| `num_kv_heads` | Number of key-value heads for GQA (defaults to `num_attention_heads`) |
| `qk_norm` | Enable query-key normalization |
| `local_attention` | Enable local (sliding window) attention |
| `local_attention_window_size` | Window size for local attention |
| `attention_gate` | Gating mechanism for attention output |

## Preset Configurations

BertBlocks includes preset configurations that reproduce known architectures:

### BertConfig

Reproduces the original BERT architecture:

```python
from bertblocks.config import BertConfig

config = BertConfig(vocab_size=30522)
```

You can also create a `BertConfig` from a HuggingFace model:

```python
config = BertConfig.from_huggingface("bert-base-uncased")
```

### ModernBertConfig

Reproduces the ModernBERT architecture:

```python
from bertblocks.config import ModernBertConfig

config = ModernBertConfig(vocab_size=50368)
```

```python
config = ModernBertConfig.from_huggingface("answerdotai/ModernBERT-base")
```

## Validation

{class}`~bertblocks.config.BertBlocksConfig` validates parameters on construction. For
example, `hidden_size` must be divisible by `num_attention_heads`, and enum-typed parameters
are checked against their allowed values. Invalid configurations raise descriptive errors.

## YAML Configuration

For training via the CLI, configurations are specified in YAML files. See the `configs/`
directory for examples:

```bash
uv run -m bertblocks fit --config configs/pretraining.yaml
```
