# Contributing to `bertblocks`

This guide covers everything you need to know to get started contributing to `bertblocks`.

## Development Setup

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/coral-nlp/bertblocks/ bertblocks
   cd bertblocks
   ```

2. Set up the dev environment:
   ```bash
   make dev-setup
   ```

   **Note for developers without CUDA-enabled GPUs**: If you don't have a CUDA-enabled graphics card locally, set the following environment variable for all `uv` commands:
   ```bash
   export FLASH_ATTENTION_SKIP_CUDA_BUILD=true
   make dev-setup
   ```


## Commit Workflow

`bertblocks` follows a three-branch workflow:

```
feature-branch → dev → main
```

1. **Feature branches**: Create feature branches from `dev` for new work; ideally, each feature branch is linked to an issue to keep track of whats being worked on
2. **Dev branch**: All feature branches merge to `dev` first; assign a maintainer for review for your merge request
3. **Main branch**: Stable releases are merged from `dev` to `main` periodically by maintainers

### Requirements before merging to dev:

Before merging to dev, make sure the following completes without issues:
```bash
make dev-check
```

## Documentation Standards

All code should be documented well, and make `bertblocks` accessible to non-experts; for merge requests, the docs must build successfully and look good.


### Docstrings
Use Google-style docstrings with custom shape annotations:

```python
def some_function(x: torch.Tensor) -> torch.Tensor:
    """Brief description of the function.

    References:
        - Smith et al. (2000): Some title (<some_arxiv_url>)

    Args:
        x (torch.Tensor, shape [total_seq_len, hidden_size] or [batch_size, seq_len, hidden_size]): Hidden state
            to add token type ids to.

    Returns:
        torch.Tensor: Description of return value.
    """
```

### Building Documentation
Ensure documentation builds successfully before requesting a dev merge:
```bash
make docs-build
```

You can view the resulting doc build in your local browser by running

```bash
make doc-serve # Serves docs at localhost:8000
```

## Repository Structure

The repository is organized into four main subpackages:

- **`bertblocks/modeling/`**: Core model components (attention, MLP, embeddings, etc.)
- **`bertblocks/pretraining/`**: Training infrastructure (objectives, optimizer, scheduler, utilities)
- **`bertblocks/config/`**: Configuration management and schemas
- **`bertblocks/integration/`**: Integration with external frameworks (HuggingFace, etc.)

## Testing

### Integration Tests
Primary testing is done through integration tests that compare against HuggingFace reference implementations. These ensure compatibility and correctness.

### Unit Tests
Unit tests can be added as needed. Place `test_*.py` files next to the source code they test (no separate test directory).

Example structure:
```
bertblocks/modeling/
├── attention.py
├── test_attention.py
├── mlp.py
└── test_mlp.py
```

## Code Style Guidelines

**Write simple, readable code**

### Naming Conventions
- **Layer names should be self-explanatory**
- **Try to keep layer names to 4 letters** to maintain clean, readable code structure

Examples:
```python
# Good
attn = MultiHeadAttention(...)
norm = LayerNorm(...)
proj = Linear(...)

# Less preferred
attention_layer = MultiHeadAttention(...)
layer_normalization = LayerNorm(...)
projection_layer = Linear(...)
```

### Type Hinting Standards

Follow modern Python type hinting practices:

- **Use string annotations** for forward references and complex types
- **Put typing-only imports behind `TYPE_CHECKING`** to avoid runtime imports
- **Use native types** (`list`, `dict`, `tuple`) instead of `typing.List`, `typing.Dict`, `typing.Tuple`
- **Use union operators** (`|`) instead of `typing.Union`
- **Use `| None`** instead of `typing.Optional`

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel

def process_tokens(
    tokens: "list[str]",
    model: "PreTrainedModel | None" = None,
    config: "dict[str, int | float]" = None
) -> "tuple[list[int], dict[str, float]]":
    """Process tokens with optional model and configuration.

    Args:
        tokens: List of token strings to process.
        model: Optional pretrained model for processing.
        config: Optional configuration dictionary.

    Returns:
        Tuple of processed token IDs and metrics.
    """
```

### Code Examples

**Using match statements:**
```python
# Good
match activation_type:
    case "relu":
        return F.relu(x)
    case "gelu":
        return F.gelu(x)
    case "swish":
        return F.silu(x)
    case _:
        raise ValueError(f"Unknown activation: {activation_type}")

# Less preferred
if activation_type == "relu":
    return F.relu(x)
elif activation_type == "gelu":
    return F.gelu(x)
elif activation_type == "swish":
    return F.silu(x)
else:
    raise ValueError(f"Unknown activation: {activation_type}")
```

## Getting Help

If you have questions or run into issues:
1. Check existing issues and discussions
2. Look at the documentation
3. Feel free to open a new issue for discussion

Happy contributing!
