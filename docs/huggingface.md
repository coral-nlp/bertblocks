# HuggingFace Integration

BertBlocks models are compatible with the HuggingFace `transformers` ecosystem. This page
explains how to load existing HuggingFace models into BertBlocks and how BertBlocks models
register with HuggingFace's `Auto` classes.

## Loading HuggingFace Models

Use {func}`~bertblocks.integration.from_huggingface` to convert a supported HuggingFace
model into a BertBlocks model:

```python
import bertblocks as bb

# Load architecture and weights
model = bb.from_huggingface("answerdotai/ModernBERT-base", load_weights=True)

# Load architecture only (random weights)
model = bb.from_huggingface("answerdotai/ModernBERT-base", load_weights=False)
```

### Supported architectures

| HuggingFace model type | Loader |
|------------------------|--------|
| ModernBERT | {func}`~bertblocks.integration.load_modernbert.from_modernbert_model` |
| BERT | {func}`~bertblocks.integration.load_bert.from_bert_model` |

The loader automatically maps the HuggingFace configuration to a
{class}`~bertblocks.config.BertBlocksConfig` and (optionally) transfers weights.

## AutoModel Registration

When you import `bertblocks`, the package registers its model classes with HuggingFace's
`Auto` classes. This means you can use standard HuggingFace APIs:

```python
from transformers import AutoModel, AutoConfig

# These work after `import bertblocks`
config = AutoConfig.from_pretrained("path/to/bertblocks-model")
model = AutoModel.from_pretrained("path/to/bertblocks-model")
```

The following task-specific auto classes are also registered:

- `AutoModelForMaskedLM`
- `AutoModelForSequenceClassification`
- `AutoModelForTokenClassification`
- `AutoModelForQuestionAnswering`

## Saving and Sharing

Since BertBlocks models inherit from HuggingFace's `PreTrainedModel`, you can save and
upload them as usual:

```python
model.save_pretrained("my-model")
model.push_to_hub("username/my-model")
```
