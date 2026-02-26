# Modeling

The `bertblocks.modeling` package contains all neural network components.

## Models

The top-level model classes that combine embedding, encoder, and task head.

```{eval-rst}
.. autoclass:: bertblocks.modeling.model.BertBlocksPreTrainedModel
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.model.BertBlocksModel
   :members:
   :show-inheritance:
```

### Task Heads

```{eval-rst}
.. autoclass:: bertblocks.modeling.model.BertBlocksForMaskedLM
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.model.BertBlocksForSequenceClassification
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.model.BertBlocksForTokenClassification
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.model.BertBlocksForQuestionAnswering
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.model.BertBlocksForMaskedDiffusion
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.model.BertBlocksForEnhancedMaskedLM
   :members:
   :show-inheritance:
```

### Output Types

```{eval-rst}
.. autoclass:: bertblocks.modeling.model.MaybeUnpaddedBaseModelOutput
   :members:

.. autoclass:: bertblocks.modeling.model.MaybeUnpaddedBaseModelOutputWithPooling
   :members:
```

## Transformer Block

```{eval-rst}
.. autoclass:: bertblocks.modeling.block.Block
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.block.EnhancedMaskingBlock
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.block.Encoder
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.modeling.block.convert_to_4d_attention_mask
```

## Attention

```{eval-rst}
.. autoclass:: bertblocks.modeling.attention.Attention
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.attention.AttentionGate
   :members:
   :show-inheritance:
```

### Attention Backends

```{eval-rst}
.. autoclass:: bertblocks.modeling.backends.AttentionBackend
   :members:

.. autoclass:: bertblocks.modeling.backends.FlashBackend
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.backends.SDPABackend
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.backends.EagerBackend
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.modeling.backends.get_attention
```

## Embeddings

```{eval-rst}
.. autoclass:: bertblocks.modeling.embedding.TokenEmbedding
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.embedding.TokenTypeEmbedding
   :members:
   :show-inheritance:
```

## Positional Encodings

```{eval-rst}
.. autoclass:: bertblocks.modeling.position.SinusoidalPositionalEncoding
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.position.LearnedPositionalEncoding
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.position.AlibiPositionalEncoding
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.position.RotaryPositionalEncoding
   :members:
   :show-inheritance:
```

## Feed-Forward Networks

```{eval-rst}
.. autoclass:: bertblocks.modeling.mlp.MLP
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.mlp.GLU
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.mlp.Linear
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.modeling.mlp.get_mlp
```

## Normalization

```{eval-rst}
.. autoclass:: bertblocks.modeling.norms.DynamicTanhNorm
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.norms.DeepNorm
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.modeling.norms.get_norm
```

## Prediction Heads

```{eval-rst}
.. autoclass:: bertblocks.modeling.head.Pooler
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.head.ProjectionPredictionHead
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.head.GLUPredictionHead
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.head.MLPPredictionHead
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.modeling.head.get_prediction_head
```

## Activations

```{eval-rst}
.. autofunction:: bertblocks.modeling.activations.get_actv_fn
```

## Loss Functions

```{eval-rst}
.. autofunction:: bertblocks.modeling.loss.get_loss_function
```

## Padding Utilities

```{eval-rst}
.. autofunction:: bertblocks.modeling.padding.unpad_input

.. autofunction:: bertblocks.modeling.padding.pad_output
```

## Scaling

```{eval-rst}
.. autoclass:: bertblocks.modeling.scale.LayerScaler
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.modeling.scale.LearnableLayerScaler
   :members:
   :show-inheritance:
```
