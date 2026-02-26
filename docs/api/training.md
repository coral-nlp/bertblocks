# Training

The `bertblocks.training` package provides PyTorch Lightning modules for pretraining
and finetuning, along with data loading, objectives, optimizers, and schedulers.

## Lightning Modules

```{eval-rst}
.. autoclass:: bertblocks.training.modules.BertBlocksPretrainingModule
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.modules.BertBlocksFinetuningModule
   :members:
   :show-inheritance:
```

## Data Modules

```{eval-rst}
.. autoclass:: bertblocks.training.modules.BertBlocksPretrainingDataModule
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.modules.BertBlocksFinetuningDataModule
   :members:
   :show-inheritance:
```

## Training Objectives (Collators)

Collators prepare batches for specific training objectives. Each collator handles
tokenization, masking, and label creation for its task.

```{eval-rst}
.. autoclass:: bertblocks.training.objectives.Collator
   :members:

.. autoclass:: bertblocks.training.objectives.MaskedLanguageModelingCollator
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.objectives.EnhancedMaskedLanguageModelingCollator
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.objectives.TokenClassificationCollator
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.objectives.SequenceClassificationCollator
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.objectives.QuestionAnsweringCollator
   :members:
   :show-inheritance:

.. autoclass:: bertblocks.training.objectives.MaskedDiffusionCollator
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.training.objectives.get_collator_cls
```

## Optimizers

```{eval-rst}
.. autofunction:: bertblocks.training.optimizer.get_optimizer
```

## Schedulers

```{eval-rst}
.. autoclass:: bertblocks.training.scheduler.InverseSqrtScheduler
   :members:
   :show-inheritance:

.. autofunction:: bertblocks.training.scheduler.get_scheduler

.. autofunction:: bertblocks.training.scheduler.get_single_scheduler
```

## Metrics

```{eval-rst}
.. autofunction:: bertblocks.training.metrics.get_metrics_for_task
```

## Sequence Packing

```{eval-rst}
.. autoclass:: bertblocks.training.packing.DistributedStoppingDataLoader
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: bertblocks.training.packing.PackedDataset
   :members:
   :show-inheritance:
```
