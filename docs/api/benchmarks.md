# Benchmarks

The `bertblocks.benchmarks` package provides evaluation suites for finetuned models.

## Running Evaluations

```{eval-rst}
.. autofunction:: bertblocks.benchmarks.run_eval
```

## Task Modules

```{eval-rst}
.. autoclass:: bertblocks.benchmarks.base.TaskModule
   :members:
   :show-inheritance:
```

### GLUE

```{eval-rst}
.. autoclass:: bertblocks.benchmarks.glue.GLUETaskModule
   :members:
   :show-inheritance:
```

Individual GLUE tasks: {class}`~bertblocks.benchmarks.glue.CoLA`,
{class}`~bertblocks.benchmarks.glue.SST2`,
{class}`~bertblocks.benchmarks.glue.MRPC`,
{class}`~bertblocks.benchmarks.glue.QQP`,
{class}`~bertblocks.benchmarks.glue.STSB`,
{class}`~bertblocks.benchmarks.glue.MNLI`,
{class}`~bertblocks.benchmarks.glue.QNLI`,
{class}`~bertblocks.benchmarks.glue.RTE`,
{class}`~bertblocks.benchmarks.glue.WNLI`.

### SuperGLEBer

```{eval-rst}
.. autoclass:: bertblocks.benchmarks.supergleber.SuperGLEBerTaskModule
   :members:
   :show-inheritance:
```
