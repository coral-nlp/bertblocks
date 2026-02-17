"""Benchmark runner for evaluation tasks."""

import logging
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import yaml
from tqdm import tqdm

from bertblocks.benchmarks.base import TaskModule

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)


def load_task_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load per-task hyperparameter overrides from a YAML file.

    Expected format:
        CoLA:
          learning_rate: 1e-5
          epochs: 5
          weight_decay: 0.001
        SST2:
          learning_rate: 3e-5

    Supported keys per task: learning_rate, epochs, weight_decay.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Task config must be a YAML mapping, got {type(config).__name__}")
    valid_keys = {"learning_rate", "epochs", "weight_decay"}
    for task_name, overrides in config.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Task config for '{task_name}' must be a mapping, got {type(overrides).__name__}")
        unknown = set(overrides.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown keys for task '{task_name}': {unknown}. Valid keys: {valid_keys}")
    return config


def run_eval(
    task_modules: list[type[TaskModule]],
    pretrained_model_name_or_path: str,
    pretrained_tokenizer_name_or_path: str | None = None,
    max_seq_length: int = 256,
    max_epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    task_config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run evaluation on a list of task modules.

    Args:
        task_modules: List of TaskModule subclasses to evaluate.
        pretrained_model_name_or_path: HuggingFace model name or path.
        pretrained_tokenizer_name_or_path: HuggingFace tokenizer name or path.
            If None, uses pretrained_model_name_or_path.
        max_seq_length: Maximum sequence length for tokenization.
        max_epochs: Number of training epochs per task.
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        train_batch_size: Batch size for training.
        eval_batch_size: Batch size for evaluation.
        task_config: Optional per-task hyperparameter overrides. Keys are task
            class names, values are dicts with optional keys: learning_rate,
            epochs, weight_decay.

    Returns:
        DataFrame with columns: Name, Group, Type, Metric, Score
    """
    if pretrained_tokenizer_name_or_path is None:
        pretrained_tokenizer_name_or_path = pretrained_model_name_or_path

    results = []
    pbar = tqdm(total=len(task_modules))
    for task_cls in task_modules:
        pbar.set_description(task_cls.__name__)
        overrides = (task_config or {}).get(task_cls.__name__, {})
        task_epochs = overrides.get("epochs", max_epochs)
        task_lr = overrides.get("learning_rate", learning_rate)
        task_wd = overrides.get("weight_decay", weight_decay)
        trainer = L.Trainer(
            logger=False,
            max_epochs=task_epochs,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
        )
        task = task_cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pretrained_tokenizer_name_or_path=pretrained_tokenizer_name_or_path,
            max_seq_length=max_seq_length,
            learning_rate=task_lr,
            weight_decay=task_wd,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
        )
        trainer.fit(task)
        metrics = trainer.test(task)
        for k, v in metrics[0].items():
            results.append(
                {
                    "Name": task.task_name,
                    "Group": task.task_group,
                    "Type": task.task_type,
                    "Metric": k,
                    "Score": v,
                }
            )
        del trainer
        del task
        pbar.update(1)
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("benchmark", type=str, choices=["glue", "supergleber"], help="Benchmark to run")
    parser.add_argument("model", type=str, help="Name or path of pretrained model")
    parser.add_argument("--tokenizer", "-t", type=str, required=False, help="Tokenizer name or path", default=None)
    parser.add_argument("--max_seq_len", "-ms", type=int, required=False, help="Maximum sequence length", default=512)
    parser.add_argument("--epochs", "-e", type=int, required=False, help="Maximum train epochs", default=3)
    parser.add_argument("--learning_rate", "-lr", type=float, required=False, help="Learning rate", default=2e-5)
    parser.add_argument("--weight_decay", "-wd", type=float, required=False, help="Weight decay", default=0.01)
    parser.add_argument("--train_batch_size", "-bt", type=int, required=False, help="Train batch size", default=64)
    parser.add_argument("--eval_batch_size", "-be", type=int, required=False, help="Eval batch size", default=128)
    parser.add_argument("--output", "-o", type=str, required=False, help="Output CSV path", default=None)
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=False,
        help="Path to YAML file with per-task hyperparameter overrides (learning_rate, epochs, weight_decay)",
        default=None,
    )

    args = parser.parse_args()

    # Benchmark modules
    match args.benchmark:
        case "glue":
            from bertblocks.benchmarks.glue import TASK_MODULES
        case "supergleber":
            from bertblocks.benchmarks.supergleber import TASK_MODULES
        case _:
            raise ValueError(f"Unknown benchmark {args.benchmark}")

    cfg = load_task_config(args.config) if args.config else None

    df = run_eval(
        task_modules=TASK_MODULES,
        pretrained_model_name_or_path=args.model,
        pretrained_tokenizer_name_or_path=args.tokenizer,
        max_seq_length=args.max_seq_len,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        task_config=cfg,
    )
    print(df)
    if args.output is not None:
        df.to_csv(args.output, index=False)
