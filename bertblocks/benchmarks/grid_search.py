"""Hyperparameter grid search for benchmark tasks."""

import itertools
import logging

import lightning as L
import pandas as pd
from tqdm import tqdm

from bertblocks.benchmarks.base import TaskModule

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)


def run_grid_search(
    task_modules: list[type[TaskModule]],
    pretrained_model_name_or_path: str,
    pretrained_tokenizer_name_or_path: str | None = None,
    max_seq_length: int = 256,
    eval_batch_size: int = 64,
    learning_rates: list[float] = (1e-5, 2e-5, 3e-5, 5e-5),
    weight_decays: list[float] = (0.0, 0.01, 0.1),
    train_batch_sizes: list[int] = (16, 32),
    max_epochs_list: list[int] = (3, 4, 5),
    output_path: str | None = "result.csv",
) -> pd.DataFrame:
    """Run hyperparameter grid search over all tasks.

    For each task, trains and evaluates with every combination of the provided
    hyperparameters. Returns a DataFrame with one row per (task, HP combo).

    Args:
        task_modules: List of TaskModule subclasses to evaluate.
        pretrained_model_name_or_path: HuggingFace model name or path.
        pretrained_tokenizer_name_or_path: HuggingFace tokenizer name or path.
            If None, uses pretrained_model_name_or_path.
        max_seq_length: Maximum sequence length for tokenization.
        eval_batch_size: Batch size for evaluation.
        learning_rates: Learning rates to search over.
        weight_decays: Weight decay values to search over.
        train_batch_sizes: Training batch sizes to search over.
        max_epochs_list: Numbers of training epochs to search over.

    Returns:
        DataFrame with columns: Name, Group, Type, Metric, Score,
        learning_rate, weight_decay, train_batch_size, max_epochs.
    """
    if pretrained_tokenizer_name_or_path is None:
        pretrained_tokenizer_name_or_path = pretrained_model_name_or_path

    combos = list(itertools.product(learning_rates, weight_decays, train_batch_sizes, max_epochs_list))
    results = []

    pbar = tqdm(total=len(task_modules) * len(combos))
    for task_cls in task_modules:
        for lr, wd, bs, epochs in combos:
            pbar.set_description(f"{task_cls.__name__} lr={lr} wd={wd} bs={bs} ep={epochs}")
            trainer = L.Trainer(
                logger=False,
                max_epochs=epochs,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )
            task = task_cls(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                pretrained_tokenizer_name_or_path=pretrained_tokenizer_name_or_path,
                max_seq_length=max_seq_length,
                learning_rate=lr,
                weight_decay=wd,
                train_batch_size=bs,
                eval_batch_size=eval_batch_size,
                max_epochs=epochs,
            )
            trainer.fit(task)
            metrics = trainer.test(task, verbose=False)
            for k, v in metrics[0].items():
                results.append(
                    {
                        "Name": task_cls.__name__,
                        "Group": task.task_group,
                        "Type": task.task_type,
                        "Metric": k,
                        "Score": v,
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "train_batch_size": bs,
                        "max_epochs": epochs,
                    }
                )
            del trainer
            del task
            if output_path is not None:
                pd.DataFrame(results).to_csv(output_path, index=False)
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(results)


def best_per_task(df: pd.DataFrame) -> pd.DataFrame:
    """Return the best-scoring row per task."""
    return df.loc[df.groupby("Name")["Score"].idxmax()].reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run benchmark with hyperparameter grid search",
        epilog=(
            "Examples:\n"
            "  # Run full GLUE benchmark\n"
            "  python -m bertblocks.benchmarks.grid_search glue bert-base-uncased\n\n"
            "  # Run on specific tasks only\n"
            "  python -m bertblocks.benchmarks.grid_search glue bert-base-uncased --task cola mrpc\n\n"
            "  # Run on a single task with a custom output path\n"
            "  python -m bertblocks.benchmarks.grid_search glue bert-base-uncased --task sst2 -o sst2_custom.csv"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("benchmark", type=str, choices=["glue"], help="Benchmark to run")
    parser.add_argument("model", type=str, help="Name or path of pretrained model")
    parser.add_argument("--task", nargs="+", default=None, metavar="TASK",
                        help="Task name(s) to run (e.g. cola mrpc). Defaults to all tasks.")
    parser.add_argument("--tokenizer", "-t", type=str, default=None, help="Tokenizer name or path")
    parser.add_argument("--max_seq_len", "-ms", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--eval_batch_size", "-be", type=int, default=64, help="Eval batch size")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path (saved after each iteration)")

    args = parser.parse_args()

    match args.benchmark:
        case "glue":
            from bertblocks.benchmarks.glue import TASK_MODULES
        case _:
            raise ValueError(f"Unknown benchmark {args.benchmark}")

    if args.task is not None:
        requested = {t.lower() for t in args.task}
        task_modules = [m for m in TASK_MODULES if m.task_name in requested]
        unknown = requested - {m.task_name for m in task_modules}
        if unknown:
            parser.error(f"Unknown task(s): {', '.join(sorted(unknown))}")
    else:
        task_modules = TASK_MODULES

    if args.output is None and args.task is not None:
        output_path = "_".join(sorted(requested)) + "_results.csv"
    else:
        output_path = args.output

    df = run_grid_search(
        task_modules=task_modules,
        pretrained_model_name_or_path=args.model,
        pretrained_tokenizer_name_or_path=args.tokenizer,
        max_seq_length=args.max_seq_len,
        eval_batch_size=args.eval_batch_size,
        output_path=output_path,
    )

    print("\n=== Best per task ===")
    print(best_per_task(df).to_string(index=False))

    if output_path is not None:
        print(f"\nAll results saved to {output_path}")
