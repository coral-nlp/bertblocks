"""Small smoke test for grid_search.py using exactly one hyperparameter combination."""

import re
from pathlib import Path

from .grid_search import run_grid_search, best_per_task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal grid search test")
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

    experiments_dir = Path(__file__).parent.parent / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    if args.output is None:
        model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.model).strip("_").lower()
        if args.task is not None:
            tasks_slug = "_".join(sorted(requested))
        else:
            tasks_slug = args.benchmark
        output_path = str(experiments_dir / f"{model_slug}_{tasks_slug}_small_results.csv")
    else:
        output_path = args.output

    df = run_grid_search(
        task_modules=task_modules,
        pretrained_model_name_or_path=args.model,
        pretrained_tokenizer_name_or_path=args.tokenizer,
        max_seq_length=args.max_seq_len,
        eval_batch_size=args.eval_batch_size,
        learning_rates=[8e-5],
        weight_decays=[1e-6],
        train_batch_sizes=[16, 32],
        max_epochs_list=[5],
        seeds=[42, 43, 44],
        output_path=output_path,
    )

    print("\n=== Best per task ===")
    print(best_per_task(df).to_string(index=False))

    if output_path is not None:
        print(f"\nAll results saved to {output_path}")
