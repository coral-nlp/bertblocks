"""Small smoke test for grid_search.py using exactly one hyperparameter combination."""

from .grid_search import run_grid_search, best_per_task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal grid search test")
    parser.add_argument("benchmark", type=str, choices=["glue"], help="Benchmark to run")
    parser.add_argument("model", type=str, help="Name or path of pretrained model")
    parser.add_argument("--tokenizer", "-t", type=str, default=None, help="Tokenizer name or path")
    parser.add_argument("--max_seq_len", "-ms", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--eval_batch_size", "-be", type=int, default=64, help="Eval batch size")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output CSV path for all results")
    args = parser.parse_args()

    match args.benchmark:
        case "glue":
            from bertblocks.benchmarks.glue import TASK_MODULES
        case _:
            raise ValueError(f"Unknown benchmark {args.benchmark}")

    df = run_grid_search(
        task_modules=TASK_MODULES,
        pretrained_model_name_or_path=args.model,
        pretrained_tokenizer_name_or_path=args.tokenizer,
        max_seq_length=args.max_seq_len,
        eval_batch_size=args.eval_batch_size,
        learning_rates=[1e-5],
        weight_decays=[0.0],
        train_batch_sizes=[16],
        max_epochs_list=[1, 2],
    )

    print("\n=== Best per task ===")
    print(best_per_task(df).to_string(index=False))

    if args.output is not None:
        df.to_csv(args.output, index=False)
        print(f"\nAll results saved to {args.output}")