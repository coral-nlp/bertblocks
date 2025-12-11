import argparse
import subprocess
import sys
from pathlib import Path

GLUE_TASKS = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]

# Task-specific batch size limits (for small datasets)
TASK_BSZ_LIMITS = {
    "mrpc": 32,
    "stsb": 32,
    "rte": 32,
    "wnli": 32,
    "cola": 32,
}


def run_all(
        script_path: str,
        model: str,
        trust_remote_code: bool,
        epochs: int,
        bsz: int,
        eval_bsz: int,
        lr: float,
        precision: str,
        num_workers: int,
        out_csv: str,
        wandb_mode: str,
        max_len: int,
        accum: int,
        seed: int,
        limit_train_batches: float,
        limit_val_batches: float,
        enable_ckpt: bool,
        extra_args: list[str],
        stop_on_error: bool,
):
    results_csv = Path(out_csv).resolve()

    for task in GLUE_TASKS:
        # Apply task-specific batch size limits
        task_bsz = TASK_BSZ_LIMITS.get(task, bsz)

        # Adjust accumulation to maintain effective batch size
        effective_target = bsz * accum
        task_accum = max(1, effective_target // task_bsz)

        cmd = [
            sys.executable, script_path,
            "--task", task,
            "--model", model,
            "--epochs", str(epochs),
            "--bsz", str(task_bsz),
            "--eval_bsz", str(eval_bsz),
            "--lr", str(lr),
            "--precision", precision,
            "--num_workers", str(num_workers),
            "--max_len", str(max_len),
            "--accum", str(task_accum),
            "--seed", str(seed),
            "--wandb_mode", wandb_mode,
            "--results_csv", str(results_csv),
            "--echo_json",
            "--limit_train_batches", str(limit_train_batches),
            "--limit_val_batches", str(limit_val_batches),
        ]

        if trust_remote_code:
            cmd.append("--trust_remote_code")
        if not enable_ckpt:
            cmd.append("--no_ckpt")

        cmd += extra_args

        print(f"\n{'=' * 60}")
        print(f"Running {task.upper()} (bsz={task_bsz}, accum={task_accum})")
        print(f"{'=' * 60}")
        print(" ".join(cmd))

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)

        if proc.returncode != 0:
            msg = f"[ERROR] Task {task} failed with exit code {proc.returncode}"
            print(msg, file=sys.stderr)
            if stop_on_error:
                raise SystemExit(msg)


def main():
    ap = argparse.ArgumentParser(
        description="Run GLUE benchmark across all tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--script", type=str, default="glue.py",
                    help="Path to training script")
    ap.add_argument("model", nargs="?", default="bert-base-uncased",
                    help="HuggingFace model name")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Allow custom code from model repos")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bsz", type=int, default=32,
                    help="Base batch size (may be reduced for small tasks)")
    ap.add_argument("--eval_bsz", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--precision", type=str, default="32-true",
                    choices=["32-true", "16-mixed", "bf16-mixed"])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="results_glue.csv")
    ap.add_argument("--wandb_mode", type=str, default="disabled",
                    choices=["disabled", "offline", "online"])
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--accum", type=int, default=1,
                    help="Gradient accumulation steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train_batches", type=float, default=1.0)
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--enable_ckpt", action="store_true",
                    help="Enable checkpointing (disabled by default)")
    ap.add_argument("--stop_on_error", action="store_true",
                    help="Stop on first task failure")

    args, extra = ap.parse_known_args()

    run_all(
        script_path=args.script,
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        epochs=args.epochs,
        bsz=args.bsz,
        eval_bsz=args.eval_bsz,
        lr=args.lr,
        precision=args.precision,
        num_workers=args.num_workers,
        out_csv=args.out_csv,
        wandb_mode=args.wandb_mode,
        max_len=args.max_len,
        accum=args.accum,
        seed=args.seed,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_ckpt=args.enable_ckpt,
        extra_args=extra,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    main()