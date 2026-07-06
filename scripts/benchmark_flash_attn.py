"""Benchmark flash_attn_varlen_func across sequence lengths and window sizes.

Tests whether local attention (sliding window) is faster than global attention
at the raw CUDA kernel level, independent of any model overhead.

Usage:
    # Default grid: seq_lens=[128,256,512,1024,2048,4096], window_sizes=[-1,32,64,128,256,512]
    python scripts/benchmark_flash_attn.py

    # Custom grid
    python scripts/benchmark_flash_attn.py --seq-lengths 512 1024 2048 --window-sizes -1 64 128

    # Quick smoke test
    python scripts/benchmark_flash_attn.py --seq-lengths 128 256 --window-sizes -1 64 --num-warmup 2 --num-iterations 5
"""

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark flash_attn_varlen_func: seq_len x window_size grid")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Sequence lengths to benchmark (default: 128 256 512 1024 2048 4096)",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[-1, 32, 64, 128, 256, 512],
        help="Window sizes for local attention. Use -1 for global attention (default: -1 32 64 128 256 512)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads (default: 12)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (default: 64)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for q/k/v tensors (default: bfloat16)",
    )
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of timed iterations (default: 100)")
    return parser.parse_args()


def sync() -> None:
    torch.cuda.synchronize()


def make_qkv(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build packed q/k/v tensors and cu_seqlens for flash_attn_varlen_func.

    All sequences in the batch have the same length (no padding).
    Returns q, k, v of shape (batch_size * seq_len, num_heads, head_dim),
    cu_seqlens of shape (batch_size + 1,), and max_seq_len.
    """
    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=device)
    return q, k, v, cu_seqlens, seq_len


def compute_stats(latencies_ms: list[float], batch_size: int, seq_len: int) -> dict[str, float]:
    t = torch.tensor(latencies_ms)
    mean_ms = t.mean().item()
    num_tokens = batch_size * seq_len
    return {
        "mean_ms": mean_ms,
        "p50_ms": t.quantile(0.50).item(),
        "p95_ms": t.quantile(0.95).item(),
        "tokens_per_sec": num_tokens / (mean_ms / 1000.0),
    }


def benchmark_one(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    window_size: tuple[int, int],
    num_warmup: int,
    num_iterations: int,
) -> list[float]:
    from flash_attn import flash_attn_varlen_func

    with torch.no_grad():
        for _ in range(num_warmup):
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seq_len,
                max_seq_len,
                dropout_p=0.0,
                causal=False,
                softcap=0.0,
                window_size=window_size,
                deterministic=False,
            )
            sync()

        latencies_ms: list[float] = []
        for _ in range(num_iterations):
            sync()
            t0 = time.perf_counter()
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seq_len,
                max_seq_len,
                dropout_p=0.0,
                causal=False,
                softcap=0.0,
                window_size=window_size,
                deterministic=False,
            )
            sync()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return latencies_ms


def window_label(w: int) -> str:
    return "global" if w == -1 else f"w={w}"


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("flash_attn requires a CUDA device")

    dtype = getattr(torch, args.dtype)

    print(f"Device       : {device} ({torch.cuda.get_device_name(device)})")
    print(f"Batch size   : {args.batch_size}")
    print(f"Num heads    : {args.num_heads}")
    print(f"Head dim     : {args.head_dim}")
    print(f"Dtype        : {args.dtype}")
    print(f"Warmup iters : {args.num_warmup}")
    print(f"Timed iters  : {args.num_iterations}")
    print(f"Seq lengths  : {args.seq_lengths}")
    print(f"Window sizes : {args.window_sizes}")

    # Results: results[seq_len][window_size] = stats dict
    results: dict[int, dict[int, dict[str, float]]] = {}

    total_combos = len(args.seq_lengths) * len(args.window_sizes)
    combo = 0
    for seq_len in args.seq_lengths:
        results[seq_len] = {}
        q, k, v, cu_seqlens, max_seq_len = make_qkv(
            args.batch_size, seq_len, args.num_heads, args.head_dim, dtype, device
        )
        for win in args.window_sizes:
            combo += 1
            window_size = (-1, -1) if win == -1 else (win, win)
            print(f"\n[{combo}/{total_combos}] seq_len={seq_len}, window={window_label(win)} ...", flush=True)
            latencies = benchmark_one(
                q,
                k,
                v,
                cu_seqlens,
                max_seq_len,
                window_size=window_size,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
            )
            results[seq_len][win] = compute_stats(latencies, args.batch_size, seq_len)

    # --- Print results tables ---
    col_labels = [window_label(w) for w in args.window_sizes]
    col_w = max(max(len(c) for c in col_labels), 10)
    row_label_w = 10  # "seq_len" column

    def hline() -> str:
        return "+" + "-" * (row_label_w + 2) + ("+" + "-" * (col_w + 2)) * len(args.window_sizes) + "+"

    def header_row() -> str:
        cells = f"{'seq_len':^{row_label_w}}"
        for lbl in col_labels:
            cells += " | " + f"{lbl:^{col_w}}"
        return "| " + cells + " |"

    def data_row(seq_len: int, key: str, fmt: str, label_suffix: str = "") -> str:
        row_lbl = f"{seq_len}{label_suffix}"
        cells = f"{row_lbl:^{row_label_w}}"
        for win in args.window_sizes:
            val = results[seq_len][win][key]
            cells += " | " + f"{fmt.format(val):^{col_w}}"
        return "| " + cells + " |"

    for metric_name, key, fmt in [
        ("Mean Latency (ms)", "mean_ms", "{:.2f}"),
        ("p50  Latency (ms)", "p50_ms", "{:.2f}"),
        ("p95  Latency (ms)", "p95_ms", "{:.2f}"),
        ("Throughput (Mtok/s)", "tokens_per_sec", "{:.1f}"),
    ]:
        print(f"\n\n{'=' * (row_label_w + (col_w + 3) * len(args.window_sizes) + 3)}")
        print(f"  {metric_name}")
        print(hline())
        print(header_row())
        print(hline())
        for seq_len in args.seq_lengths:
            if key == "tokens_per_sec":
                # Convert tok/s -> Mtok/s
                row_lbl = f"{seq_len}"
                cells = f"{row_lbl:^{row_label_w}}"
                for win in args.window_sizes:
                    val = results[seq_len][win][key] / 1e6
                    cells += " | " + f"{fmt.format(val):^{col_w}}"
                print("| " + cells + " |")
            else:
                print(data_row(seq_len, key, fmt))
        print(hline())


if __name__ == "__main__":
    main()
