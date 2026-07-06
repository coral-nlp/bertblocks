"""Benchmark HuggingFace ModernBERT vs BertBlocks ModernBERT.

Usage examples:
    # Quick CPU smoke test (no weights downloaded)
    python scripts/benchmark_modernbert.py --no-weights --batch-size 4 --max-length 64 \
        --num-warmup 2 --num-iterations 5

    # Full GPU benchmark with uneven documents
    python scripts/benchmark_modernbert.py --batch-size 32 --max-length 512 --uneven

    # Vary batch size
    python scripts/benchmark_modernbert.py --batch-size 8 --max-length 256 --num-iterations 50
"""

import argparse
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from transformers import AutoConfig, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark HuggingFace vs BertBlocks ModernBERT")
    parser.add_argument(
        "--model",
        default="answerdotai/ModernBERT-base",
        help="HuggingFace model ID or local path (default: answerdotai/ModernBERT-base)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: 512)")
    parser.add_argument(
        "--uneven",
        action="store_true",
        help="Use randomly varying document lengths instead of full-length sequences",
    )
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of timed iterations (default: 100)")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on: 'cuda', 'cpu', etc. Auto-detects CUDA if not specified.",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Skip loading pretrained weights (random init). Useful for quick local testing.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="BertBlocks attention backend (default: sdpa).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model parameters and inputs (default: bfloat16).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile both models with torch.compile(dynamic=True) before benchmarking.",
    )
    parser.add_argument(
        "--profiler-save-dir",
        default=None,
        help="If set, run PyTorch profiler and save Chrome trace JSON files to this directory.",
    )
    return parser.parse_args()


def make_inputs(
    batch_size: int,
    max_length: int,
    vocab_size: int,
    uneven: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random input_ids and attention_mask.

    For even mode all sequences are full length (attention_mask all ones).
    For uneven mode each sequence gets a random length in [max_length // 8, max_length],
    with zero-padding to max_length.
    """
    # Use token IDs in [100, vocab_size) to stay away from special tokens at the start.
    input_ids = torch.randint(100, vocab_size, (batch_size, max_length), device=device)

    if uneven:
        min_length = max(1, max_length // 8)
        lengths = torch.randint(min_length, max_length + 1, (batch_size,))
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        # Zero out padding positions in input_ids too (good practice)
        input_ids = input_ids * attention_mask
    else:
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long, device=device)

    return input_ids, attention_mask


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
    device: torch.device,
) -> list[float]:
    """Run warmup then timed iterations, returning per-iter latencies in milliseconds."""
    model.eval()

    with torch.no_grad():
        for _ in tqdm(range(num_warmup), desc="Warmup", leave=False):
            model(input_ids, attention_mask=attention_mask)
            sync(device)

        latencies_ms: list[float] = []
        for _ in tqdm(range(num_iterations), desc="Benchmark"):
            sync(device)
            t0 = time.perf_counter()
            model(input_ids, attention_mask=attention_mask)
            sync(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return latencies_ms


def profile_model(
    name: str,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    save_dir: str,
) -> None:
    """Run a single forward pass under the PyTorch profiler and export a Chrome trace."""
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    with torch.no_grad(), profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function(name):
            model(input_ids, attention_mask=attention_mask)

    safe_name = name.replace(" ", "_")
    trace_path = os.path.join(save_dir, f"{safe_name}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"  Profiler trace saved: {trace_path}")


def compute_stats(latencies_ms: list[float], batch_size: int, num_tokens: int) -> dict[str, float]:
    t = torch.tensor(latencies_ms)
    mean_ms = t.mean().item()
    return {
        "mean_ms": mean_ms,
        "p50_ms": t.quantile(0.50).item(),
        "p95_ms": t.quantile(0.95).item(),
        "p99_ms": t.quantile(0.99).item(),
        "seqs_per_sec": batch_size / (mean_ms / 1000.0),
        "tokens_per_sec": num_tokens / (mean_ms / 1000.0),
    }


def print_results(name: str, stats: dict[str, float]) -> None:
    print(f"\n  {name}")
    print(f"    Mean latency : {stats['mean_ms']:>8.2f} ms")
    print(f"    p50  latency : {stats['p50_ms']:>8.2f} ms")
    print(f"    p95  latency : {stats['p95_ms']:>8.2f} ms")
    print(f"    p99  latency : {stats['p99_ms']:>8.2f} ms")
    print(f"    Throughput   : {stats['seqs_per_sec']:>10.1f} seq/s  |  {stats['tokens_per_sec']:>12.0f} tok/s")


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device       : {device}")
    print(f"Model        : {args.model}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Max length   : {args.max_length}")
    print(f"Uneven docs  : {args.uneven}")
    print(f"Warmup iters : {args.num_warmup}")
    print(f"Timed iters  : {args.num_iterations}")
    print(f"Load weights : {not args.no_weights}")
    print(f"Attn backend : {args.attn_implementation}")
    print(f"Compile      : {args.compile}")

    # Fetch config to get vocab_size (small JSON, no model weights downloaded).
    config = AutoConfig.from_pretrained(args.model)
    vocab_size: int = config.vocab_size
    dtype = getattr(torch, args.dtype)

    # Generate a fixed set of inputs used for both models.
    input_ids, attention_mask = make_inputs(
        batch_size=args.batch_size,
        max_length=args.max_length,
        vocab_size=vocab_size,
        uneven=args.uneven,
        device=device,
    )

    # Number of non-padding tokens per batch (for throughput calculation).
    num_tokens = int(attention_mask.sum().item())

    # --- Load HuggingFace model ---
    print("\nLoading HuggingFace model...")
    if args.no_weights:
        hf_model = AutoModel.from_config(config).to(device=device, dtype=dtype)
    else:
        hf_model = AutoModel.from_pretrained(args.model, attn_implementation=args.attn_implementation).to(
            device=device, dtype=dtype
        )
    hf_model.eval()

    # --- Load BertBlocks model ---
    print("Loading BertBlocks model...")
    import bertblocks as bb

    bb_config = bb.BertBlocksConfig.from_config(hf_model.config, attn_implementation=args.attn_implementation)
    # bb_config.block_pos_enc_kind = "alibi"
    # bb_config.global_attention_every_n_layers = 0

    bb_model = bb.BertBlocksModel(bb_config).to(device=device, dtype=dtype)
    bb_model.eval()

    if args.compile:
        print("\nCompiling models (dynamic=True)...")
        hf_model = torch.compile(hf_model, dynamic=True)
        bb_model = torch.compile(bb_model, dynamic=True)

    # --- Benchmark ---
    print("\nRunning benchmarks...")
    hf_latencies = benchmark_model(hf_model, input_ids, attention_mask, args.num_warmup, args.num_iterations, device)
    bb_latencies = benchmark_model(bb_model, input_ids, attention_mask, args.num_warmup, args.num_iterations, device)

    hf_stats = compute_stats(hf_latencies, args.batch_size, num_tokens)
    bb_stats = compute_stats(bb_latencies, args.batch_size, num_tokens)

    speedup = hf_stats["mean_ms"] / bb_stats["mean_ms"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_results("HuggingFace ModernBERT", hf_stats)
    print_results("BertBlocks  ModernBERT", bb_stats)
    print(f"\n  Speedup (BertBlocks / HuggingFace) : {speedup:.2f}x")
    print("=" * 60)

    if args.profiler_save_dir:
        os.makedirs(args.profiler_save_dir, exist_ok=True)
        print("\nRunning profiler...")
        profile_model("HuggingFace_ModernBERT", hf_model, input_ids, attention_mask, device, args.profiler_save_dir)
        profile_model("BertBlocks_ModernBERT", bb_model, input_ids, attention_mask, device, args.profiler_save_dir)


if __name__ == "__main__":
    main()
