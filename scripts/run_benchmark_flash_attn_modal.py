"""Run the flash_attn_varlen_func benchmark on Modal.

Benchmarks the CUDA kernel efficiency across a grid of sequence lengths and
window sizes to determine whether local attention is actually faster than global.

Usage:
    # Run with defaults
    modal run scripts/run_benchmark_flash_attn_modal.py

    # Custom grid
    modal run scripts/run_benchmark_flash_attn_modal.py \
        --seq-lengths "512 1024 2048 4096" \
        --window-sizes "-1 64 128 256"

    # Quick test
    modal run scripts/run_benchmark_flash_attn_modal.py \
        --seq-lengths "128 256" --window-sizes "-1 64" \
        --num-warmup 2 --num-iterations 5

To change the GPU type, edit the gpu= argument in the @app.function decorator.
"""

from pathlib import Path

import modal

VOL_MOUNT_PATH = Path("/vol")

app = modal.App(name="benchmark-flash-attn")

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.12-py3")
    .uv_sync(extras=["cu130"])
    .add_local_dir("scripts", remote_path="/scripts")
    .add_local_python_source("bertblocks")
)

output_vol = modal.Volume.from_name("seltz-neural-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="l4",
    timeout=30 * 60,
    volumes={str(VOL_MOUNT_PATH): output_vol},
)
def run_benchmark(
    seq_lengths: list[int] = [128, 256, 512, 1024, 2048, 4096],
    window_sizes: list[int] = [-1, 32, 64, 128, 256, 512],
    batch_size: int = 32,
    num_heads: int = 12,
    head_dim: int = 64,
    dtype: str = "bfloat16",
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> None:
    import subprocess

    cmd = [
        "python",
        "/scripts/benchmark_flash_attn.py",
        "--seq-lengths",
        *[str(s) for s in seq_lengths],
        "--window-sizes",
        *[str(w) for w in window_sizes],
        "--batch-size",
        str(batch_size),
        "--num-heads",
        str(num_heads),
        "--head-dim",
        str(head_dim),
        "--dtype",
        dtype,
        "--num-warmup",
        str(num_warmup),
        "--num-iterations",
        str(num_iterations),
    ]
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(
    seq_lengths: str = "128 256 512 1024 2048 4096",
    window_sizes: str = "-1 32 64 128 256 512",
    batch_size: int = 32,
    num_heads: int = 12,
    head_dim: int = 64,
    dtype: str = "bfloat16",
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> None:
    run_benchmark.remote(
        seq_lengths=[int(s) for s in seq_lengths.split()],
        window_sizes=[int(w) for w in window_sizes.split()],
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )
