"""Run the ModernBERT benchmark on Modal.

Usage:
    # Run with defaults (batch_size=32, max_length=512, even docs)
    modal run scripts/run_benchmark_modal.py

    # Custom configuration
    modal run scripts/run_benchmark_modal.py --batch-size 16 --max-length 256 --uneven

To change the GPU type, edit the gpu= argument in the @app.function decorator.
"""

from pathlib import Path

import modal

VOL_MOUNT_PATH = Path("/vol")

app = modal.App(name="benchmark")

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.12-py3")
    .uv_sync(extras=["cu130"])
    .env(
        {
            "HF_HOME": (VOL_MOUNT_PATH / "hf_cache").as_posix(),
        }
    )
    .add_local_dir("scripts", remote_path="/scripts")
    .add_local_python_source("bertblocks")
)

secrets = [
    modal.Secret.from_name("huggingface-secret"),
]

output_vol = modal.Volume.from_name("seltz-neural-vol", create_if_missing=True)


@app.function(
    image=image,
    gpu="l4",
    timeout=30 * 60,
    volumes={str(VOL_MOUNT_PATH): output_vol},
)
def run_benchmark(
    model: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32,
    max_length: int = 512,
    uneven: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 100,
    attn_implementation: str = "flash_attention_2",
    compile: bool = False,
    profiler_save_dir: str | None = None,
) -> None:
    import subprocess

    cmd = [
        "python",
        "/scripts/benchmark_modernbert.py",
        "--model",
        model,
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--num-warmup",
        str(num_warmup),
        "--num-iterations",
        str(num_iterations),
        "--attn-implementation",
        attn_implementation,
    ]
    if uneven:
        cmd.append("--uneven")
    if compile:
        cmd.append("--compile")
    if profiler_save_dir:
        cmd.extend(["--profiler-save-dir", profiler_save_dir])
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(
    model: str = "answerdotai/ModernBERT-base",
    batch_size: int = 32,
    max_length: int = 512,
    uneven: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 100,
    attn_implementation: str = "flash_attention_2",
    compile: bool = False,
    profiler_save_dir: str | None = None,
) -> None:
    run_benchmark.remote(
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        uneven=uneven,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        attn_implementation=attn_implementation,
        compile=compile,
        profiler_save_dir=profiler_save_dir,
    )
