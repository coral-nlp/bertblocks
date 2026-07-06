"""Run BertBlocks pretraining on Modal (H100), logging to Weights & Biases.

Merge a trainer config, a data config, and a model-size config. Local ``configs/...`` paths are
rewritten to the in-image ``/configs`` mount automatically.

Usage:
    modal run examples/run_modal.py::train \
        --config configs/pretrain/trainer.yaml \
        --config configs/pretrain/data-fineweb-edu.yaml \
        --config configs/pretrain/model-17m.yaml

Swap ``model-17m.yaml`` for ``model-68m.yaml`` / ``model-150m.yaml`` to scale up.

Requires two Modal secrets: ``wandb-secret`` (WANDB_API_KEY) and ``huggingface-secret`` (HF_TOKEN).
"""

import subprocess
from pathlib import Path

import modal

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

VOL_MOUNT_PATH = Path("/vol")
REMOTE_CONFIG_DIR = Path("/configs")

app = modal.App(name="bertblocks-pretrain")

# BertBlocks has no private Seltz dependencies, so the image is a straight uv_sync of the CUDA extra
# (wandb is a core dependency and is installed by uv_sync).
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.12-py3")
    .uv_sync(extras=["cu130"])
    .env(
        {
            "HF_HOME": (VOL_MOUNT_PATH / "hf_cache").as_posix(),
        }
    )
    .add_local_dir("configs", REMOTE_CONFIG_DIR.as_posix())
    .add_local_python_source("bertblocks")
)

output_vol = modal.Volume.from_name("bertblocks-vol", create_if_missing=True)

secrets = [
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("huggingface-secret"),
]


def convert_config_paths(*arglist: str) -> list[str]:
    """Rewrite local ``--config configs/...`` paths to the in-image ``/configs`` mount."""
    args: list[str] = []
    i = 0
    while i < len(arglist):
        if arglist[i] == "--config" and i + 1 < len(arglist):
            args.append("--config")
            if arglist[i + 1].startswith("configs/"):
                args.append(Path(f"/{arglist[i + 1]}").as_posix())
            else:
                args.append((REMOTE_CONFIG_DIR / arglist[i + 1]).as_posix())
            i += 2
        else:
            args.append(arglist[i])
            i += 1
    return args


@app.function(
    image=image,
    gpu="H100",
    volumes={str(VOL_MOUNT_PATH): output_vol},
    timeout=16 * HOURS,
    secrets=secrets,
)
def train(*arglist: str) -> None:
    """Run BertBlocks pretraining (``python -m bertblocks fit``) on Modal."""
    args = convert_config_paths(*arglist)

    subprocess.run(
        [
            "python",
            "-m",
            "bertblocks",
            "fit",
            *args,
            # Point the W&B logger (and thus `trainer.log_dir`) at the persistent volume so the
            # HuggingFace checkpoint saved in `on_save_checkpoint` and the Lightning .ckpt files
            # survive the container. Mirrors seltz-neural's run_modal.
            "--trainer.logger.save_dir",
            (VOL_MOUNT_PATH / "logs").as_posix(),
            "--trainer.default_root_dir",
            (VOL_MOUNT_PATH / "checkpoints").as_posix(),
        ],
        check=True,
    )
