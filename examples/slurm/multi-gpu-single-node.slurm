#!/bin/bash
#SBATCH --job-name=bertblocks-pretrain
#SBATCH --gres=gpu:hopper:2 # 2x H100 80GB
#SBATCH --ntasks-per-node=2 # All located on the same host
#SBATCH --cpus-per-task=8   # 2x8 = 16 cores total (make num_workers match this)
#SBATCH --mem=128GB         # 128GB total
#SBATCH --time=47:00:00     # We have a max job time of 48:00:00 on the cluster
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG=${1:?Usage: sbatch pretrain.slurm <config.yaml>}

mkdir -p logs

# Launch training (one process per GPU with Lightning DDP)
srun \
    --container-image=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel \
    --container-remap-root \
    --container-writable \
    bash -c "
        cd ~/bertblocks
        export UV_PROJECT_ENVIRONMENT=/tmp/bertblocks-venv
        export TRITON_CACHE_DIR=/tmp/triton_cache
        export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
        export MASTER_ADDR=127.0.0.1
        export NCCL_TIMEOUT=300  # seconds
        export NCCL_SOCKET_IFNAME=bond0
        export NCCL_P2P_LEVEL=NVL
        pip install uv
        uv sync --extra flash-attn --extra optimizers --extra wandb
        uv run python -m bertblocks fit \
            --config $CONFIG \
            --trainer.devices=2 \
            --trainer.strategy=ddp \
            --trainer.num_nodes=1 \
            --trainer.accumulate_grad_batches=24 \
            --data.init_args.train_batch_size=32 \
            --data.init_args.max_sequence_length=512 \
    "
