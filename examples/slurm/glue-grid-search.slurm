#!/bin/bash -e
#SBATCH --job-name=bertblocks-glue-grid-search
#SBATCH --mem=96g
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --container-workdir=/mnt/ceph/storage/data-tmp/current/willy/bertblocks
#SBATCH --container-image=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
#SBATCH --gres=gpu:ampere:1

MODEL=${1:?Usage: sbatch glue-grid-search.slurm <model_name_or_path> [output.csv]}
OUTPUT=${2:-glue_grid_search_results.csv}

mkdir -p logs

pip install uv
uv sync

uv run python -m bertblocks.benchmarks.grid_search glue "$MODEL" --output "$OUTPUT"
