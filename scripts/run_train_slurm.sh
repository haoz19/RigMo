#!/bin/bash
# Multi-node SLURM launcher for RigMo-VAE training.
# Reproduces the 8-node x 8-GPU run from the paper.
#
# Usage:
#   sbatch scripts/run_train_slurm.sh [CONFIG]
#SBATCH --job-name=rigmo
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/slurm-%j.out
#SBATCH --no-requeue

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/rigmo_vae_temporal.yaml}"

mkdir -p "$REPO_DIR/logs"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

NGPUS=${SLURM_GPUS_PER_TASK:-8}
NNODES=${SLURM_NNODES:-8}

echo "=========================================="
echo "RigMo-VAE multi-node training"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Nodes:      $NNODES (head: $head_node @ $head_node_ip)"
echo "  GPUs/node:  $NGPUS"
echo "  Total GPUs: $((NNODES * NGPUS))"
echo "  Config:     $CONFIG"
echo "=========================================="

export LOGLEVEL=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

srun bash -c "
cd '$REPO_DIR'
# train.py reads CUDA_VISIBLE_DEVICES to determine the per-node GPU count, so it
# MUST list all GPUs on the node (torchrun does not set this itself).
export CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$(( $NGPUS - 1 )))
echo \"[\$(hostname)] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
torchrun \
    --nproc_per_node='$NGPUS' \
    --nnodes='$NNODES' \
    --rdzv_id='$SLURM_JOB_ID' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='${head_node_ip}:29500' \
    --max_restarts=0 \
    train.py \
    --config '$CONFIG' \
    --train
"

echo "Training finished."
