#!/bin/bash
# Single-node (1+ GPU) training for the RigMo-VAE temporal-attention model.
#
# Usage:
#   bash scripts/train_single_node.sh [CONFIG] [NUM_GPUS]
#
# Examples:
#   bash scripts/train_single_node.sh                                   # 1 GPU, single-node config
#   bash scripts/train_single_node.sh configs/rigmo_vae_temporal_single_node.yaml 4
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="${1:-configs/rigmo_vae_temporal_single_node.yaml}"
NUM_GPUS="${2:-1}"

# train.py reads CUDA_VISIBLE_DEVICES to determine the GPU count, so list them all.
# (Respect a pre-set CUDA_VISIBLE_DEVICES if the user already exported one.)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

echo "Repo:   $REPO_DIR"
echo "Config: $CONFIG"
echo "GPUs:   $NUM_GPUS  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train.py \
    --config "$CONFIG" \
    --train \
    trainer.devices="$NUM_GPUS" trainer.num_nodes=1
