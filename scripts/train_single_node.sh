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

echo "Repo:   $REPO_DIR"
echo "Config: $CONFIG"
echo "GPUs:   $NUM_GPUS"

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train.py \
    --config "$CONFIG" \
    --train
