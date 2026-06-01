# RigMo-VAE

Training code for the **RigMo-VAE**, the rig–motion autoencoder from:

> **RigMo: Unifying Rig and Motion Learning for Generative Animation**
> Hao Zhang, Jiahao Luo, Bohui Wan, Yizhou Zhao, Zongrui Li, Michael Vasilkovsky, Chaoyang Wang, Jian Wang, Narendra Ahuja, Bing Zhou
> *Snap Inc., UIUC, UC Santa Cruz, CMU, NTU*
> Paper: https://arxiv.org/pdf/2601.06378 · Project page: https://rigmo-page.github.io/

RigMo jointly learns **rig** and **motion** directly from raw mesh sequences, with no
human-provided skeletons or skinning weights. A dual-path topology-aware encoder
disentangles static geometry (rigging branch) and dynamic motion (motion branch) into
two compact latent spaces. The decoder produces explicit **Gaussian bones** (from which
skinning weights are derived) and per-frame **SE(3)** transformations, and a Gaussian
Linear Blend Skinning (LBS) module reconstructs the deformed mesh.

This repository contains a minimal, self-contained version of the RigMo-VAE training
pipeline, including the **temporal-attention** variant used in the paper experiments.

## Architecture

```
vertices [B, T, N, 3]
        │
        ▼
TopologyAwareEncoder ── rigging branch (V0)      → V0 anchors  (Gaussian bones)
                     └─ motion  branch (V_delta)  → bone-motion features
        │
        ├─ StaticParamDecoder        → Gaussian bones  G = [Δc, s, q]
        ├─ DynamicVAEEncoder/Decoder → per-bone local SE(3)  (latent z, KL)
        └─ RootVAEEncoder/Decoder    → global root SE(3)      (latent z, KL)
        │  (optional TemporalTransformerBlock mixes information across frames)
        ▼
GaussianSkinningLBS → animated_vertices [B, T-1, N, 3]
```

Key modules:
- `step1x3d_geometry/models/autoencoders/mesh_motion_vae.py` — encoder, decoders, LBS, losses.
- `step1x3d_geometry/systems/mesh_motion_autoencoder.py` — Lightning training/val/test system.
- `step1x3d_geometry/datamodules/mesh_motion.py` — dataset / data module.

## Installation

```bash
conda create -n rigmo python=3.10 -y
conda activate rigmo

# Install a PyTorch build that matches your CUDA version, e.g. CUDA 12.4:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Data format

The data module (`FullMeshMotionNPZ-datamodule`) expects a root directory of sequences,
where each sequence is a directory of per-frame `.npz` files:

```
data/rigmo_data/
├── <sequence_name>/
│   ├── frame_0000.npz       # arrays: vertices [N, 3], neighbor_idx [N, k]
│   ├── frame_0001.npz
│   └── ...
├── val/                     # reserved sub-dir used for validation
│   └── <sequence_name>/ ...
└── test/                    # reserved sub-dir used for testing
    └── <sequence_name>/ ...
```

Each `frame_*.npz` stores:
- `vertices`: `float32` array of shape `[N, 3]`.
- `neighbor_idx`: `int` array of shape `[N, k]` — per-vertex neighbor indices (mesh
  topology) used by the topology-aware attention.

Sequences are normalized per-sequence so the first frame's bounding box maps to a unit
cube centered at the origin. Point this at your dataset by editing `data.root_dir` in the
config.

## Training

Single node (1 GPU, good for a quick sanity run):

```bash
bash scripts/train_single_node.sh configs/rigmo_vae_temporal_single_node.yaml 1
```

Single node, multiple GPUs (e.g. 8):

```bash
bash scripts/train_single_node.sh configs/rigmo_vae_temporal_single_node.yaml 8
```

Multi-node via SLURM (reproduces the 8-node × 8-GPU run from the paper):

```bash
sbatch scripts/run_train_slurm.sh configs/rigmo_vae_temporal.yaml
```

You can also call the entrypoint directly:

```bash
python train.py --config configs/rigmo_vae_temporal_single_node.yaml --train
```

Other modes: `--validate`, `--test`, `--export` (use `--resume path/to/ckpt.ckpt` to load
a checkpoint).

### Logging

TensorBoard and CSV logging are enabled by default and written under `outputs/`.
Weights & Biases is disabled by default; set `system.loggers.wandb.enable: true` in the
config and run `wandb login` to enable it.

## Configuration notes

- `system.shape_model.use_temporal_attn: true` enables cross-frame temporal attention
  (the "temporal-attn" variant). Set to `false` for the baseline.
- `data.num_frames` and `system.shape_model.num_frames` must match.
- `system.shape_model.num_tokens` is the number of Gaussian bones `K`.
- `system.shape_model.use_checkpoint: true` enables gradient checkpointing to save memory.

## Citation

```bibtex
@article{zhang2026rigmo,
  title   = {RigMo: Unifying Rig and Motion Learning for Generative Animation},
  author  = {Zhang, Hao and Luo, Jiahao and Wan, Bohui and Zhao, Yizhou and Li, Zongrui
             and Vasilkovsky, Michael and Wang, Chaoyang and Wang, Jian and Ahuja, Narendra
             and Zhou, Bing},
  journal = {arXiv preprint arXiv:2601.06378},
  year    = {2026}
}
```
