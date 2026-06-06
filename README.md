<div align="center">

# RigMo: Unifying Rig and Motion Learning for Generative Animation

<p>
  <a href="https://haoz19.github.io/RigMo-page/"><img src="https://img.shields.io/badge/Project-Page-1f9bcf?style=flat-square" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2601.06378"><img src="https://img.shields.io/badge/arXiv-2601.06378-b31b1b?style=flat-square" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/python-3.10-3776ab?style=flat-square" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c?style=flat-square" alt="PyTorch 2.5">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-green?style=flat-square" alt="License: CC BY-NC 4.0"></a>
</p>

**Hao Zhang**<sup>1,2</sup>&nbsp;&nbsp; Jiahao Luo<sup>1,3</sup>&nbsp;&nbsp; Bohui Wan<sup>2</sup>&nbsp;&nbsp; Yizhou Zhao<sup>1,4</sup>&nbsp;&nbsp; Zongrui Li<sup>5</sup>&nbsp;&nbsp; Michael Vasilkovsky<sup>1</sup>&nbsp;&nbsp; Chaoyang Wang<sup>1</sup>&nbsp;&nbsp; Jian Wang<sup>1</sup>&nbsp;&nbsp; Narendra Ahuja<sup>2</sup>&nbsp;&nbsp; Bing Zhou<sup>1</sup>

<sup>1</sup>Snap Inc.&nbsp;&nbsp; <sup>2</sup>UIUC&nbsp;&nbsp; <sup>3</sup>UC Santa Cruz&nbsp;&nbsp; <sup>4</sup>CMU&nbsp;&nbsp; <sup>5</sup>NTU

</div>

---

> **TL;DR — Rigging and motion should not be learned in isolation.** RigMo is the first
> generative framework that discovers both **rig structure** and **motion dynamics**
> directly from raw mesh sequences, with no ground-truth rigs, skeletons, or per-sequence
> optimization. It factorizes deformation into explicit **Gaussian bones** and
> structure-aware motion, turning arbitrary deforming meshes into fully animatable assets.

This repository contains a minimal, self-contained implementation of the **RigMo-VAE**
training pipeline, including the **temporal-attention** variant used in the paper. The
Motion-DiT generative stage is not included here.

## ✨ Highlights

- **Annotation-free rigging** — learns Gaussian bones + skinning weights from raw mesh
  sequences, no artist-designed skeletons.
- **Dual-path encoder** — disentangles canonical geometry (rigging branch) from temporal
  deformation (motion branch).
- **Explicit & interpretable** — outputs Gaussian bones and per-frame SE(3) transforms,
  reconstructed via differentiable Gaussian LBS.
- **Temporal attention** — optional cross-frame attention for smoother, more coherent
  motion (`use_temporal_attn`).
- **Scalable** — multi-node training out of the box (reproduces the 8-node × 8-GPU run).

## 🧠 Architecture

```
                    vertices  V ∈ [B, T, N, 3]
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
      Rigging branch (V₀)               Motion branch (Vₜ − Vₜ₋₁)
   topology-aware self-attn          temporal–spatial self-attn
              │                                 │
   FPS → K bone tokens                          │
              ▼                                 ▼
   ┌──────────────────────┐        ┌──────────────────────────────┐
   │ StaticParamDecoder   │        │ Dynamic VAE  (local SE(3), z) │
   │ → Gaussian bones G   │        │ Root    VAE  (global SE(3), z)│
   │   G = [Δc, s, q]     │        │ (+ optional TemporalAttn)     │
   └──────────┬───────────┘        └───────────────┬──────────────┘
              └───────────────┬───────────────────┘
                              ▼
              GaussianSkinningLBS  →  V̂ ∈ [B, T−1, N, 3]
```

| File | Role |
|------|------|
| `step1x3d_geometry/models/autoencoders/mesh_motion_vae.py` | Encoder, decoders, Gaussian LBS, losses |
| `step1x3d_geometry/systems/mesh_motion_autoencoder.py` | Lightning training / val / test system |
| `step1x3d_geometry/datamodules/mesh_motion.py` | Dataset + data module |
| `train.py` | Entrypoint (`--train` / `--validate` / `--test` / `--export`) |

## 🚀 Installation

```bash
conda create -n rigmo python=3.10 -y
conda activate rigmo

# Install a PyTorch build matching your CUDA version, e.g. CUDA 12.4:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## 📦 Dataset

RigMo-VAE trains on sequences of deforming meshes (DeformingThings4D, Objaverse-XL renders,
TrueBones). A **ready-to-train preprocessed copy** (~18,985 sequences, ~534k frames) is
released on the Hugging Face Hub:

📥 **[huggingface.co/datasets/haoz19/RigMo-data](https://huggingface.co/datasets/haoz19/RigMo-data)**
&nbsp;(gated — click *Request access*; approved instantly-ish, then download)

### Download & extract

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # once, with a token that has access to the gated dataset

# 1. Download the archives (~28 GB)
huggingface-cli download haoz19/RigMo-data \
  --repo-type dataset --local-dir rigmo_data_archives

# 2. Extract into ./data/rigmo_data  (needs `zstd` + `tar`)
mkdir -p data/rigmo_data
for f in rigmo_data_archives/*.tar.zst; do
  tar -I zstd -xf "$f" -C data/rigmo_data
done
```

This yields the directory layout the data module (`FullMeshMotionNPZ-datamodule`) expects:

```
data/rigmo_data/
├── deformingthings4d/        # sequences derived from DeformingThings4D
├── objxl_rendered_*/         # Objaverse-XL render shards (8 dirs)
├── val/                      # reserved sub-dir for validation
└── test/                     # reserved sub-dir for testing (optional; falls back to val)
    └── <sequence_name>/
        ├── frame_0000.npz    # arrays:  vertices [N, 3]  ·  neighbor_idx [N, k]
        └── ...
```

Each `frame_*.npz` stores:

| Key | Shape | Description |
|-----|-------|-------------|
| `vertices` | `[N, 3]` `float32` | per-frame vertex positions (`N = 5000`) |
| `neighbor_idx` | `[N, k]` `int64` | per-vertex mesh neighbors (used by topology-aware attention) |

Sequences are normalized at load time so the first frame's bounding box maps to a unit
cube centered at the origin. The default config already points to `data/rigmo_data`; override
with `data.root_dir=/your/path` if you extract elsewhere.

## 🏋️ Training

**Single node, 1 GPU** (quick sanity run):

```bash
bash scripts/train_single_node.sh configs/rigmo_vae_temporal_single_node.yaml 1
```

**Single node, multiple GPUs** (e.g. 8):

```bash
bash scripts/train_single_node.sh configs/rigmo_vae_temporal_single_node.yaml 8
```

**Multi-node via SLURM** (reproduces the 8-node × 8-GPU run from the paper):

```bash
sbatch scripts/run_train_slurm.sh configs/rigmo_vae_temporal.yaml
```

Direct invocation:

```bash
python train.py --config configs/rigmo_vae_temporal_single_node.yaml --train
# other modes: --validate / --test / --export   (add --resume path/to/ckpt.ckpt)
```

**Logging.** TensorBoard + CSV logs are written under `outputs/` by default. To enable
Weights & Biases, set `system.loggers.wandb.enable: true` in the config and run
`wandb login`.

## ⚙️ Key configuration

| Field | Meaning |
|-------|---------|
| `system.shape_model.use_temporal_attn` | enable cross-frame temporal attention (the *temporal-attn* variant) |
| `system.shape_model.num_tokens` | number of Gaussian bones `K` |
| `system.shape_model.use_checkpoint` | gradient checkpointing to save memory |
| `data.num_frames` / `shape_model.num_frames` | sequence length (must match) |
| `trainer.num_nodes` / `trainer.devices` | distributed layout |

Two configs are provided: `configs/rigmo_vae_temporal.yaml` (the paper's 8-node setup) and
`configs/rigmo_vae_temporal_single_node.yaml` (single-node quick start).

## 📚 Citation

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

## 📄 License

The code in this repository is released under the
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE)
— free for non-commercial research and academic use with attribution. For commercial
licensing, please contact the authors.

The accompanying dataset is distributed separately under its own terms (see the
[Hugging Face dataset card](https://huggingface.co/datasets/haoz19/RigMo-data)); it is derived from DeformingThings4D, Objaverse-XL,
and TrueBones, and remains subject to those sources' original licenses.

## 🙏 Acknowledgements

The model code builds on the [Step1X-3D](https://github.com/stepfun-ai/Step1X-3D) geometry
framework. We thank the authors of DeformingThings4D, Objaverse-XL, and TrueBones for their
datasets.
