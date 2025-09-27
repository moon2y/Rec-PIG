# PIG: Recsys‑Conditioned Personalized Image Generation

> **PIG** is a research codebase that fuses user preference embeddings from a recommender system with **Stable Diffusion v1.5** to generate **personalized images**.  
> Pipeline: **Sequence recommendation (multi‑modal BERT4Rec‑style)** → **User embedding extraction** → **Personalization via Deep Prompt Tokens (HyperNet) or LoRA** → **Text/Img2Img generation**.

---

## Table of Contents
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [1) Data Preprocessing](#1-data-preprocessing)
  - [2) Train the Recommender & Export User Embeddings](#2-train-the-recommender--export-user-embeddings)
  - [3) Train Personalized Diffusion (DPT or LoRA)](#3-train-personalized-diffusion-dpt-or-lora)
  - [4) Personalized Inference](#4-personalized-inference)
- [Configuration Notes](#configuration-notes)
- [Troubleshooting](#troubleshooting)
- [Results & Logs](#results--logs)

---

## Key Features
- **End‑to‑end pipeline**: MovieLens preprocessing → multi‑modal sequence model → user embedding export → **personalized** Stable Diffusion.
- **Deep Prompt HyperNet (DPT)**: generate **K dynamic tokens** from user embeddings and inject them into the text encoder.
- **LoRA** option for lightweight personalization.
- **Img2Img & Text‑only** modes, CFG‑safe negative prompt embedding alignment.
- **Multi‑GPU DDP** via `torch.distributed` / `torchrun`, with optional `xformers` and gradient checkpointing.

> ⚠️ This repository is for **research**. Please respect the licenses of datasets and base models (e.g., Stable Diffusion, CLIP, etc.).

---

## Repository Structure
```text
PIG/
├─ data/
│  ├─ raw/
│  │  └─ ml-latest/                # MovieLens original files (user-provided)
│  └─ preprocessed/                # RecBole atomic format (.inter / .item)
├─ diffusion/
│  ├─ train_deepprompt.py          # DPT training (HyperNet + optional proj_u2txt, user_ctr_head)
│  ├─ train_lora.py                # LoRA training
│  ├─ infer.py                     # Personalized generation (DPT/LoRA/Img2Img)
│  ├─ hypernet.py                  # DeepPromptHyperNet
│  ├─ utils.py, paths.py           # Logging / paths
│  ├─ checkpoints/                 # Saved weights
│  └─ results/                     # Generated images
├─ embedding/
│  ├─ clip/                        # Cached CLIP embeddings (optional)
│  ├─ user/                        # User preference embeddings (npy/json)
│  └─ vae/                         # VAE latent cache (optional)
├─ mmbert4rec/
│  ├─ data.py                      # Dataset & sequence builder
│  ├─ model.py                     # Multi-modal fusion & sequence model
│  ├─ trainer.py                   # Training loop
│  ├─ run.py                       # Entry point for training
│  └─ infer.py                     # Export user embeddings
├─ preprocessing/
│  ├─ mllatest_preprocess.py       # MovieLens → RecBole atomic (.inter/.item)
│  ├─ clip_mllatest.py             # CLIP text/poster embedding cache
│  └─ latents.py                   # (Optional) VAE latent extraction
└─ log/
   ├─ diffusion/                   # Diffusion logs
   └─ mmbert4rec/                  # Recommender logs
```

---

## Installation

- **Python**: 3.9+  
- **PyTorch**: 2.x (with CUDA for GPUs)  
- **Libraries**: `diffusers`, `transformers`, `accelerate`, `pandas`, `numpy`, `tqdm`, `Pillow`, `opencv-python` (optional), `sentencepiece`  
- **Optional**: `xformers` (for memory/speed), `faiss-cpu`/`faiss-gpu` (for large-scale embedding ops)

Example (CUDA 12.1 wheels; adjust to your CUDA/PyTorch env):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate
pip install pandas numpy tqdm pillow opencv-python sentencepiece
pip install xformers   # optional
```

---

## Quickstart

### 1) Data Preprocessing

1. Download **MovieLens `ml-latest`** and place the CSVs under:
   ```text
   data/raw/ml-latest/
     - ratings.csv
     - links.csv
     - movies.csv
     ...
   ```

2. Convert to **RecBole atomic** format:
   ```bash
   python preprocessing/mllatest_preprocess.py
   # Output: data/preprocessed/ml-latest.inter, ml-latest.item
   ```

3. (Optional) Cache **CLIP embeddings** (title text, poster image):
   ```bash
   python preprocessing/clip_mllatest.py
   # Output under embedding/clip/ (as configured inside the script)
   ```

4. (Optional) Cache **VAE latents** if your DPT setup expects them:
   ```bash
   python preprocessing/latents.py
   ```

---

### 2) Train the Recommender & Export User Embeddings

**Train**
```bash
python mmbert4rec/run.py \
  --item_path data/preprocessed/ml-latest.item \
  --inter_path data/preprocessed/ml-latest.inter \
  --epochs 20 --lr 1e-3 --batch_size 256 \
  --save_dir mmbert4rec/checkpoints_mllatest
```

**Export user embeddings**
```bash
python mmbert4rec/infer.py \
  --item_path data/preprocessed/ml-latest.item \
  --inter_path data/preprocessed/ml-latest.inter \
  --ckpt mmbert4rec/checkpoints_mllatest/last.pt \
  --out_npy embedding/user/user_pref_embs_all.npy \
  --out_users_json embedding/user/user_pref_embs_all_users.json
```

**Expected outputs**
- `embedding/user/user_pref_embs_all.npy` → shape `(num_users, d)`
- `embedding/user/user_pref_embs_all_users.json` → user ID mapping / meta

---

### 3) Train Personalized Diffusion (DPT or LoRA)

#### (A) DPT (Deep Prompt Tokens / HyperNet)
Trains a hyper-network that converts user embeddings into **K prompt tokens** injected into the text encoder.

```bash
# Use GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
MASTER_PORT=29505 \
torchrun --nproc_per_node=2 -m diffusion.train_deepprompt \
  --index ./embedding/vae/index.jsonl \
  --user_emb_npy ./embedding/user/user_pref_embs_all.npy \
  --user_emb_json ./embedding/user/user_pref_embs_all_users.json \
  --batch 1 --grad_accum 8 --epochs 5 --lr 1e-4 \
  --K 8 --train_proj \
  --xformers --grad_checkpoint --num_workers 8
```

- `--K`: number of DPT tokens
- `--train_proj`: also train `proj_u2txt` (user→text projection)
- `--xformers` & `--grad_checkpoint`: memory/speed optimizations (optional)

**Checkpoints** will be stored under `diffusion/checkpoints/` (e.g., `dpt_hypernet_ep{N}.pt`, `proj_u2txt.pt`, optionally `user_ctr_head_*.pt`).

#### (B) LoRA
```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 -m diffusion.train_lora \
  --epochs 5 --lr 1e-4 --batch 1 --grad_accum 8 \
  --xformers --grad_checkpoint
```

---

### 4) Personalized Inference

You can run **Img2Img** or **text-only** generation with either DPT or LoRA.

**DPT + proj_u2txt**
```bash
python diffusion/infer.py \
  --use_dpt \
  --dpt_ckpt diffusion/checkpoints/dpt_hypernet_ep5.pt \
  --proj_ckpt diffusion/checkpoints/proj_u2txt.pt \
  --user_emb_npy embedding/user/user_pref_embs_all.npy \
  --user_json   embedding/user/user_pref_embs_all_users.json \
  --init_image  path/to/reference_poster.jpg \
  --prompt      "A cinematic poster of <usr> ..." \
  --negative_prompt "blurry, low quality" \
  --steps 40 --guidance 7.5 \
  --out_dir diffusion/results/run_001
```

**LoRA**
```bash
python diffusion/infer.py \
  --use_lora \
  --lora_ckpt diffusion/checkpoints/lora_ep5.pt \
  --init_image path/to/reference_poster.jpg \
  --prompt "A modern minimalist poster for <usr> ..." \
  --out_dir diffusion/results/run_002
```

Notes:
- Defaults to **Stable Diffusion v1.5** (`runwayml/stable-diffusion-v1-5`).
- When using **DPT**, the code makes sure `negative_prompt_embeds` length **matches** the positive path (accounts for user tokens) for CFG stability.
- `<usr>` is a reserved token that gets **expanded by DPT** to inject user preference into text prompts.

---

## Configuration Notes
- **GPU selection**: use `CUDA_VISIBLE_DEVICES` to pin GPUs.
- **Batch/accumulation**: tune `--batch` and `--grad_accum` for your VRAM.
- **xformers**: gives substantial memory & speed benefits but may require correct CUDA/PyTorch builds.

---

## Troubleshooting
- **OOM**  
  Lower `--batch`, increase `--grad_accum`, enable `--grad_checkpoint`, use lower `--K`, or downscale input images.
- **Tokenizer length mismatch (CFG / negative prompts)**  
  Ensure DPT token count is mirrored in the negative branch (this repository’s `infer.py` already handles it).

---

## Results & Logs
- **Checkpoints**: `diffusion/checkpoints/`  
- **Generated images**: `diffusion/results/`  
- **Logs**: `log/diffusion/`, `log/mmbert4rec/`  

Adjust base directories in `diffusion/paths.py` and logging behavior in `diffusion/utils.py` as needed.

---
