---
name: lvsa-quickstart
description: Install LVSA and generate your first long video. Use when setting up LVSA from scratch, picking SDPA vs FlashInfer backend, configuring LVSA_REFERENCE_LATENT_FRAMES for a model, or verifying the sparse path engaged via [LVSA] log lines.
---

# LVSA Quickstart

## Overview

LVSA (Long Video Sparse Attention) is a training-free block-sparse attention engine for video diffusion transformers (Wan 2.x, HunyuanVideo 1.5, CogVideoX). It accelerates long-video generation 1.4–3.8× and enables generation beyond the training horizon where dense attention OOMs on 80 GB GPUs. Code: <https://github.com/JiusiServe/LongVideoSparseAttention>.

This skill covers: installation, choosing a backend, setting the per-model reference, running a first generation, and verifying that LVSA is actually engaged (not silently falling back to dense).

## Install

```bash
git clone https://github.com/JiusiServe/LongVideoSparseAttention
cd LVSA

# Use uv (or any venv tool)
uv venv --python 3.12
source .venv/bin/activate

# Core library
uv pip install -e ".[diffusers,hunyuan,dev]"

# FlashInfer backend (optional but recommended on CUDA — fastest at long sequences)
uv pip install -e ".[flashinfer]"
# Requires nvcc + CUDA toolkit for JIT compilation.

# vllm-omni plugin (optional — only for serving; use a separate .venv-vllm).
# vllm-omni 0.22.0rc1 is a pre-release built from git (not on PyPI); versions
# intentionally differ (it rebases onto vllm 0.22.0). For the older
# pip-installable 0.18.0 pair, use the release/v0.18.x branch.
uv pip install -e lvsa-vllm-omni/
uv pip install "vllm==0.22.0"
uv pip install --no-build-isolation \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.22.0rc1"

# VQeval (optional — for quality benchmarking)
uv pip install -e vqeval/
```

Verify:
```bash
pytest tests/ lvsa-vllm-omni/tests/ -v
# Expect 280+ tests passing (CPU-only, no GPU needed)
```

## Pick a backend

| Backend | When to use | Hardware |
|---|---|---|
| **SDPA** (default) | Always available; reasonable at training horizon | CUDA + Ascend NPU |
| **FlashInfer** | Fastest at extension (T_lat ≥ 49) | CUDA only (needs nvcc) |

The `--flashinfer` flag enables FlashInfer; otherwise SDPA. **If FlashInfer install fails on your box** (no nvcc / wrong CUDA), the dispatcher will refuse to fall back silently — it'll raise a clear error and you can re-run without `--flashinfer`.

## Pick the right reference per model

This is the single most common LVSA configuration mistake. **Always set explicitly.**

| Model | `LVSA_REFERENCE_LATENT_FRAMES` | Video frames at 1× |
|---|---|---|
| Wan 2.1 / 2.2 (1.3B, 14B) | `21` | 81 |
| HunyuanVideo 1.5 | `33` | 129 |
| CogVideoX 5B | `13` | 49 |

For the standalone example scripts the value is wired automatically by `lvsa/adapters/<model>.py::reference_latent_frames()`. Override only if your fork uses non-default geometry.

## Run a first generation

### Single GPU, training horizon (no extension)

```bash
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 81 \
    --lvsa --auto-keyframes \
    --output-name dog_1x.mp4
```

At T_lat ≤ reference, the auto-scheduler returns `kfi=1` → fully-dense attention via the LVSA path. You get the implementation-bypass speedup (~1.5–2×) but no pattern-driven sparsity.

### Single GPU, 4× horizon (the headline case)

```bash
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 321 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name dog_4x.mp4
```

Add `--riflex --riflex-s 4.0` to stack RIFLEx RoPE rescaling on top.

### Multi-GPU (Ulysses context parallel)

```bash
torchrun --nproc_per_node=2 examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 481 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name dog_6x.mp4
```

**Constraint:** `seq_len = T_lat × patches_per_frame` must be divisible by `nproc_per_node`.

## Verify LVSA engaged

After the run, look for these `[LVSA]` lines (the LVSA prints prefix with `[LVSA]` for standalone, `[LVSA]` for vllm-omni):

```
[LVSA] --rotate-keyframes: computed key_frame_interval=6 (latent frames)
[LVSA] total_lat_frames=81 local_seq=126360 rank0_frames~[0,80] window=3 n_first=1 kfi=6 global_count=14 attended_per_frame=21/81
[LVSA] installed on 30 blocks num_patches=1560 total_lat_frames=81 backend=FlashInfer CSR: MB=81 nnz=1701 density=25.9% block_size=1560 compact=81/81frames
```

Read it as:
- `attended_per_frame=N/T` — N out of T frames are in each query's attention set. **Sparsity = `1 − N/T`**. At training reference, N==T (dense).
- `backend=FlashInfer|SDPA` — confirms the requested backend was selected.
- `installed on N blocks` — number of attention layers LVSA wrapped.

If you see no `[LVSA]` lines, the `--lvsa` flag wasn't passed. If `attended_per_frame=N/T` shows `N == T` at extension lengths, your `--num-frames` is below the training reference (no real sparsity engaged) — or the geometry detection failed.

## Adding NPU support

On Ascend NPU, the SDPA path works automatically through `torch_npu`. `lvsa/device.py` detects NPU and routes memory probes through `torch.npu.*`. FlashInfer is CUDA-only and won't run on NPU.

No NPU-specific custom kernels ship in v1.0 — only the SDPA path is exercised on NPU.

## Common first-run gotchas

| Symptom | Fix |
|---|---|
| "ModuleNotFoundError: lvsa" | `pip install -e .` from repo root |
| FlashInfer JIT fails ("Could not find nvcc") | Install CUDA toolkit (nvcc) **or** drop `--flashinfer` |
| Generated mp4 missing in Docker | Pass relative paths; bind-mount the repo; see `lvsa-troubleshooting` skill |
| `attended_per_frame=N/T` shows N=T at 4× | `LVSA_REFERENCE_LATENT_FRAMES` is too high for the model — verify against the table above |

See [`lvsa-troubleshooting`](../lvsa-troubleshooting/SKILL.md) for the full failure-mode catalog.
