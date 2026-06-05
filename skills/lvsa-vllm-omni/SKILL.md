---
name: lvsa-vllm-omni
description: Configure and run the LVSA vllm-omni serving plugin. Use when enabling LVSA in vllm-omni, choosing per-model env vars (LVSA_WAN_HOOK / LVSA_REFERENCE_LATENT_FRAMES), debugging silent fallbacks via [LVSA-FALLBACK] warnings, setting geometry overrides for non-default resolutions, or composing with Ulysses CP.
---

# LVSA vllm-omni plugin

## Overview

`lvsa-vllm-omni` is a separate pip package that registers itself as a vllm-omni attention backend. Zero changes required in vllm-omni core — everything plugs in via the plugin entry-point system.

Path: [`lvsa-vllm-omni/`](../../lvsa-vllm-omni/)

## Install

```bash
pip install -e .                  # LVSA core
pip install -e lvsa-vllm-omni/    # The plugin (registers entry-point)
# vllm 0.22.0 is on PyPI; vllm-omni 0.22.0rc1 is a pre-release built from git
# (NOT on PyPI). Versions intentionally differ — vllm-omni 0.22.0rc1 rebases
# onto vllm 0.22.0. Use a separate venv (torch 2.11 / CUDA 13).
pip install "vllm==0.22.0"
pip install --no-build-isolation \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.22.0rc1"
```

After install, the plugin auto-loads when vllm-omni starts. For the older
pip-installable `0.18.0`/`0.18.0` pair, use the `release/v0.18.x` branch.

## Enable for a model

### HunyuanVideo 1.5

```bash
LVSA_HUNYUAN_HOOK=1 \
LVSA_AUTO_KEYFRAMES=1 \
LVSA_REFERENCE_LATENT_FRAMES=33 \
LVSA_ROTATE_KEYFRAMES=1 \
vllm serve --omni --model HunyuanVideo-1.5-Diffusers-480p_t2v \
  --diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'
```

vllm-omni 0.22 selects the backend per role via `--diffusion-attention-config`
(the `DIFFUSION_ATTENTION_BACKEND` env var was removed). `python -m
lvsa_vllm_omni.serve` injects that flag for you.

### Wan 2.x

```bash
LVSA_WAN_HOOK=1 \
LVSA_AUTO_KEYFRAMES=1 \
LVSA_REFERENCE_LATENT_FRAMES=21 \
LVSA_ROTATE_KEYFRAMES=1 \
vllm serve --omni --model Wan2.2-T2V-14B \
  --diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'
```

Wan requires `LVSA_WAN_HOOK=1` explicitly. Without it, Wan's `_sp_plan` pre-shards the sequence and geometry detection fails silently.

### Convenience wrapper

A shell wrapper that sets the right env vars per model family:

```bash
examples/vllm_omni_serve.sh wan      /path/to/Wan2.1-T2V-1.3B-Diffusers
examples/vllm_omni_serve.sh hunyuan  /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v

# Override port / dtype
PORT=8200 DTYPE=float16 examples/vllm_omni_serve.sh wan ...
```

## Environment variables

### Core (always set)

| Var | Default | Purpose |
|---|---|---|
| `--diffusion-attention-config` (CLI flag, not env) | (platform default) | `'{"per_role": {"self": {"backend": "LVSA"}}}'` to engage. The serve wrapper injects it. |
| `LVSA_REFERENCE_LATENT_FRAMES` | `21` | Per-model training horizon. **CRITICAL.** Wan=21, HV=33, Cog=13. |
| `LVSA_AUTO_KEYFRAMES` | `1` | Auto-derive keyframe interval from frame count |
| `LVSA_WAN_HOOK` | `0` | **Required for Wan**. Off for HunyuanVideo. |
| `LVSA_ROTATE_KEYFRAMES` | `0` | Shift keyframe grid each denoising step (recommended at extension) |

### Tuning

| Var | Default | Purpose |
|---|---|---|
| `LVSA_SPARSITY_SCALE` | `1.0` | Multiplier on attention budget. See `lvsa-tuning` skill. |
| `LVSA_WINDOW_SIZE` | `12` | Local window half-width (video frames) |
| `LVSA_N_FIRST_FRAMES` | `4` | Leading global frames |
| `LVSA_KEY_FRAME_INTERVAL` | `16` | Manual keyframe interval (ignored if AUTO_KEYFRAMES) |
| `LVSA_EXPAND_WINDOW` | `1` | Extend local window when globals overlap |

### Geometry (for non-default resolutions)

| Var | Default | Purpose |
|---|---|---|
| `LVSA_PATCHES_PER_FRAME` | `1560` (480×832) | Tokens per latent frame |
| `LVSA_VIDEO_HEIGHT` | (unset) | Use with VAE_SPATIAL_FACTOR + PATCH_SIZE for derivation |
| `LVSA_VIDEO_WIDTH` | (unset) | Same |
| `LVSA_VAE_SPATIAL_FACTOR` | `8` | VAE spatial compression |
| `LVSA_PATCH_SIZE` | `2` | Patchify factor |
| `LVSA_VAE_TEMPORAL_FACTOR` | `4` | VAE temporal compression |

For 720p or larger, set `LVSA_VIDEO_HEIGHT/WIDTH` and the plugin will derive `patches_per_frame` automatically.

### Diagnostics

| Var | Default | Purpose |
|---|---|---|
| `LVSA_BACKEND` | `sdpa` | `sdpa` or `flashinfer` |
| `LVSA_TOTAL_LATENT_FRAMES` | (unset) | Override auto-detected T_lat (rarely needed) |
| `LVSA_MEM_LOG` | `0` | Per-step memory log (`[LVSA-MEM] step=N alloc=… reserved=… peak=…`) |

## Verifying engagement

After startup + first request, look for:

```
[LVSA] Step counter: n_blocks=N (from env)
[LVSA-hook] Installed LVSA hook on HunyuanVideo15Attention ...
[LVSA] Geometry detected: T_lat=33 P=1560 text_tokens=512
[LVSA] reference_latent_frames=33 target_latent_frames=33 extension_ratio=1.00x
```

If you see `[LVSA-FALLBACK] origin=forward_cuda reason=geometry_detect ...`, the plugin couldn't infer the geometry from the seq_len it received. Walk through:

1. Is `seq_len = T_lat * P + enc_tokens` for any P in `candidate_patches_per_frame()`?
2. Is `LVSA_PATCHES_PER_FRAME` set if you're at a non-default resolution?
3. For Wan: is `LVSA_WAN_HOOK=1`?

## Multi-GPU

Standard Ulysses context parallel. Set `--ulysses-degree N` in the vllm-omni CLI; the plugin handles the global K/V gather automatically.

```bash
LVSA_HUNYUAN_HOOK=1 \
LVSA_REFERENCE_LATENT_FRAMES=33 \
LVSA_AUTO_KEYFRAMES=1 \
vllm serve --omni --model HunyuanVideo-1.5 --ulysses-degree 2 \
  --diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'
```

**Constraint**: `seq_len = T_lat × patches_per_frame` must be divisible by `N`.

## Plugin structure

```
lvsa-vllm-omni/
├── lvsa_vllm_omni/
│   ├── __init__.py              # Entry-point: register_lvsa_backend()
│   ├── backend.py               # LVSABackend (vllm attention backend)
│   ├── attention_impl.py        # LVSAAttentionImpl → sparse_windowed_attention()
│   ├── wan_hook.py              # Wan-specific monkey patch
│   ├── hunyuan_hook.py          # HunyuanVideo-specific monkey patch
│   ├── register.py              # Hook auto-install based on env vars
│   ├── config.py                # LVSAConfig dataclass (env var parsing)
│   ├── global_kv.py             # Global K/V gather helpers
│   ├── step_tracker.py          # Per-step state
│   └── _fallback.py             # Silent-fallback warnings
└── tests/                       # CPU-only integration tests (~50 tests)
```

## Common issues

| Symptom | Fix |
|---|---|
| No `[LVSA]` log lines | Check `--diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'` is passed (or use the serve wrapper); for Wan also `LVSA_WAN_HOOK=1` |
| `[LVSA-FALLBACK] reason=geometry_detect` | Set `LVSA_PATCHES_PER_FRAME` (or HEIGHT/WIDTH) for your resolution |
| Quality regression at 1× | `LVSA_REFERENCE_LATENT_FRAMES` wrong for the model |
| No speedup despite engagement | At T_lat ≤ ref, kfi=1 → fully dense. Lower `LVSA_SPARSITY_SCALE` to see real sparsity |

See [`lvsa-troubleshooting`](../lvsa-troubleshooting/SKILL.md) for the full failure-mode catalog and [`docs/VLLM_OMNI_INTEGRATION.md`](../../docs/VLLM_OMNI_INTEGRATION.md) for the architecture details.
