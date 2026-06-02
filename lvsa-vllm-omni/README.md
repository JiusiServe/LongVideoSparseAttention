# LVSA vllm-omni Plugin

Drop-in sparse attention backend for [vllm-omni](https://github.com/vllm-project/vllm), enabling LVSA-accelerated video generation served via the OpenAI-compatible API.

Supported models:

| Model | Integration path | Status |
|-------|-----------------|--------|
| Wan 2.1 / 2.2 | `LVSABackend` (attention backend) | Stable |
| HunyuanVideo 1.5 | `LVSABackend` (attention backend) | Stable |

---

## Installation

### From source (alongside the main `lvsa` package)

```bash
cd LongVideoSparseAttention
pip install -e . -e lvsa-vllm-omni/
```

### Docker (recommended)

```bash
cd LongVideoSparseAttention

# Build (all PyPI wheels, no source compilation)
docker build -f lvsa-vllm-omni/Dockerfile.128 -t lvsa-vllm-omni .

# Serve Wan
docker run --gpus all \
  -v /path/to/models:/models \
  -p 8100:8100 \
  --ipc=host \
  lvsa-vllm-omni \
  python -m lvsa_vllm_omni.serve /models/Wan2.1-T2V-1.3B-Diffusers --port 8100 --dtype bfloat16

# Serve HunyuanVideo
docker run --gpus all \
  -v /path/to/models:/models \
  -p 8100:8100 \
  --ipc=host \
  lvsa-vllm-omni \
  python -m lvsa_vllm_omni.serve /models/HunyuanVideo-1.5-Diffusers-480p_t2v --port 8100 --dtype bfloat16
```

### Dependencies

- `lvsa>=0.1.0` (the core library)
- `torch>=2.1.0`
- `vllm==0.22.0` (PyPI) + `vllm-omni==0.22.0rc1` (built from git; **not** on PyPI).
  Versions intentionally differ — vllm-omni 0.22.0rc1 rebases onto vllm 0.22.0.
  For the older pip-installable pair (`0.18.0`/`0.18.0`), use the `release/v0.18.x` branch.
- Optional: `flashinfer-python>=0.2` (for FlashInfer LVSA backend)
- `flash_attn` is **not required** — torch SDPA dispatches to Flash Attention v2 internally

---

## Usage

### Option 1: Serve wrapper (simplest)

```bash
python -m lvsa_vllm_omni.serve /models/Wan2.1-T2V-1.3B-Diffusers --port 8100
```

This registers the LVSA backend and launches `vllm serve --omni` in one step. Works for both Wan and HunyuanVideo.

### Option 2: Explicit per-role AttentionConfig

```bash
vllm serve /models/Wan2.1-T2V-1.3B-Diffusers --omni --port 8100 \
    --diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'
```

vllm-omni 0.22 selects the attention backend per role via `AttentionConfig`
(the `DIFFUSION_ATTENTION_BACKEND` env var was removed). This requires that
`register_lvsa_backend()` has been called before model loading — the entry
point and Docker image handle that automatically (see
[Backend Registration](#backend-registration)). The serve wrapper in Option 1
injects this flag for you.

### Sending requests

The API uses multipart form-data:

```bash
# Submit generation job
curl -X POST http://localhost:8100/v1/videos \
  -F "prompt=A dog running in the forest." \
  -F "size=832x480" \
  -F "num_frames=81" \
  -F "num_inference_steps=20" \
  -F "seed=42"

# Poll for completion
curl http://localhost:8100/v1/videos/{video_id}

# Download result
curl http://localhost:8100/v1/videos/{video_id}/content -o output.mp4
```

---

## Configuration

All LVSA parameters are set via environment variables (`LVSA_*`) or a JSON config string (`LVSA_CONFIG`).

### Sparsity pattern (all read by `LVSAConfig.from_env()`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LVSA_WINDOW_SIZE` | 12 | Half-width of sliding window (video frames) |
| `LVSA_N_FIRST_FRAMES` | 4 | Leading frames always in global context |
| `LVSA_KEY_FRAME_INTERVAL` | 16 | Periodic keyframe interval |
| `LVSA_AUTO_KEYFRAMES` | true | Auto-compute keyframe interval from `LVSA_REFERENCE_LATENT_FRAMES` |
| `LVSA_ROTATE_KEYFRAMES` | false | Shift keyframe grid each step (anti-looping) |
| `LVSA_SPARSITY_SCALE` | 1.0 | Multiplier on per-query attention budget. `<1` = more sparse, `>1` = less sparse. Quality/speed knob. |
| `LVSA_REFERENCE_LATENT_FRAMES` | 21 | Model's training-horizon latent frame count. **Set per model**: Wan=21, HunyuanVideo=33, CogVideoX=13. |

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `LVSA_BACKEND` | sdpa | Attention kernel: `sdpa` or `flashinfer` |
| `LVSA_CFG_PASSES` | 2 | Forward passes per denoising step. Classifier-free guidance runs 2 (cond + uncond). Set to `1` if you run with `guidance_scale=1` (no CFG). The internal step counter increments once per `n_blocks × cfg_passes` attention forwards. |
| `LVSA_N_BLOCKS` | auto | Override transformer block count. Auto-calibrated on first repeated `layer_id`; set explicitly only if calibration misfires. |

### Geometry detection (read by `candidate_patches_per_frame()`, not `LVSAConfig`)

The plugin must infer `seq_len = T_lat × P + enc_tokens` from the raw token tensor. Default `P=1560` covers Wan/HunyuanVideo at 480×832; override for non-default resolutions or custom models.

| Variable | Default | Description |
|----------|---------|-------------|
| `LVSA_PATCHES_PER_FRAME` | — | Explicit override: comma-separated list of candidate `P` values (e.g. `1560,2080`). Highest priority — disables resolution-derivation below. |
| `LVSA_VIDEO_HEIGHT` | — | Frame height in pixels. Used to derive `P` if `LVSA_PATCHES_PER_FRAME` is unset. |
| `LVSA_VIDEO_WIDTH` | — | Frame width in pixels. |
| `LVSA_VAE_SPATIAL_FACTOR` | 8 | VAE spatial downsample (used with `LVSA_VIDEO_HEIGHT`/`WIDTH` to compute `P`). |
| `LVSA_PATCH_SIZE` | 2 | Transformer patch size (used with VAE spatial factor and resolution to compute `P`). |
| `LVSA_VAE_TEMPORAL_FACTOR` | 4 | VAE temporal compression factor (converts video↔latent frame counts). |
| `LVSA_TOTAL_LATENT_FRAMES` | auto | Override latent frame count when geometry detection misfires. |

### Per-model hooks

| Variable | Default | Description |
|----------|---------|-------------|
| `LVSA_HUNYUAN_HOOK` | false | Install HunyuanVideo LVSA hook at class level |
| `LVSA_WAN_HOOK` | false | Install Wan LVSA hook (intercepts before `_sp_plan`). **Required for Wan**. |

### Distributed serving

| Variable | Default | Description |
|----------|---------|-------------|

### Diagnostics

| Variable | Default | Description |
|----------|---------|-------------|
| `LVSA_WARN_FALLBACK` | true | Print one-line warnings on silent dense fallback (recommended on) |
| `LVSA_MEM_LOG` | 0 | Per-step device memory logging — emits `[LVSA-MEM] step=N alloc=… reserved=… peak=…` once per denoising step. Device-agnostic (CUDA + Ascend NPU). Wired in all three plugin paths (`attention_impl`, `hunyuan_hook`, `wan_hook`). |
| `LVSA_STEP_TIME_LOG` | 0 | Per-step wall-clock log — emits `[LVSA-TIME] step=N dt=…s` for the step that just completed. Same coverage as `LVSA_MEM_LOG`. |
| `LVSA_MASK_LOG` | — | Per-step compact attention-mask dump (G=global, W=window, X=both, .=skip). Values: `1` every step, `once` first step only, `N` step N only, `N-M` inclusive range, `N,M,K` specific steps. Useful to confirm the sparsity pattern at a given denoising step. |
| `LVSA_N_BLOCKS` | auto | Override the expected attention-block count per step. Used by `step_tracker` to detect step boundaries; auto-detected on the first step normally. |
| `LVSA_CONFIG` | — | JSON string overriding all `LVSA_*` env vars above |

Example with FlashInfer and custom window:

```bash
LVSA_BACKEND=flashinfer LVSA_WINDOW_SIZE=8 \
  python -m lvsa_vllm_omni.serve /models/Wan2.1-T2V-1.3B-Diffusers --port 8100
```

---

## Benchmarks

Hardware: NVIDIA A100 80GB PCIe, bfloat16.

### Single-GPU: LVSA-FlashInfer vs vllm-omni default (single layer)

| Model | 1x | 1.5x | 2x | 3x |
|-------|-----|------|-----|-----|
| **Wan 2.1 1.3B** | 0.87x | **1.69x** | **2.29x** | **4.98x** |
| **Wan 2.1 14B** | 0.86x | **1.66x** | **2.13x** | **4.97x** |
| **HunyuanVideo 1.5** | **1.27x** | **1.80x** | **3.08x** | **4.44x** |

---

## Project Structure

```
lvsa-vllm-omni/
├── lvsa_vllm_omni/             # Package
│   ├── backend.py              # LVSABackend — AttentionBackend interface
│   ├── attention_impl.py       # LVSAAttentionImpl — LVSA dispatch + dense fallback
│   ├── config.py               # LVSAConfig — env var / JSON config parsing
│   ├── global_kv.py            # Global K/V extraction by frame indexing
│   ├── step_tracker.py         # Thread-local denoising step counter
│   ├── register.py             # Backend registration + per-model hooks
│   ├── serve.py                # Wrapper entry point for vllm serve --omni
│   └── hunyuan_hook.py         # Monkey-patch for HunyuanVideo joint attention
├── tests/                      # Unit tests (no vllm-omni dependency needed)
├── scripts/                    # Benchmarking, generation, and e2e test scripts
├── Dockerfile                  # PyPI wheels, no source compilation
├── Dockerfile.128              # Same as Dockerfile (legacy name)
├── pyproject.toml
└── README.md
```

**How it works:**

1. `register_lvsa_backend()` adds `"LVSA"` to vllm-omni's `DiffusionAttentionBackendEnum`
2. vllm-omni instantiates `LVSAAttentionImpl` for each attention layer
3. On each forward pass, the impl detects video geometry (latent frames, patches-per-frame)
4. Cross-attention and warmup runs fall back to dense SDPA automatically
5. The `BlockCountStepTracker` auto-calibrates by counting attention calls per step
6. Ring SP / Ulysses SP fall through to vllm-omni's default parallel attention path — LVSA's per-rank LVSA pattern is applied only when the full sequence is present on a single rank.

---

## Backend Registration

LVSA registers as a `vllm_omni.general_plugins` entry point:

```toml
[project.entry-points."vllm_omni.general_plugins"]
lvsa = "lvsa_vllm_omni.register:register_lvsa_backend"
```

vllm-omni calls `load_omni_general_plugins()` in the main process, engine, and every diffusion worker, so this single declaration registers LVSA everywhere. No manual setup is needed beyond `pip install -e lvsa-vllm-omni/`.

`register_lvsa_backend()` also triggers the optional monkey-patches (each env-var guarded and no-op otherwise):
- `maybe_install_hunyuan_hook()` — dual-stream LVSA on HunyuanVideo (activated by `LVSA_HUNYUAN_HOOK=1`)
- `maybe_install_wan_hook()` — LVSA on Wan (activated by `LVSA_WAN_HOOK=1`, needed because Wan pre-shards sequences via `_sp_plan`)

---

## Known Limitations

- **TP=3 not supported**: vllm-omni requires hidden dimensions divisible by TP size. HunyuanVideo's dimensions are not divisible by 3.
- **Ulysses SP broken for HunyuanVideo**: attention mask shape mismatch when using `ulysses_degree > 1`. Use `ring_degree` instead.
- **Ring SP requires `LVSA_TOTAL_LATENT_FRAMES`**: must be set via env var for the ring processor to activate. Auto-detection is not yet supported.
- **FlashInfer compact buffers**: the `LVSAAttentionImpl` FlashInfer path uses extra memory for compact KV buffers, which may cause OOM at high frame counts with TP=2.

## CFG and step-counter semantics

The plugin's internal step counter ticks once per **denoising step**, computed from total
attention-forward calls divided by `n_blocks × cfg_passes`. It feeds the rotating-keyframes
offset in LVSA.

If your pipeline runs CFG with `guidance_scale > 1` (two forward passes per step — cond
+ uncond), keep `LVSA_CFG_PASSES=2` (the default). If you run with `guidance_scale == 1`
(single forward), set `LVSA_CFG_PASSES=1`. Mismatching this value with your pipeline's
CFG behavior misaligns the rotation phase but does not affect correctness.

Common configurations:

| Pipeline | guidance | `LVSA_CFG_PASSES` | `LVSA_TOTAL_STEPS` |
|---|---|---|---|
| Wan/HunyuanVideo with CFG (default) | > 1 | 2 (default) | match `num_inference_steps` |
| Anything with CFG short-circuited | == 1 | 1 | match `num_inference_steps` |
| `LVSA_CFG_PASSES=k` for k forwards/step | — | k | match `num_inference_steps` |

To verify the counter is calibrated correctly, look for this log line during the first
denoising step:

```
[LVSA] Step counter auto-calibrated: n_blocks=30 cfg_passes=2
```

`n_blocks` should match your model's transformer block count (30 for Wan 1.3B, 40 for
HunyuanVideo 1.5). `cfg_passes` should match what you set via `LVSA_CFG_PASSES`.

---

## Tests

```bash
cd lvsa-vllm-omni
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests cover config parsing, global K/V extraction, attention routing (self vs cross), step tracking, and full pipeline simulation (no vllm-omni dependency required).
