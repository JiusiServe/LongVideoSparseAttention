# LVSA × vllm-omni — Examples

Two ways to use the LVSA plugin with vllm-omni:

| Mode | Files | Use case |
|---|---|---|
| **Online** (HTTP server) | `serve_wan.sh`, `serve_hunyuan.sh`, `online_client.py`, `online_curl.sh` | Production: serve generation behind an OpenAI-compatible API |
| **Offline** (in-process Python) | `offline_wan.py`, `offline_hunyuan.py` | Scripting / batch eval / quick smoke tests |

Both modes engage LVSA through `LVSA_*` environment variables — see [`../README.md`](../README.md) for the full env-var reference.

## Prerequisites

`vllm==0.18.0` pins `torch==2.10`, which is incompatible with the torch the
standalone LVSA engine uses (`2.12`). **Use a separate venv for vllm-omni
work** so it doesn't break the standalone engine in your main `.venv`.

```bash
# From the LVSA repo root:
uv venv .venv-vllm --python 3.12        # separate venv, dedicated to vllm-omni
source .venv-vllm/bin/activate

uv pip install -e .                                # core lvsa
uv pip install -e lvsa-vllm-omni/                  # this plugin
uv pip install "vllm==0.18.0" "vllm-omni==0.18.0"  # validated pair
```

> **Two-venv rule**: keep `.venv` for standalone (`examples/wan_generate.py`,
> CPU tests, benchmark scripts) and `.venv-vllm` for vllm-omni serving + the
> scripts in this folder. The `.venv-vllm` torch is older (2.10), so
> standalone GPU work that depends on torch 2.12 features won't run there.
>
> **Version pair**: `vllm` and `vllm-omni` minor versions must match.
> Installing the latest of each (`pip install vllm vllm-omni`) will fetch
> `0.21.x` and `0.20.x` respectively, which mismatch and emit a runtime
> warning. The LVSA plugin is validated against the `0.18.0` pair, also
> pinned in [`../Dockerfile`](../Dockerfile).

## Online — server + HTTP client

### 1. Start the server

```bash
# Wan 2.x
examples/serve_wan.sh /path/to/Wan2.1-T2V-1.3B-Diffusers
#  → http://localhost:8098

# HunyuanVideo 1.5
examples/serve_hunyuan.sh /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v
```

Override env vars as needed:

```bash
PORT=8200 TP=2 \
LVSA_SPARSITY_SCALE=0.5 \
examples/serve_wan.sh /path/to/Wan2.1-T2V-1.3B-Diffusers
```

The wrapper sets the right `LVSA_*` env vars per model family before forwarding to `python -m lvsa_vllm_omni.serve`.

### 2. Submit a generation request

**Python client** (recommended):

```bash
python examples/online_client.py \
    --host localhost:8098 \
    --prompt "A dog running in the forest." \
    --num-frames 81 --size 832x480 --fps 16 \
    --steps 40 --guidance 4.0 --seed 42 \
    --output dog.mp4
```

**Or bash + curl + jq**:

```bash
PROMPT="A dog running in the forest." \
NUM_FRAMES=81 \
OUTPUT=dog.mp4 \
examples/online_curl.sh
```

Both submit the request to `POST /v1/videos`, poll the job status, and download the resulting mp4 once `status == "completed"`.

### Request payload reference

vllm-omni accepts form-data (not JSON) on `POST /v1/videos`. Common fields:

| Field | Notes |
|---|---|
| `prompt` (required) | Text description |
| `num_frames` or `seconds` | Generation length |
| `size` | `WIDTHxHEIGHT` (e.g. `832x480`) |
| `fps` | Frame rate (16-24 typical) |
| `num_inference_steps` | Denoising steps (40-50) |
| `guidance_scale` | CFG (Wan 2.2 high-noise stage, or HunyuanVideo single stage) |
| `guidance_scale_2` | Wan 2.2 low-noise stage (omit for Wan 2.1 / HV) |
| `flow_shift` | HunyuanVideo: 5.0 @ 480p, 9.0 @ 720p typical |
| `boundary_ratio` | Wan 2.2 only (~0.875 typical) |
| `seed` | RNG seed |
| `negative_prompt` | Optional |

## Offline — direct Python API

For when you want to script generation without standing up an HTTP server (e.g. batch eval, smoke tests, integration into a larger pipeline).

```bash
# Wan
python examples/offline_wan.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 81 --steps 40 --seed 42 \
    --output-name dog_offline.mp4

# HunyuanVideo
python examples/offline_hunyuan.py \
    --model /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running in the forest." \
    --num-frames 129 --steps 50 --seed 42 \
    --output-name dog_hv_offline.mp4

# Dense baseline (no LVSA)
python examples/offline_wan.py --no-lvsa --model ... --prompt ...

# Aggressive sparsity
python examples/offline_wan.py --sparsity-scale 0.5 --model ... --prompt ...
```

These scripts:
1. Set the LVSA env vars **before** importing `vllm_omni` (so the hooks install correctly).
2. Call `register_lvsa_backend()` to wire LVSA into vllm-omni's backend enum.
3. Instantiate `Omni(...)` and call `generate_one(...)`.
4. Write the resulting frames to `.mp4` via `diffusers.utils.export_to_video`.

## Verifying LVSA engaged

Look for these log lines after either mode:

```
[LVSA-hook] Installed LVSA hook on WanSelfAttention (T_lat=21, sparsity_scale=1.0)
[LVSA-hook] Step counter calibrated: n_blocks=30 cfg_passes=2
[LVSA-MASK] step=0  T_lat=21  W=3  |G|=21  kfi=1
```

If you see `[LVSA-FALLBACK]` warnings instead, see [`../../docs/troubleshooting.md`](../../docs/troubleshooting.md).

## Multi-GPU

For tensor-parallel (Ulysses-style) generation across GPUs, set the `TP` env var on the server scripts or `--tensor-parallel-size` on the offline scripts:

```bash
TP=2 examples/serve_wan.sh /path/to/Wan2.1-T2V-14B-Diffusers
# or
python examples/offline_wan.py --tensor-parallel-size 2 --model ... --prompt ...
```

**Constraint**: `seq_len = T_lat × patches_per_frame` must be divisible by `tensor_parallel_size`.

## Common gotchas

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: vllm` | `pip install vllm` separately — vllm-omni does not pull it as a hard dep |
| `[LVSA-FALLBACK] reason=geometry_detect` | Set `LVSA_PATCHES_PER_FRAME` or `LVSA_VIDEO_HEIGHT/WIDTH` for non-default resolutions |
| Server stuck in `queued` state | Worker process crashed — check stderr of the serve command |
| `[LVSA] Warning: LVSA_TOTAL_LATENT_FRAMES not set` | The hook needs to know the latent-frame count at install time. Set `LVSA_TOTAL_LATENT_FRAMES=(num_frames-1)//4 + 1` |
