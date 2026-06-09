# Examples

Standalone generation scripts for each supported model + a vllm-omni serving recipe.

| File | Purpose |
|---|---|
| [`wan_generate.py`](wan_generate.py) | Wan 2.1 / 2.2 single- or multi-GPU generation |
| [`hunyuan_generate.py`](hunyuan_generate.py) | HunyuanVideo 1.5 single- or multi-GPU generation |
| [`cosmos_generate.py`](cosmos_generate.py) | Cosmos 3.0 single-GPU generation (experimental; needs diffusers main) |
| [`cogvideox_generate.py`](cogvideox_generate.py) | CogVideoX 5B (experimental — correctness only) |
| [`vllm_omni_serve.sh`](vllm_omni_serve.sh) | Minimal vllm-omni serving wrapper |

All scripts use the device helpers in [`lvsa/device.py`](../lvsa/device.py), so they run unchanged on CUDA and Ascend NPU (NPU uses the SDPA path; FlashInfer requires CUDA).

---

## Standalone scripts

### `wan_generate.py` — Wan 2.1 / 2.2

```bash
# Single GPU, 1× horizon (training reference), fully-dense via LVSA path
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 81 \
    --lvsa --auto-keyframes \
    --output-name dog.mp4

# Single GPU, 4× horizon with FlashInfer + rotating keyframes
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 321 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name dog_4x.mp4

# Multi-GPU (Ulysses-style context parallel), 6× horizon
torchrun --nproc_per_node=2 examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 481 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name dog_6x.mp4
```

Key flags:
- `--lvsa` — enable LVSA sparse attention (off by default; dense path otherwise)
- `--flashinfer` — use FlashInfer backend (CUDA only, fastest at T_lat ≥ 49)
- `--auto-keyframes` — auto-derive keyframe interval from frame count
- `--rotate-keyframes` — shift keyframe grid each denoising step (recommended at extension)
- `--sparsity-scale` — multiplier on the attention budget (default 1.0; 0.5 = aggressive)
- `--riflex --riflex-s 2.0` — compose with RIFLEx RoPE rescaling

Full `--help` output: `python examples/wan_generate.py --help`.

### `hunyuan_generate.py` — HunyuanVideo 1.5

```bash
# Training reference (129 frames)
python examples/hunyuan_generate.py \
    --model /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "Ocean waves crashing on a rocky coastline at sunset." \
    --num-frames 129 \
    --lvsa --flashinfer --auto-keyframes \
    --output-name ocean_1x.mp4

# 2× horizon (257 frames) — Dense OOMs on 80GB, LVSA fits
python examples/hunyuan_generate.py \
    --model /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "Ocean waves crashing on a rocky coastline at sunset." \
    --num-frames 257 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-latent --output ocean_2x
```

The `--output-latent` flag writes a `.pt` latent tensor (skipping the VAE decode), useful at sequence lengths where the VAE itself OOMs. Decode offline on a higher-memory GPU.

### `cosmos_generate.py` — Cosmos 3.0 (experimental)

```bash
# 1× horizon (189 frames @ 720p, the native reference). At T_lat=48 == ref,
# LVSA runs the dense regime (identical output) — sanity/baseline.
python examples/cosmos_generate.py \
    --model /path/to/Cosmos3-Nano \
    --prompt "A dog running in the forest." \
    --num-frames 189 --height 720 --width 1280 --steps 35 \
    --lvsa --output-name cosmos_1x

# 2× horizon (317 frames) — sparse attention engages (T_lat=80 > ref=48)
python examples/cosmos_generate.py \
    --model /path/to/Cosmos3-Nano \
    --prompt "A dog running in the forest." \
    --num-frames 317 --height 720 --width 1280 --steps 35 \
    --lvsa --output-name cosmos_2x
```

**Requirements / notes:**
- Needs **diffusers main** (`>=0.39.0.dev0`) for `Cosmos3OmniPipeline`. Standard-release diffusers (used by the other models) stops at Cosmos 2.5.
- Cosmos is **separate-stream**: the diffusers `Cosmos3AttnProcessor` runs the text/VLM `und` tokens as causal self-attention and the video `gen` tokens as full attention over `cat([k_und, k_gen])`. LVSA wraps **only the gen pathway** (window gen↔gen + keyframes, all `und` kept global) and leaves the `und` causal path byte-identical. It engages via a **processor swap** (`lvsa/cosmos3.py::install_cosmos3_lvsa`), not the adapter ABC.
- MVP scope: **single-GPU** (no `torchrun`), **SDPA** backend, fixed keyframes (no `--rotate-keyframes`). FlashInfer and Ulysses CP are follow-ups.
- `reference_latent_frames=48` (189-frame native horizon) is the default; override with `--sparsity-scale < 1` for more sparsity below the cap.
- The script passes `enable_safety_checker=False` to `from_pretrained` so the pipeline doesn't construct `CosmosSafetyChecker` (which needs the external `cosmos_guardrail` package).
- Status: correctness-validated (CPU 1×==dense equivalence test + 1-step GPU smoke across dense / 1× / 2×). SDPA shows no speedup over dense up to Cosmos's ~400-frame single-shot cap — matching the plugin findings; the speedup lever is FlashInfer (planned). Output dir defaults to `out/adhoc/` (gitignored).

### `cogvideox_generate.py` — CogVideoX 5B (experimental)

```bash
python examples/cogvideox_generate.py \
    --model /path/to/CogVideoX-5b \
    --prompt "A dog running in the forest." \
    --num-frames 49 \
    --lvsa \
    --output-name cog.mp4
```

**Note**: CogVideoX uses joint text-video attention with shared QKV. LVSA produces correct output but does not yield wall-time speedup on this architecture. Included for completeness; not a recommended use case for v1.0.

---

## vllm-omni serving

For production serving via the OpenAI-compatible API:

```bash
# Wan 2.x on port 8100
examples/vllm_omni_serve.sh wan /path/to/Wan2.1-T2V-1.3B-Diffusers

# HunyuanVideo 1.5
examples/vllm_omni_serve.sh hunyuan /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v

# Custom port / dtype
PORT=8200 DTYPE=float16 examples/vllm_omni_serve.sh wan /path/to/Wan
```

The wrapper sets the required `LVSA_*` env vars (correct `LVSA_REFERENCE_LATENT_FRAMES` per model, `LVSA_WAN_HOOK=1` for Wan, etc.) and runs `python -m lvsa_vllm_omni.serve`. See [`../docs/VLLM_OMNI_INTEGRATION.md`](../docs/VLLM_OMNI_INTEGRATION.md) for the full configuration reference.

---

## Hardware notes

| Hardware | Backend | Distributed |
|---|---|---|
| CUDA (single A100 80GB) | SDPA or FlashInfer | n/a |
| CUDA multi-GPU | SDPA or FlashInfer | `torchrun --nproc_per_node=N` (Ulysses) |
| Ascend NPU | SDPA only (FlashInfer is CUDA-only) | `torchrun` with `hccl` backend (auto via `lvsa/device.py`) |
| CPU | SDPA (for testing, no real generation) | n/a |

Verify the LVSA path is engaged by grepping the log for `[LVSA]` lines after the run. See [`../docs/troubleshooting.md`](../docs/troubleshooting.md) if you see `[LVSA-FALLBACK]` warnings.
