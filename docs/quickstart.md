# Quick Start

## Prerequisites

- LVSA installed (see [`install.md`](install.md))
- A model checkpoint downloaded locally (see model list at the end of `install.md`)
- A CUDA GPU with ≥ 24 GB VRAM (Wan 1.3B at 81 frames) or ≥ 80 GB (anything beyond), or an Ascend NPU with comparable memory

## Goal

Generate your first long-video clip with sparse attention in 3 minutes.

## Path A — Standalone script (single GPU, no serving)

```bash
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 81 \
    --lvsa --auto-keyframes \
    --output-name dog.mp4
```

This runs **with sparse attention** at the model's training horizon (81 frames). On an A100 80 GB this takes ~3 minutes for 50 denoising steps. The output `dog.mp4` will be a 81-frame video at 24 fps.

To extend beyond the training horizon (where sparse attention pays off most):

```bash
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 321 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name dog_4x.mp4
```

This generates 321 frames (4× the training horizon, ~13 seconds at 24 fps) using FlashInfer + rotating keyframes. Expected wall-clock on A100: ~12 minutes (vs. ~27 minutes for dense attention).

### Multi-GPU context parallel

For longer sequences across multiple GPUs, use Ulysses-style context parallelism:

```bash
torchrun --nproc_per_node=2 examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 481 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name dog_6x.mp4
```

**Constraint**: `seq_len = T_lat × patches_per_frame` must be divisible by the GPU count.

### Cosmos 3.0 (experimental, standalone)

Cosmos 3.0 uses a separate code path (a processor swap, not the adapter ABC) and needs **diffusers main** (`>=0.39.0.dev0`):

```bash
python examples/cosmos_generate.py \
    --model /path/to/Cosmos3-Nano \
    --prompt "A dog running in the forest." \
    --num-frames 317 --height 720 --width 1280 --steps 35 \
    --lvsa --output-name cosmos_2x
```

Single-GPU only, SDPA backend, `reference_latent_frames=48` (189-frame native horizon → sparse engages above it). See [`../examples/README.md`](../examples/README.md#cosmos_generatepy--cosmos-30-experimental) for details and the architecture note on the [processor-swap path](architecture.md#when-a-model-doesnt-fit-the-abc--the-processor-swap-path-cosmos-30).

## Path B — vLLM-Omni serving (production deployment)

This path uses the LVSA plugin inside [vLLM-Omni](https://github.com/vllm-project/vllm-omni) to serve generation requests over an OpenAI-compatible API.

### Single GPU

```bash
docker run --rm --gpus '"device=0"' --ipc=host --shm-size=2g \
    -v /path/to/models:/models \
    lvsa-vllm-omni:latest \
    python lvsa-vllm-omni/examples/offline_lvsa.py --family hunyuan \
        --model /models/HunyuanVideo-1.5-Diffusers-480p_t2v \
        --num-frames 129 --steps 50 --guidance 6.0 --flow-shift 5 \
        --prompt "A dog running in the forest." \
        --output-name benchmarks/hv_1x
```

`offline_lvsa.py --family hunyuan` engages LVSA for you: it selects the LVSA
attention **backend** via the per-role `diffusion_attention_config`
( `{"per_role": {"self": {"backend": "LVSA"}}}` ) on the `Omni(...)` call (Wan
uses the same backend path; only Cosmos uses a hook). Under the hood the key
settings are:
- per-role AttentionConfig selecting `LVSA` for the self-attention role
  (vllm-omni 0.22 replaced the `DIFFUSION_ATTENTION_BACKEND` env var)
- `LVSA_AUTO_KEYFRAMES=1` — auto-derive the keyframe interval
- `LVSA_REFERENCE_LATENT_FRAMES=33` — model's training horizon in latent frames (33 for HunyuanVideo, 21 for Wan, 13 for CogVideoX)

For Wan, also set `LVSA_WAN_HOOK=1` and `LVSA_REFERENCE_LATENT_FRAMES=21`.

### Non-standard resolutions

For models running at resolutions other than 480×832, set the geometry override env vars so the plugin computes the correct sparsity pattern:

```
LVSA_PATCHES_PER_FRAME=...     # tokens per latent frame (after VAE + patchify)
LVSA_VIDEO_HEIGHT=...
LVSA_VIDEO_WIDTH=...
LVSA_VAE_SPATIAL_FACTOR=8      # default for Wan/HV
LVSA_PATCH_SIZE=2              # default for Wan/HV
LVSA_VAE_TEMPORAL_FACTOR=4     # default for Wan/HV
```

See [`lvsa-vllm-omni/README.md`](../lvsa-vllm-omni/README.md) for the full env-var reference.

## Verifying LVSA actually engaged

After generation, check the log for:

```
[LVSA] reference_latent_frames=33  target_latent_frames=33  extension_ratio=1.00x
[LVSA] computed key_frame_interval=1 (latent frames)
[LVSA] kfi=1  global_count=33  attended_per_frame=33/33
[LVSA] installed on N blocks  num_patches=1560  total_lat_frames=33  backend=FlashInfer
```

The four lines confirm:
1. Backend selected
2. Reference matches your model
3. Keyframe interval auto-computed
4. Per-query attended set built

If `attended_per_frame=N/T` shows `N == T`, attention is fully dense (T ≤ ref). If `N < T`, you're getting genuine sparse attention.

If you see `[LVSA-FALLBACK]` warnings instead, see [`troubleshooting.md`](troubleshooting.md).

## What to try next

1. **Sweep the speedup curve**: re-run with `--num-frames` set to 161, 241, 321, 481 (Wan) or 65, 129, 193, 257 (HunyuanVideo) to see the dense vs. LVSA gap widen.
2. **Try `sparsity_scale=0.5`** at the training reference (e.g., HunyuanVideo at 129 frames) to see what aggressive sparsity does — see [`tuning.md`](tuning.md).
3. **Compose with RIFLEx**: add `--riflex --riflex-s 2.0` to stack RoPE rescaling on top of sparse attention.
4. **Compare quality** with the bundled [`vqeval`](../vqeval/) subpackage — LVSA's `loop_quality` and `dynamic_quality` metrics tell the dense-vs-LVSA story most clearly.
5. **Reproduce the paper numbers** — see [`benchmarks/README.md`](../benchmarks/README.md).
6. **Add your own model**: see [`architecture.md`](architecture.md) for the adapter pattern.
