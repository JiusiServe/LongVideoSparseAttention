# Troubleshooting

LVSA's failure modes are mostly **silent**: the run completes, an mp4 is produced, and only careful inspection reveals that the sparse path didn't engage or the geometry was wrong. This page covers the most common issues.

## Symptom 1 — No speedup (LVSA matches dense wall-clock)

**Root cause**: the LVSA backend silently fell back to dense at every block. Most often because geometry detection failed.

### Diagnosis

**First, confirm engagement.** On the first geometry-matched generation forward, the hooks print a one-time, default-on confirmation:

```
[LVSA] engaged (wan): T_lat=21 P=1560 seq_len=32760 kfi=1 W=3 |G|=21 -> DENSE (T_lat <= ref)
[LVSA] engaged (hunyuan): T_lat=49 P=1560 seq_len=76440 kfi=2 W=3 |G|=25 -> SPARSE
```

This distinguishes three states that otherwise look identical:
- **`-> SPARSE`** (kfi > 1, horizon above the training reference) — LVSA is doing real work.
- **`-> DENSE (T_lat <= ref)`** — LVSA engaged but is intentionally dense at/below the reference horizon (kfi=1). Expect no speedup here; that's correct, not a bug.
- **No `[LVSA] engaged` line at all** — the sparse path never ran for generation. The line is deduped per generation, so it appears once; the `[LVSA-FALLBACK]` you see at the small warmup `seq_len` (e.g. 1024) is the dummy run, not generation.

For per-step sparsity detail (keyframe grid, attended budget), set `LVSA_MASK_LOG=once` (or `=1`).

Then look for `[LVSA-FALLBACK]` warnings during the *generation* steps (not warmup):

```
[LVSA-FALLBACK] origin=forward_cuda reason=geometry_detect seq_len=25740 known_ppf=[1560]
[LVSA-FALLBACK] origin=forward_cuda reason=no_t_lat seq_len=…
```

If you see no `[LVSA]` lines at all, the backend wasn't selected — for vllm-omni 0.22 check that `--diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'` is passed (the `python -m lvsa_vllm_omni.serve` wrapper injects it; the `DIFFUSION_ATTENTION_BACKEND` env var was removed in 0.22).

If you see backend selection but no per-block engagement on Wan, check `LVSA_WAN_HOOK=1` (required for Wan; without it Wan's `_sp_plan` pre-shards the sequence and geometry detection fails silently).

### Fix

Set the env vars per [quickstart.md](quickstart.md). Always include `LVSA_WARN_FALLBACK=1` (default on) so silent fallbacks become visible.

For non-default resolutions, also set the geometry env vars: `LVSA_PATCHES_PER_FRAME`, `LVSA_VIDEO_HEIGHT`, `LVSA_VIDEO_WIDTH`, `LVSA_VAE_SPATIAL_FACTOR`, `LVSA_PATCH_SIZE`, `LVSA_VAE_TEMPORAL_FACTOR`.

## Symptom 2 — OOM at long sequences (T_lat ≥ 65 on 80 GB GPU)

**Root cause**: the **VAE decode** (not attention) is what blew up memory. LVSA reduces attention, but the VAE then has to decode all those latent frames back to pixels at once. On HunyuanVideo 1.5 (480p) this OOMs at ~65+ video frames on an 80 GB GPU — the dense VAE decode is the ceiling, not LVSA (the sparse attention forward completes cleanly first).

### Diagnosis

The OOM trace points at `vae.decode` / the VAE decoder's `up_block` (e.g. `pipeline_hunyuan_video_1_5.py … self.vae.decode(…)`). The attention forward and the whole denoising loop will have completed.

### Fix (vLLM-Omni)

**Enable VAE tiling + slicing** — they decode the latents in chunks and are effectively lossless. With them, HunyuanVideo 1.5 runs at **129 frames (reference length) on a single 80 GB A100** with LVSA. This is the default in `offline_lvsa.py` (`--family hunyuan`) and `serve_hunyuan.sh`; if you call the Python API directly:

```python
Omni(
    model="/models/HunyuanVideo-1.5-Diffusers-480p_t2v",
    diffusion_attention_config={"per_role": {"self": {"backend": "LVSA"}}},
    vae_use_tiling=True,
    vae_use_slicing=True,
)
```

On the serve CLI, pass `--vae-use-tiling --vae-use-slicing` (the `serve_hunyuan.sh` wrapper adds them by default). For very long horizons that still exceed memory, add `enable_cpu_offload=True`.

### Fix (standalone, alternative)

Use `--output-latent` (standalone) or `output_type="latent"` (vLLM-Omni Python API) to skip VAE decode and save the denoised latent tensor to a `.pt` file. Decode it offline on a higher-memory GPU.

```bash
python examples/hunyuan_generate.py \
    --model /models/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --num-frames 257 --output-dir benchmarks --output-name out_2x \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-latent
```

The mp4 path becomes a `.pt` path; load with:

```python
import torch
data = torch.load("out_2x.pt")          # {"latent": tensor, "num_frames": 257, ...}
# Then decode with the pipeline's VAE on a higher-memory GPU
```

## Symptom 3 — Output mp4 missing after Docker run

**Root cause**: an absolute host path was passed to `--output`, which the container interprets relative to its own filesystem (not the bind-mount), so the mp4 lands inside the container and disappears with `--rm`.

### Diagnosis

The run log says `[export] /home/me/.../out.mp4 (1.4s)` but `ls /home/me/.../out.mp4` on the host returns nothing.

### Fix

Pass paths **relative to the container's working directory** (`/workdir/code` in the LVSA Docker image), and bind-mount the host repo at that path:

```bash
# WRONG — path resolves inside container, lost on --rm
docker run ... lvsa-vllm-omni:latest \
    python examples/hunyuan_generate.py --output-name /home/gael/results/out.mp4

# RIGHT — relative path, lands on host via the bind-mount
docker run ... -v $(pwd):/workdir/code -w /workdir/code ... \
    lvsa-vllm-omni:latest \
    python examples/hunyuan_generate.py --output-name benchmarks/results/out.mp4
```

Always include `&& chown -R $(id -u):$(id -g) <output_dir>` at the end of the inner command so files written as root inside the container are readable on the host.

## Symptom 4 — Quality regression at training reference

**Root cause**: `LVSA_REFERENCE_LATENT_FRAMES` is wrong for your model. The auto-keyframe scheduler computed sparsity using the wrong budget, so even at training horizon you're getting partial attention coverage instead of full attention.

### Diagnosis

Check the run log:

```
[LVSA] reference_latent_frames=21  target_latent_frames=33  extension_ratio=1.57x
```

If you're running HunyuanVideo at 129 frames and you see `reference_latent_frames=21` and `extension_ratio` > 1.0 (when it should be exactly 1.0), the scheduler is using Wan's default reference. The result: per-query attended budget shrinks to ~21 instead of 33, producing ~36% accidental sparsity at training reference.

### Fix

```bash
# HunyuanVideo
LVSA_REFERENCE_LATENT_FRAMES=33

# Wan 2.x
LVSA_REFERENCE_LATENT_FRAMES=21

# CogVideoX
LVSA_REFERENCE_LATENT_FRAMES=13
```

This is the single most common LVSA configuration mistake. Always set it explicitly.

## Symptom 5 — `dynamic_quality` drops 5+ points but the user wanted quality parity

**Root cause**: this isn't a bug — it's the documented trade-off of any sparse-attention scheme. Long-range attention pairs that contribute to large-scale motion coherence are skipped, so motion-heavy prompts can lose ~5 points on VQeval `dynamic_quality` and ~0.02 points on VBench `motion_smoothness`.

### Diagnosis

Compare per-dimension VQeval numbers between dense and LVSA on the same prompt set. If `dynamic_quality` is the only dimension dropping while `loop_quality` and `text_alignment` improve, this is the trade-off.

### Fix

Three options, in order of cost:

1. **Use `sparsity_scale = 1.0`** (default). At T_lat ≤ reference this gives fully-dense attention via the LVSA path; you keep the implementation-overhead-bypass speedup (~1.5–2×) without any sparsity-driven quality cost.
2. **Increase `LVSA_WINDOW_SIZE`** from 12 to 16 video frames (4 → 5 latents). Larger window means more long-range mixing inside each query's attended set. Costs ~10% wall time.
3. **Switch to dense for motion-heavy prompts**. The `dynamic_quality` regression is highly prompt-dependent — most prompts see <2-point drops; only fast-motion prompts hit the 5-point regression.

## Symptom 6 — Very large `loop_quality` swings between methods

VQeval's `loop_quality` metric is sensitive to whether the video exhibits static / repeating content. At extended horizons, dense attention is known to drift toward static or near-periodic output (see the discussion in the paper's quality section), which depresses its loop_quality. LVSA's rotating-keyframe pattern keeps the attention set moving across denoising steps, which tends to keep loop_quality higher at extension.

Practically: large dense-vs-LVSA loop_quality gaps at extrapolation are expected and reflect the underlying failure mode of dense at those horizons, not a measurement artifact. Report both numbers side by side rather than only the delta.

## Symptom 7 — Graduated schedule env vars (`LVSA_SCHEDULE_*`) don't seem to do anything

**Root cause**: `LVSA_SCHEDULE_START` / `LVSA_SCHEDULE_END` are **soft-deprecated** in v1.0. The recommended runtime knob is `LVSA_SPARSITY_SCALE`. The schedule vars still exist for backwards compatibility (default `0`) but produce no behavior at default settings.

### Fix

Use `sparsity_scale` instead. See [`tuning.md`](tuning.md) for the supported range and trade-offs.

## Diagnostics one-liner

When something looks off, dump the LVSA-relevant log lines:

```bash
grep -E "\[LVSA" run.log | head -20
```

If the output is empty or shows only fallback warnings, walk through Symptoms 1–4 above.

## Logs to capture when filing a bug report

- Full stdout/stderr of the run
- Output of `grep -E "\[LVSA" run.log` (LVSA-specific lines)
- Output of `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` mid-run (or peak), or `npu-smi info` for Ascend
- Generated mp4 / .pt file size
- Exact `docker run` command + env vars
- LVSA version: `git rev-parse HEAD` in the repo
