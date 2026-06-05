---
name: lvsa-troubleshooting
description: Diagnose LVSA failure modes. Use when LVSA shows no speedup vs Dense, fails to engage (silent fallback), OOMs at long sequences, the output mp4 is missing after a Docker run, quality regresses at training reference, or LVSA_SCHEDULE_* env vars don't seem to do anything.
---

# LVSA Troubleshooting

LVSA's failure modes are mostly **silent**: the run completes, an mp4 is produced, and only careful inspection reveals that the sparse path didn't engage or the geometry was wrong. This skill covers the seven most common issues.

## Diagnostics one-liner

When something looks off, dump the LVSA-relevant log lines:

```bash
grep -E "\[(LVSA|LVSA)" run.log | head -20
```

If the output is empty or shows only fallback warnings, walk through Symptoms 1–4 below.

---

## Symptom 1 — No speedup (LVSA matches Dense wall-clock)

**Root cause**: the LVSA backend silently fell back to dense at every block. Most often because geometry detection failed.

### Diagnosis
```
[LVSA-FALLBACK] origin=forward_cuda reason=geometry_detect seq_len=25740 known_ppf=[1560]
[LVSA-FALLBACK] origin=forward_cuda reason=no_t_lat seq_len=…
```

If you see no `[LVSA]` lines at all, the backend wasn't selected — for vllm-omni 0.22 check that `--diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'` is passed (the `python -m lvsa_vllm_omni.serve` wrapper injects it; the `DIFFUSION_ATTENTION_BACKEND` env var was removed in 0.22). For standalone, check `--lvsa`.

### Fix
For vllm-omni:
- Pass `--diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'` (or use the serve wrapper)
- `LVSA_AUTO_KEYFRAMES=1`
- `LVSA_REFERENCE_LATENT_FRAMES=<21|33|13>`
- For Wan: also `LVSA_WAN_HOOK=1` (without it Wan's `_sp_plan` pre-shards the sequence)

For non-default resolutions (not 480×832), set the geometry env vars:
- `LVSA_PATCHES_PER_FRAME`, `LVSA_VIDEO_HEIGHT`, `LVSA_VIDEO_WIDTH`, `LVSA_VAE_SPATIAL_FACTOR`, `LVSA_PATCH_SIZE`, `LVSA_VAE_TEMPORAL_FACTOR`

---

## Symptom 2 — OOM at long sequences (T_lat ≥ 65 on 80 GB GPU)

**Root cause**: the **VAE decode** (not attention) blew up memory. LVSA reduces attention to ~50% of dense at 2× horizon, but the VAE then has to decode all those latent frames at once.

### Diagnosis
The OOM trace points at `vae.decode` or `unsqueeze` inside the VAE. The attention forward completes cleanly.

### Fix
Use `--output-latent` (standalone) or `output_type="latent"` (vllm-omni Python API) to skip VAE decode and save the denoised latent tensor to a `.pt` file. Decode it offline on a higher-memory GPU.

```bash
python examples/hunyuan_generate.py \
    --model /models/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --num-frames 257 --output-dir benchmarks --output-name out_2x \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-latent
```

The mp4 path becomes a `.pt` path:
```python
import torch
data = torch.load("out_2x.pt")   # {"latent": tensor, "num_frames": 257, ...}
```

---

## Symptom 3 — Output mp4 missing after Docker run

**Root cause**: an absolute host path was passed to `--output`, the container interprets it relative to its own filesystem (not the bind-mount), so the mp4 lands inside the container and disappears with `--rm`.

### Diagnosis
The run log says `[export] /home/me/.../out.mp4 (1.4s)` but `ls /home/me/.../out.mp4` on the host returns nothing.

### Fix
Use **relative paths** against the container's working directory + bind-mount:
```bash
# WRONG — path lost on --rm
docker run ... lvsa-vllm-omni:latest \
    python examples/hunyuan_generate.py --output-name /home/me/out.mp4

# RIGHT — relative path, lands on host
docker run ... -v $(pwd):/workdir/code -w /workdir/code ... \
    lvsa-vllm-omni:latest \
    python examples/hunyuan_generate.py --output-name benchmarks/results/out.mp4
```

Always append `&& chown -R $(id -u):$(id -g) <output_dir>` so files written as root are readable on host.

---

## Symptom 4 — Quality regression at training reference

**Root cause**: `LVSA_REFERENCE_LATENT_FRAMES` is wrong for your model. Auto-keyframe scheduler computes sparsity using the wrong budget, so even at training horizon you get partial attention coverage.

### Diagnosis
```
[LVSA] reference_latent_frames=21 target_latent_frames=33 extension_ratio=1.57x
```

If you're running HunyuanVideo at 129 frames and `reference_latent_frames=21` with `extension_ratio` > 1.0 (should be exactly 1.0), the scheduler is using Wan's default.

### Fix
```bash
LVSA_REFERENCE_LATENT_FRAMES=33    # HunyuanVideo
LVSA_REFERENCE_LATENT_FRAMES=21    # Wan 2.x
LVSA_REFERENCE_LATENT_FRAMES=13    # CogVideoX
```

The single most common LVSA configuration mistake. **Always set explicitly.**

---

## Symptom 5 — `dynamic_quality` drops 5+ points

**Root cause**: documented trade-off of any sparse-attention scheme. Long-range pairs that contribute to large-scale motion coherence are skipped. Motion-heavy prompts can lose ~5 points on VQeval `dynamic_quality` and ~0.02 on VBench `motion_smoothness`.

### Diagnosis
Compare per-dimension VQeval between dense and LVSA on the same prompts. If only `dynamic_quality` drops while `loop_quality` and `text_alignment` improve, this is the trade-off.

### Fix
1. **`sparsity_scale = 1.0`** (default) — at T_lat ≤ ref, fully-dense via LVSA path; speedup without sparsity cost.
2. **Increase `window_size` to 16** (`W=4` latent) — more long-range mixing per query. Costs ~10% wall time.
3. **Switch to dense for motion-heavy prompts.**

---

## Symptom 6 — `loop_quality` improves by +30 points (too good?)

**Root cause**: prompt-specific. LVSA's rotating-keyframe pattern dithers attention each step, preventing dense's looping/static failure mode. When dense already loops (loop_quality < 40), LVSA can lift by +30 to +40. Prompts where dense wasn't looping see ~+5.

### Diagnosis
Look at dense's baseline `loop_quality`. If < 40 and LVSA's > 60, the gain is real but largely a function of dense's weakness.

### Fix
Not a bug — this is the rotating-keyframe mechanism working. When reporting, include the dense baseline's `loop_quality` so reviewers see dense's behavior on that prompt.

---

## Symptom 7 — Graduated schedule env vars (`LVSA_SCHEDULE_*`) don't do anything

**Root cause**: `LVSA_SCHEDULE_START` and `LVSA_SCHEDULE_END` are **soft-deprecated** in v1.0. The recommended runtime knob is `LVSA_SPARSITY_SCALE`. The schedule vars still exist for backwards compatibility (default `0`) but produce no behavior at default settings.

### Fix
Use `sparsity_scale` instead. See the [`lvsa-tuning`](../lvsa-tuning/SKILL.md) skill.

---

## Filing a bug

Capture:
- Full stdout/stderr of the run
- Output of `grep -E "\[(LVSA|LVSA)" run.log`
- Output of `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` (or `npu-smi info` on Ascend)
- Generated mp4 / .pt file size
- Exact `docker run` / `python` command + env vars
- LVSA version: `git rev-parse HEAD`

Open at <https://github.com/JiusiServe/LongVideoSparseAttention/issues>.
