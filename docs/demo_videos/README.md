# Demo videos

Four hand-picked clips from the paper-reproduction runs. All videos at 480p, 50 denoising steps, single A100 80GB, seed 16.

## `01_wan13b_4x_dense_dog.mp4` — Dense baseline at 4× horizon

- **Model**: Wan 2.1 T2V 1.3B
- **Length**: 333 frames (~4× the 81-frame training horizon)
- **Method**: Dense attention
- **Wall time**: 1930 s (mean of 5 prompts)
- **Why this is here**: paired comparison with clip `02`. At 4× horizon Dense begins to produce static/looping output — VBench's `subject_consistency` shoots to 0.986 not because the video is high quality, but because consecutive frames are nearly identical.

## `02_wan13b_4x_lvsa-fi_dog.mp4` — LVSA-FlashInfer, headline speedup

- **Model**: Wan 2.1 T2V 1.3B
- **Length**: 333 frames (4× horizon)
- **Method**: LVSA + FlashInfer + rotating keyframes
- **Wall time**: 802 s (mean of 5 prompts) — **2.41× faster than Dense**, **3.27× faster than UltraViCo** (2621 s)
- **Quality**: VQeval composite 62.3 vs Dense 52.4 (+9.9); VBench-Long `imaging_quality` 0.587 vs Dense 0.489 (+0.10).
- **Why this matters**: this is the headline single result of the paper — LVSA-FI dominates the SotA grid at 4× horizon. Faster than every baseline, higher quality on every dimension that doesn't reward static-frame repetition.

Reproduce:

```bash
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "A dog running in the forest." \
    --num-frames 333 --steps 50 --seed 16 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --output-name 02_wan13b_4x_lvsa-fi_dog.mp4
```

## `03_hv15_2x_lvsa-fi_ocean.mp4` — HunyuanVideo cross-model demo

- **Model**: HunyuanVideo 1.5 (480p)
- **Length**: 257 frames (~2× the 129-frame training horizon)
- **Method**: LVSA + FlashInfer + rotating keyframes
- **Wall time**: ~1617 s on single A100 (vs Dense which **OOMs** at this length on 80 GB).
- **Why this matters**: this length is **only reachable** with LVSA on a single 80 GB GPU. Dense's KV cache for 257 latent frames exceeds the memory budget. LVSA's compact KV buffer (via FlashInfer block-sparse CSR) keeps it under the limit.

## `04_wan14b_6x_lvsa-fi_dog.mp4` — Extreme 6× extrapolation

- **Model**: Wan 2.1 T2V 14B
- **Length**: 481 frames (~6× the 81-frame training horizon)
- **Method**: LVSA + FlashInfer + rotating keyframes
- **Why this matters**: at 6× horizon the rotating-keyframe pattern is doing the most work — without it the video falls into a static loop. With LVSA the motion stays coherent for the full 20-second clip.

## What you should see

| Clip | Wall (1×A100) | Quality vs Dense |
|---|---|---|
| 01 wan13b 4× Dense | 1930 s | reference baseline; VBench `subject_consistency`=0.986 (static), `imaging_quality`=0.49 |
| 02 wan13b 4× LVSA-FI | **802 s** | composite +9.9, `imaging_quality` +0.10, `subject_consistency`=0.939 (motion preserved) |
| 03 hv15 2× LVSA-FI | ~1617 s | Dense **OOMs** at this length |
| 04 wan14b 6× LVSA-FI | longer (single-GPU) | dynamic quality preserved at extreme extension |

For the full quality breakdown across 5 prompts and 3 horizons on the SotA grid, see the per-dimension tables in the [paper](https://arxiv.org/abs/...) and the regenerated figures in [`../figures/`](../figures/).

## Disk usage

~18 MB total across 4 mp4s.
