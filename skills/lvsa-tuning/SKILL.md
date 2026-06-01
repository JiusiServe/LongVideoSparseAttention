---
name: lvsa-tuning
description: Tune LVSA for quality vs speed. Use when adjusting sparsity_scale, choosing window_size and n_first_frames, deciding when --rotate-keyframes pays off, composing LVSA with RIFLEx, or hitting a quality regression and needing to back off sparsity.
---

# LVSA Tuning

## The three primary knobs

| Knob | What it controls | When to touch it |
|---|---|---|
| `reference_latent_frames` | Per-query attention budget anchor | Set once per model (Wan=21, HV=33, Cog=13). Don't change at runtime. |
| `sparsity_scale` | Multiplier on the budget | The runtime quality/speed dial. Default `1.0`; lower = sparser. |
| `window_size`, `n_first_frames` | Local-window geometry | Usually leave at defaults (12 frames / 4 frames). Only touch if you want a tighter floor. |

## sparsity_scale — the headline dial

`LVSA_SPARSITY_SCALE` (env var) or `--sparsity-scale` (CLI). Scales the auto-keyframe scheduler's per-query budget.

```
scaled_ref     = max(n_first + 1, int(reference_frames × sparsity_scale))
target_attended = min(scaled_ref, T_lat)
```

Empirical results on HunyuanVideo at 129 frames (training reference, single-prompt "dog"):

| `sparsity_scale` | Per-step | Speedup vs dense | VQeval composite | VQeval loop |
|---|---|---|---|---|
| (dense baseline) | 44.0 s | — | 57.6 | 32.6 |
| `0.5` (aggressive) | 18.3 s | **2.40×** | **65.2** (+7.6) | **73.6** (+41.0) |
| `1.0` (default) | 22.5 s | 1.96× | **61.3** (+3.7) | **63.0** (+30.4) |

### Rule of thumb

| Goal | `sparsity_scale` | Why |
|---|---|---|
| Match dense quality at training reference, take implementation speedup | `1.0` | At T_lat ≤ ref this collapses to kfi=1 (fully dense). Speedup comes from bypassing native attention overhead. |
| Maximum speedup at training reference | `0.5` | Engages pattern-driven sparsity even at T_lat=ref. Big loop-quality gains; ~5pt drop on `dynamic_quality`. |
| Aggressive extrapolation, OOM-prevention at 3×+ horizon | `0.5` | Shrinks compact-K buffer, helps fit on 80 GB. |
| Conservative quality at extrapolation | `0.75` | Reduces sparsity gradient; less speedup but keeps motion intact. |

### Important nuances

- **At T_lat ≤ reference**, any `sparsity_scale ≥ 1.0` collapses to `kfi=1` (fully dense). The visible speedup is implementation efficiency only.
- **`sparsity_scale = 2.0` is equivalent to `1.0` at T_lat ≤ reference** (both give kfi=1). The conservative knob is meaningful only at extrapolation lengths.
- **`sparsity_scale = 0.5` activates real pattern sparsity even at training reference**: HV's budget shrinks from 33 to 16 latents at 1×, giving ~52% coverage.
- **The large loop_quality gain at `s=0.5`** comes from `--rotate-keyframes` dithering the attention pattern each step. Disable rotation and the loop gain disappears.

## window_size + n_first_frames

Defaults:
- `window_size = 12` video frames = `3` latent frames (W=3)
- `n_first_frames = 4` video frames = `1` latent frame (n_first=1)

Floor of attended frames per query: `2W+1 + n_first = 8` latent frames.

**When to reduce W**: never, unless your `reference_latent_frames` is below the floor. The defaults are tuned for current models.

**When to increase W** (e.g. W=4):
- Motion-heavy prompts losing `dynamic_quality` at extension — bigger window = more long-range mixing inside each query's attended set.
- Costs ~10% wall time per `W += 1`.

## Should I use --rotate-keyframes?

| At length | Without rotation | With rotation |
|---|---|---|
| T_lat ≤ reference | No effect (kfi=1 means every frame is a global anyway) | No effect |
| Slight extension (T ≈ 1.5×) | Static keyframes can introduce period artifacts | Smoother |
| Heavy extension (T ≥ 3×) | Output starts to loop / freeze | **Strongly preferred** — this is the mechanism that prevents the "frozen video" failure mode |

**Default `--rotate-keyframes` on whenever you're extending.** Off at training horizon adds nothing.

## Composing with RIFLEx

[RIFLEx](https://arxiv.org/abs/2502.15894) rescales the RoPE frequencies to extrapolate beyond the training horizon. It's orthogonal to LVSA (RoPE-only, no attention compute change) and stacks cleanly:

```bash
python examples/wan_generate.py \
    --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
    --prompt "..." \
    --num-frames 321 \
    --lvsa --flashinfer --rotate-keyframes --auto-keyframes \
    --riflex --riflex-s 4.0
```

At extension lengths RIFLEx + LVSA-FI is the recommended recipe. On the SotA grid (Wan 1.3B, 5 prompts):

| Horizon | LVSA-FI alone | LVSA-FI + RIFLEx |
|---|---|---|
| 2× | 1.43× faster than Dense | ~same speed, slight quality bump |
| 4× | 2.41× faster than Dense | ~same speed, +1 VQeval |

RIFLEx adds **zero** measurable wall-time overhead (verified: 0.99–1.00× Dense).

## Verifying engagement

After every run, the `[LVSA]` log line tells you exactly what the scheduler did:

```
[LVSA] kfi=6 global_count=14 attended_per_frame=21/81
```

- `kfi=6` — every 6th frame is a periodic global anchor (auto-derived)
- `global_count=14` — total global frames in the pattern (n_first + periodic)
- `attended_per_frame=21/81` — each query attends to 21 frames out of 81 → 74% sparsity

For non-default geometry, use the inline helper in [`docs/tuning.md`](../../docs/tuning.md) to compute the budget yourself.

## Diagnostics

| Symptom | Likely root cause | Fix |
|---|---|---|
| No quality improvement vs Dense | `sparsity_scale` too high at training horizon | Drop to `0.5` |
| Motion quality regressed | Window too small for fast-motion prompt | Try `--window-size 16` (W=4) |
| Video loops at extension | `--rotate-keyframes` not set | Add the flag |
| `attended_per_frame=N/T` shows N==T at extension | `reference_latent_frames` too high | Verify per-model value |

See [`lvsa-troubleshooting`](../lvsa-troubleshooting/SKILL.md) for the full failure-mode catalog.
