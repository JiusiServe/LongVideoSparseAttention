# Tuning LVSA

## The three primary knobs

LVSA exposes three runtime knobs that trade attention coverage against speedup:

1. **`reference_latent_frames`** — model-dependent fixed value. Determines the per-query attention budget the auto-keyframe scheduler targets.
2. **`sparsity_scale`** — multiplier on `reference_latent_frames`. The runtime knob that lets users dial sparser/denser without touching code.
3. **`window_size`, `n_first_frames`** — geometry of the local window and number of leading global anchors. Usually left at defaults.

## Knob 1 — `reference_latent_frames` (per-model)

Set once per model:

| Model | `LVSA_REFERENCE_LATENT_FRAMES` | Equivalent video frames |
|---|---|---|
| Wan 2.1 / 2.2 (1.3B, 14B) | `21` | 81 |
| HunyuanVideo 1.5 | `33` | 129 |
| CogVideoX 5B | `13` | 49 |

**This is the single most common configuration mistake.** Forgetting `LVSA_REFERENCE_LATENT_FRAMES=33` on HunyuanVideo silently uses Wan's default `21`, producing wrong sparsity geometry. **Always set it explicitly.**

### Why the value matters

The auto-keyframe scheduler runs:

```
scaled_ref     = max(n_first + 1, int(reference_frames × sparsity_scale))
target_attended = min(scaled_ref, T_lat)
target_globals = max(n_first, target_attended − (2W + 1))

# pick the largest kfi (sparsest globals) such that |globals| ≥ target_globals
```

If `T_lat ≤ scaled_ref`, the function short-circuits to `kfi=1` (every frame is a global → fully dense attention). Real sparsity only engages when `T_lat > scaled_ref`.

## Knob 2 — `sparsity_scale` (runtime quality/speed knob)

> **Note:** `sparsity_scale` is **not used in the published paper experiments** (all paper runs use `sparsity_scale=1.0`). It's exposed here as an exploratory knob; quantitative recommendations below are mechanism-level, not validated empirical claims.

`LVSA_SPARSITY_SCALE` multiplies the per-query attention budget:

```
scaled_ref     = max(n_first + 1, int(reference_frames × sparsity_scale))
target_attended = min(scaled_ref, T_lat)
```

### How the knob behaves

- **At `T_lat ≤ reference`**, any `sparsity_scale ≥ 1.0` collapses to `kfi=1` (fully dense). The visible speedup at the default `1.0` is implementation efficiency only — the LVSA attention path bypasses overhead in the model's native attention processor; not pattern-driven sparsity.
- **`sparsity_scale = 2.0` is equivalent to `1.0` at `T_lat ≤ reference`** (both give `kfi=1`). The conservative knob is meaningful only at extrapolation lengths.
- **`sparsity_scale < 1.0` activates pattern-driven sparsity even at training reference**: e.g. on HunyuanVideo (ref=33) at 1×, scale=0.5 shrinks the budget from 33 to 16 latents — engaging real ~52% coverage.
- **`sparsity_scale < 1.0` reduces per-query attended frames** and therefore long-range mixing. This trades off `dynamic_quality` (motion coherence) against speedup; users who want maximum motion fidelity should keep `s=1.0`.

### Rule of thumb (mechanism, not measured)

| Goal | `sparsity_scale` |
|---|---|
| Match dense quality at training reference (paper default) | `1.0` |
| Force pattern sparsity at training reference | `< 1.0` (e.g. `0.5`) |
| Aggressive extrapolation at 3×+ horizon, OOM-prevention | `< 1.0` |
| Conservative quality at extrapolation, willing to give up some speed | `> 1.0` (e.g. `1.5`) |

## Knob 3 — `window_size` and `n_first_frames`

Defaults (`12 / 4` in video frames = `3 / 1` in latent for VAE temporal factor 4) are usually fine. Reduce only if you explicitly want a tighter floor:

```
floor_attended = 2W + 1 + n_first   (in latent frames)
```

With `W=4 latents (LVSA_WINDOW_SIZE=16)`, `n_first=1 latent (LVSA_N_FIRST_FRAMES=4)`, the floor is **10 latents**. If your `reference_latent_frames` is smaller than this, the floor becomes binding and you cannot reach the requested budget. Choose `W` so that `2W+1 + n_first ≤ reference_latent_frames`.

## Sparsity vs sequence length (default config: W=4 latents, n_first=1, scale=1.0)

| `T_lat` | HunyuanVideo (ref=33) | Wan (ref=21) |
|---|---|---|
| 21 (Wan 1×) | 0% (T < ref) | **0%** ← ref |
| 33 (HV 1×) | **0%** ← ref | 21% |
| 41 | 0% (kfi=2 only gives 21 globals < 24 target) | 44% |
| 49 (HV 1.5×) | 31% | 55% |
| 65 (HV 2×) | 35% | 66% |
| 81 (Wan 4×) | 56% | 74% |
| 121 (Wan 6×) | 72% | 82% |

HV stays at 0% sparsity for a wider band above `ref` than Wan because the algorithm requires `target_globals = ref − (2W+1) = 24` globals, and `kfi=2` only reaches 24 globals at `T_lat ≥ 47`. To engage sparsity closer to `ref` on HunyuanVideo, lower `sparsity_scale` (e.g. `0.7`).

## Compute the budget yourself

Inline:

```python
from lvsa.sparse_attention import (
    compute_auto_kfi, compute_global_indices, get_window_bounds,
)
T = 49             # latent frames you want to generate
W = 4              # latent window half-width
n_first = 1        # latent leading globals
ref = 33           # HunyuanVideo
scale = 1.0

kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref, sparsity_scale=scale)
globals_idx = set(compute_global_indices(T, n_first, kfi))
print(f"kfi={kfi}, |globals|={len(globals_idx)}")

attended_per_query = []
for f in range(T):
    lo, hi = get_window_bounds(f, W, T, expand=True,
                               global_set=globals_idx,
                               global_count=len(globals_idx))
    win = set(range(lo, hi+1)) if lo <= hi else set()
    attended_per_query.append(len(globals_idx | win))

print(f"avg attended = {sum(attended_per_query)/len(attended_per_query):.1f}/{T}  "
      f"sparsity = {100*(1 - sum(attended_per_query)/(len(attended_per_query)*T)):.1f}%")
```

## Verifying engagement

After every generation, the log prints:

```
[LVSA] reference_latent_frames=33  target_latent_frames=49  extension_ratio=1.48x
[LVSA] computed key_frame_interval=2 (latent frames)
[LVSA] kfi=2  global_count=25  attended_per_frame=34/49
```

Read this as:
- `extension_ratio` = `T_lat / reference`. Below 1.0 = sub-reference, exactly 1.0 = at training horizon, above 1.0 = extrapolation.
- `attended_per_frame=N/T` — the per-query attended count vs total frames. **Sparsity = `1 − N/T`**.
- If `N == T`, you are at fully-dense attention (T ≤ scaled_ref).

If you see `[LVSA-FALLBACK]` warnings, the backend silently disengaged — see [`troubleshooting.md`](troubleshooting.md).
