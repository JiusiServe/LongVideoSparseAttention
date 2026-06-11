# Architecture

## Overview

LVSA is a **model-agnostic sparse-attention engine** plus **per-model adapters** that plug it into specific video diffusion transformers.

```
┌──────────────────────────────────────────────────────┐
│ Model: Wan / HunyuanVideo / CogVideoX / your model   │
└─────────────────┬────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────┐
│ ModelAdapter (lvsa/adapters/<model>.py)              │
│   • patches_per_frame, latent_frames                 │
│   • reference_latent_frames                          │
│   • extract_qkv, apply_rotary, output_projection     │
│   • install_processor, setup_context_parallel       │
└─────────────────┬────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────┐
│ DistributedLVSAProcessor (lvsa/lvsa_processor.py)      │
│   ─ window math, global indexing, rotation           │
│   ─ CSR construction                                 │
│   ─ KV gather / context-parallel coordination        │
└─────────────────┬────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────┐
│ Backend dispatch (lvsa/sparse_attention.py)          │
│   ─ SDPA (default; CUDA + Ascend NPU + CPU)          │
│   ─ FlashInfer (CUDA only)                           │
└──────────────────────────────────────────────────────┘
```

The engine never imports model code; the adapter never implements attention math. Adding a new model is one adapter file (~200–330 lines).

## The sparse pattern

For each query frame `f`, LVSA attends to:

1. **Global anchors**: `n_first` frames at the start (default 4 video frames = 1 latent) **plus** periodic keyframes at interval `kfi`.
2. **Local window**: `2W+1` frames centered on `f` (default `W=12 video = 3 latent` half-width), expanded outward when global frames overlap.

The keyframe grid **rotates by one position each denoising step**, so frame 7 is a global at step 0, frame 8 at step 1, etc. Over the 50-step trajectory, every frame eventually serves as a global anchor — preventing the fixed-grid bias that causes dense attention to produce frozen/looping output.

The per-query attention budget is bounded by `n_first + min(2W+1, T - num_globals)` — typically equal to the model's training horizon. Wall-clock scales linearly in `T` instead of quadratically.

## Auto-keyframe scheduler

`compute_auto_kfi(T, W, n_first, reference_frames, sparsity_scale)`:

```
scaled_ref     = max(n_first + 1, int(reference_frames × sparsity_scale))
target_attended = min(scaled_ref, T)
target_globals = max(n_first, target_attended − (2W + 1))

if T ≤ scaled_ref:
    return 1                  # every frame is a global → fully dense

# Pick the largest kfi (sparsest globals) such that |globals| ≥ target_globals
for kfi in range(T, 0, -1):
    if |globals_at(kfi)| >= target_globals:
        return kfi
```

The function guarantees the per-query budget is at least `target_attended`, then picks the sparsest possible keyframe interval that satisfies it. At extrapolation lengths this produces increasingly sparse patterns; at training reference it returns `kfi=1` (fully dense).

## The ModelAdapter ABC

`lvsa/adapters/base.py` defines 11 abstract methods grouped into 4 families. Adding a new model means subclassing `ModelAdapter` and implementing each.

### Geometry (3 methods)

```python
def patches_per_frame(self, height: int, width: int, pipe: Any) -> int: ...
def latent_frames(self, num_frames: int, pipe: Any) -> int: ...
def reference_latent_frames(self, pipe: Any) -> int: ...
```

- **`patches_per_frame`**: spatial tokens per latent frame after VAE compression + patchification. For 480×832, VAE factor 8, patch size 2: `(480/8/2) × (832/8/2) = 1560`.
- **`latent_frames`**: `(num_frames − 1) // vae_temporal + 1`. Wan/HV use 4.
- **`reference_latent_frames`**: training horizon in latent frames. Critical — anchors the scheduler.

### QKV extraction (3 methods)

```python
def extract_qkv(self, attn, hidden, encoder) -> (Q, K, V): ...
def extract_cross_attn_kv(self, attn, encoder) -> Optional[(K, V)]: ...
def split_encoder_for_cross_attn(self, attn, encoder) -> (text, image_or_None): ...
```

Single-stream models (Wan) concatenate text+video before projection. Dual-stream (HunyuanVideo) projects them through separate linears and joins for the attention call. I2V-style cross-attention is optional.

### Position encoding + output (2 methods)

```python
def apply_rotary(self, q, k, rotary_emb, local_seq, rank, world) -> (Q, K): ...
def output_projection(self, attn, hidden, query_dtype) -> [B, seq, C]: ...
```

Each model has its own RoPE format (Wan: 1D, HunyuanVideo: 3D `(cos, sin)` tuples). The adapter handles model-specific slicing for context-parallel.

### Encoder + dual-stream + integration (3 methods)

```python
def extract_encoder_qkv(self, attn, encoder) -> Optional[(Q, K, V)]: ...
def format_output(self, attn, hidden, encoder_output, encoder_seq_len, dtype) -> ...: ...
def cross_attention(self, attn, query, encoder_image, backend) -> Optional[Tensor]: ...
```

`extract_encoder_qkv` returns `None` for single-stream models. `format_output` returns either a tensor (single-stream) or `(hidden, encoder)` tuple (dual-stream).

### Pipeline integration (3 methods)

```python
def install_processor(self, pipe, processor) -> int: ...
def setup_context_parallel(self, transformer, world) -> None: ...
def patch_rotary_for_cp(self, rank: int, world: int) -> None: ...
```

These wire the adapter into the model's pipeline (find the right attention layers, monkey-patch the RoPE, attach a context-parallel plan).

## Reference adapters

| Model | Adapter | LoC | Notes |
|---|---|---|---|
| Wan 2.1 / 2.2 | `lvsa/adapters/wan.py` | ~220 | Single-stream, 1D RoPE, T5 encoder. Simplest reference. |
| HunyuanVideo 1.5 | `lvsa/adapters/hunyuan_video.py` | ~330 | Dual-stream, 3D RoPE, Qwen2.5-VL encoder. |
| CogVideoX | `lvsa/adapters/cogvideox.py` | ~190 | Joint-attention, shared QKV. |

### When a model doesn't fit the ABC — the processor-swap path (Cosmos 3.0)

Not every model maps onto the `ModelAdapter` ABC. **Cosmos 3.0** (`lvsa/cosmos3.py`) is *separate-stream*: diffusers' `Cosmos3AttnProcessor.__call__(attn, und_seq, gen_seq, rotary_emb)` takes the text/VLM (`und`) and video (`gen`) tokens as **two separate tensors** and runs two attentions — `und` causal self-attention, and `gen` full attention over `cat([k_und, k_gen])`. That asymmetric `(und, gen)` signature doesn't fit the ABC's single-sequence `extract_qkv`/`format_output` contract.

Instead of forcing the ABC, Cosmos uses a **drop-in attention-processor swap**:

- `Cosmos3LVSAAttnProcessor` replicates the `und` causal pathway byte-identically and replaces only the `gen` full-attention with the LVSA windowed pattern (window gen↔gen + keyframes/`n_first`, with **all** `k_und` kept global — appended after the gen anchors).
- `install_cosmos3_lvsa(transformer, num_frames, height, width, ...)` builds one shared processor (geometry is identical per layer) and calls `set_processor` on every `transformer.layers[i].self_attn`.
- It reuses the model-agnostic core directly (`LVSAMetadata`, `lvsa_sdpa`, `build_global_kv`, `compute_auto_kfi`) — no adapter, no CP plan (MVP is single-GPU).

The contract is proven by a **1×-equivalence test**: at the native horizon (`T_lat == reference_latent_frames`) the auto-scheduler returns `kfi=1`, every gen frame becomes a global anchor, and the gen pathway reduces to the *exact* dense `cat([k_und, k_gen])` attention — matching diffusers' stock processor to within bf16 noise. Use this pattern when a new model's attention is asymmetric or otherwise doesn't fit the ABC.

For a step-by-step adapter authoring walkthrough (and the processor-swap alternative), see the [`lvsa-add-model` skill](../skills/lvsa-add-model/).

## Backends

The same sparse pattern can be dispatched through two attention kernels:

| Backend | When fastest | Memory | How sparse pattern is encoded | Hardware |
|---|---|---|---|---|
| **SDPA** | Short sequences (T_lat ≤ 33), all NPU runs | high (full QKV per frame) | per-frame Q/K/V slicing, no per-pair mask | CUDA, NPU, CPU |
| **FlashInfer** | Long sequences (T_lat ≥ 49) | low (compact KV buffer) | block-sparse CSR (variable-length attention) | CUDA only |

Both produce numerically equivalent outputs within FP16/BF16 noise. FlashInfer's block-sparse CSR has the lowest per-call overhead at long sequences, which is why the headline 3.81× speedup on HunyuanVideo at 1.5× horizon is FlashInfer-routed.

## Multi-GPU paths

LVSA supports **Ulysses-style context parallelism** on top of standard PyTorch distributed primitives. The sequence dim is sharded across ranks; each rank holds a full model copy and processes its local token shard. Global K/V are gathered before the sparse-attention call.

`setup_context_parallel()` on the adapter wires the rank-aware sharding into the model's pipeline. The processor handles the rest.

**Constraint**: `seq_len = T_lat × patches_per_frame` must be divisible by `world_size`.

## Code map

```
lvsa/
├── sparse_attention.py        LVSAMetadata, compute_auto_kfi, lvsa_sdpa, lvsa_flashinfer, top-level dispatcher
├── lvsa_processor.py           DistributedLVSAProcessor, rebuild logic, _compute_lvsa{_sdpa,_flashinfer}
├── parallel.py                CP setup, LVSA installation entry point (install_lvsa_processors)
├── rope.py                    RoPE apply/slice helpers
├── device.py                  Device-agnostic helpers (CUDA / Ascend NPU / CPU)
├── riflex.py                  Optional RIFLEx RoPE rescaling
├── cosmos3.py                 Cosmos 3.0 (experimental) — geometry + Cosmos3LVSAAttnProcessor + install_cosmos3_lvsa (processor swap, not the ABC)
└── adapters/
    ├── base.py                ModelAdapter ABC
    ├── wan.py                 Wan 2.1 / 2.2
    ├── hunyuan_video.py       HunyuanVideo 1.5
    └── cogvideox.py           CogVideoX 5B (experimental)
```

## Adding a new model — TL;DR

1. Copy the closest existing adapter (`wan.py` for single-stream, `hunyuan_video.py` for dual-stream).
2. Set `reference_latent_frames` to the model's training horizon in latent frames.
3. Adjust `extract_qkv`, `apply_rotary`, `output_projection`, `install_processor` for the new model's attention layer names and RoPE format.
4. Add a smoke test to `tests/test_adapters.py` (CPU-only).
5. Wire into a fresh `examples/<your_model>_generate.py` (copy + edit).

For the full step-by-step, see the [`lvsa-add-model` skill](../skills/lvsa-add-model/).
