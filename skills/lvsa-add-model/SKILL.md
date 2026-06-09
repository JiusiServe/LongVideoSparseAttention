---
name: lvsa-add-model
description: Add LVSA support for a new video diffusion model. Use when implementing the ModelAdapter ABC for a new DiT (single-stream like Wan, dual-stream like HunyuanVideo, or joint-attention like CogVideoX), wiring it into examples/<model>_generate.py, or adding a vllm-omni hook for it.
---

# Adding a new model to LVSA

## Overview

LVSA's engine is model-agnostic; per-model behavior lives in `lvsa/adapters/<model>.py` as a subclass of `ModelAdapter` (defined in `lvsa/adapters/base.py`). Adding a new model is **one adapter file** (~200-330 lines) plus a tiny example wrapper.

Reference adapters:

| Model | File | Style | LoC | Use as template if |
|---|---|---|---|---|
| Wan 2.1 / 2.2 | `lvsa/adapters/wan.py` | Single-stream | ~220 | Your model concatenates text+video before projection |
| HunyuanVideo 1.5 | `lvsa/adapters/hunyuan_video.py` | Dual-stream | ~330 | Text and video have separate Q/K/V projections |
| CogVideoX 5B | `lvsa/adapters/cogvideox.py` | Joint-attention | ~190 | Shared Q/K/V projection over joint text+video |

## First decide: ABC adapter vs. processor swap

The `ModelAdapter` ABC assumes a **single joint sequence** flows through one attention call (it exposes `extract_qkv` / `format_output` over one hidden-state tensor). Most DiTs fit. **Some don't** — if the model's attention takes the text and video streams as *separate tensors* with *asymmetric* attention (e.g. one causal, one full), forcing the ABC fights the design.

For those, use a **drop-in attention-processor swap** instead — see `lvsa/cosmos3.py` (Cosmos 3.0) as the reference:

- diffusers' `Cosmos3AttnProcessor.__call__(attn, und_seq, gen_seq, rotary_emb)` runs `und` (text/VLM) as causal self-attn and `gen` (video) as full attn over `cat([k_und, k_gen])`.
- `Cosmos3LVSAAttnProcessor` replicates the `und` path byte-identically and LVSA-izes only the `gen` path (window gen↔gen + keyframes; all `k_und` appended as global). It calls the core directly (`LVSAMetadata`, `lvsa_sdpa`, `build_global_kv`, `compute_auto_kfi`) — **no ABC subclass**.
- `install_cosmos3_lvsa(transformer, num_frames, height, width, ...)` swaps the processor onto every `transformer.layers[i].self_attn` (diffusers' `set_processor`).
- Prove correctness with a **1×-equivalence test**: at `T_lat == reference_latent_frames`, `compute_auto_kfi` → `kfi=1` → every frame global → the LVSA gen path must equal the model's stock dense attention to within bf16 noise. (`tests/test_cosmos3_processor.py` is the template.)

Decision rule: **single joint sequence → ABC adapter** (rest of this skill). **Separate streams / asymmetric attention → processor swap** (mirror `lvsa/cosmos3.py`, skip the ABC). The geometry helpers (`patches_per_frame`, `latent_frames`, `reference_latent_frames`) are needed either way.

## The 11 methods to implement (ABC adapter path)

The `ModelAdapter` ABC groups methods into 4 families.

### A. Geometry (3 methods)

```python
def patches_per_frame(self, height: int, width: int, pipe: Any) -> int:
    """Spatial tokens per latent frame after VAE + patchify."""
    # For 480x832, VAE factor 8, patch size 2:
    # (480 / 8 / 2) * (832 / 8 / 2) = 1560
    ...

def latent_frames(self, num_frames: int, pipe: Any) -> int:
    """(num_frames - 1) // vae_temporal + 1. Wan/HV use 4."""
    ...

def reference_latent_frames(self, pipe: Any) -> int:
    """Training horizon in latent frames. CRITICAL — anchors the auto scheduler.
    Wan 1.3B/14B = 21, HunyuanVideo 1.5 = 33, CogVideoX 5B = 13."""
    ...
```

### B. QKV extraction (3 methods)

```python
def extract_qkv(self, attn, hidden, encoder) -> tuple[Tensor, Tensor, Tensor]:
    """Return Q, K, V for the joint sequence. Single-stream: concat text+video
    before projection. Dual-stream: each modality has its own linear."""
    ...

def extract_cross_attn_kv(self, attn, encoder) -> Optional[tuple[Tensor, Tensor]]:
    """Return (K, V) for an I2V-style image-conditioned cross-attention, or None."""
    ...

def split_encoder_for_cross_attn(self, attn, encoder) -> tuple[Tensor, Optional[Tensor]]:
    """For dual-modal encoders that split text vs image tokens."""
    ...
```

### C. Position encoding + output (2 methods)

```python
def apply_rotary(self, q, k, rotary_emb, local_seq, rank, world) -> tuple[Tensor, Tensor]:
    """Apply RoPE. Wan uses 1D RoPE; HunyuanVideo uses 3D (cos, sin) tuples."""
    ...

def output_projection(self, attn, hidden, query_dtype) -> Tensor:
    """Final linear (and any layernorm) after attention."""
    ...
```

### D. Dual-stream + integration (3 methods)

```python
def extract_encoder_qkv(self, attn, encoder) -> Optional[tuple[Tensor, Tensor, Tensor]]:
    """Dual-stream models project encoder Q/K/V through a separate linear.
    Return None for single-stream."""
    ...

def format_output(self, attn, hidden, encoder_output, encoder_seq_len, dtype):
    """Single-stream: return a tensor. Dual-stream: return (hidden, encoder) tuple."""
    ...

def cross_attention(self, attn, query, encoder_image, backend) -> Optional[Tensor]:
    """I2V cross-attention call, or None."""
    ...
```

### E. Pipeline integration (3 methods)

```python
def install_processor(self, pipe, processor) -> int:
    """Walk the transformer, replace each self-attention's processor with
    `DistributedLVSAProcessor`. Return number of blocks installed on."""
    ...

def setup_context_parallel(self, transformer, world) -> None:
    """Attach a CP plan (which dim to split, gather strategy)."""
    ...

def patch_rotary_for_cp(self, rank: int, world: int) -> None:
    """Monkey-patch the model's RoPE to be rank-aware for context-parallel runs."""
    ...
```

## Step-by-step

### 1. Copy the closest existing adapter

```bash
# Single-stream model
cp lvsa/adapters/wan.py lvsa/adapters/<your_model>.py

# Dual-stream model
cp lvsa/adapters/hunyuan_video.py lvsa/adapters/<your_model>.py
```

### 2. Set the geometry

```python
def patches_per_frame(self, height: int, width: int, pipe: Any) -> int:
    vae_sf = pipe.vae.config.scaling_factor      # spatial compression
    patch = pipe.transformer.config.patch_size   # patchify factor
    h = height // vae_sf // patch
    w = width // vae_sf // patch
    return h * w

def latent_frames(self, num_frames: int, pipe: Any) -> int:
    vae_tf = pipe.vae.config.temporal_compression_ratio
    return (num_frames - 1) // vae_tf + 1

def reference_latent_frames(self, pipe: Any) -> int:
    return 21   # CHANGE: read from model card / paper
```

### 3. Wire QKV + RoPE

Identify the model's attention class (usually `<ModelName>Attention` or `<ModelName>SelfAttn`). Adapt `extract_qkv` and `apply_rotary` to match its forward signature.

For Wan-style 1D RoPE:
```python
def apply_rotary(self, q, k, rotary_emb, local_seq, rank, world):
    from .rope import apply_rotary_1d
    return apply_rotary_1d(q, k, rotary_emb, local_seq, rank, world)
```

For HunyuanVideo-style 3D `(cos, sin)` RoPE:
```python
def apply_rotary(self, q, k, rotary_emb, local_seq, rank, world):
    from .rope import apply_rotary_3d_cos_sin
    return apply_rotary_3d_cos_sin(q, k, rotary_emb, local_seq, rank, world)
```

### 4. Install the processor

```python
def install_processor(self, pipe, processor) -> int:
    transformer = pipe.transformer
    n = 0
    for block in transformer.blocks:                  # adapt path
        block.attn1.processor = processor             # adapt attribute
        n += 1
    return n
```

### 5. Register the adapter

In `lvsa/adapters/__init__.py`:
```python
def get_adapter(name: str) -> ModelAdapter:
    if name == "<your_model>":
        from .<your_model> import YourModelAdapter
        return YourModelAdapter()
    ...
```

### 6. Write an example script

```bash
cp examples/wan_generate.py examples/<your_model>_generate.py
```

Edit:
- Import your model's pipeline class
- Replace `WanAdapter` references with your adapter
- Adjust default `--num-frames` to the model's training horizon

### 7. Add a CPU smoke test

In `tests/test_adapters.py`, add:
```python
class TestYourModelAdapter:
    def test_geometry(self):
        adapter = YourModelAdapter()
        assert adapter.reference_latent_frames(MockPipe()) == 21
    # ... etc
```

Run:
```bash
pytest tests/test_adapters.py::TestYourModelAdapter -v
```

### 8. (Optional) Wire vllm-omni hook

If your model has special pre-attention behavior in vllm-omni (sequence sharding, custom RoPE), add a hook at `lvsa-vllm-omni/lvsa_vllm_omni/<your_model>_hook.py` modeled on `wan_hook.py` or `hunyuan_hook.py`. Register it in `register.py` and document the env var (`LVSA_<YOUR_MODEL>_HOOK=1`) in `lvsa-vllm-omni/README.md`.

## Validation

After your adapter compiles and tests pass on CPU, run a small GPU smoke:

```bash
python examples/<your_model>_generate.py \
    --model /path/to/your_model_checkpoint \
    --prompt "test" \
    --num-frames <training_horizon> \
    --lvsa --auto-keyframes \
    --output-name smoke.mp4
```

Look for `[LVSA] installed on N blocks` in the log — that confirms the adapter found your attention layers.

## Common pitfalls

- **Wrong `reference_latent_frames`** — leads to silent sparsity at training horizon. Verify against the model card.
- **`patches_per_frame` doesn't divide `seq_len`** — geometry detection fails. Triple-check VAE spatial factor and patch size against the model config.
- **Custom attention layer not found in `install_processor`** — the walk path (`pipe.transformer.blocks[i].attn1`) differs per model. Print the model structure first.
- **Multi-GPU breaks** — CP requires `seq_len % world_size == 0`. If not divisible, add a constraint check at adapter init.
