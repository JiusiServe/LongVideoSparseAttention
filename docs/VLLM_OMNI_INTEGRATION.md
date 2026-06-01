# LVSA × vLLM-Omni Integration

How the LVSA plugin slots into [vLLM-Omni](https://github.com/vllm-project/vllm-omni). This is a usage-level guide; for the full env-var reference see [`../lvsa-vllm-omni/README.md`](../lvsa-vllm-omni/README.md).

---

## At a glance

`lvsa-vllm-omni` is a separate pip package that registers itself as a vLLM-Omni attention backend. **Zero changes required in vLLM-Omni core** — everything plugs in through the plugin entry-point system.

| LVSA Concept | vllm-omni Equivalent |
|---|---|
| `DistributedLVSAProcessor.__call__` | `AttentionImpl.forward_cuda()` |
| `ModelAdapter` (QKV, RoPE, output) | Already handled by vllm-omni transformer code |
| FlashInfer / SDPA dispatch | `DiffusionAttentionBackendEnum` registry |
| `install_lvsa_processors()` | `register_diffusion_backend()` + env var |
| `set_step()` / `set_window_size()` | `DiffusionAttentionMetadata` per-step |
| Ulysses CP | `ParallelAttentionStrategy` Protocol |

## Package layout

```
lvsa-vllm-omni/
├── lvsa_vllm_omni/
│   ├── __init__.py                # Entry-point plugin registration
│   ├── backend.py                 # LVSABackend
│   ├── attention_impl.py          # LVSAAttentionImpl — calls sparse_windowed_attention()
│   ├── wan_hook.py                # Wan 2.x — intercepts before `_sp_plan` shards the seq
│   ├── hunyuan_hook.py            # HunyuanVideo 1.5 — installs processor + RoPE patches
│   ├── register.py                # Auto-installs hooks based on env vars
│   ├── config.py                  # LVSAConfig dataclass (parses LVSA_* env vars)
│   ├── global_kv.py               # Helpers for global K/V gather
│   ├── step_tracker.py            # Per-step state (used by keyframe rotation)
│   └── _fallback.py               # Silent-fallback warnings
└── tests/                         # CPU-only integration tests
```

## How it activates

```bash
pip install -e lvsa-vllm-omni/
pip install "vllm==0.18.0" "vllm-omni==0.18.0"   # validated pair

# Enable for HunyuanVideo
DIFFUSION_ATTENTION_BACKEND=LVSA \
LVSA_AUTO_KEYFRAMES=1 \
LVSA_REFERENCE_LATENT_FRAMES=33 \
vllm serve --omni --model HunyuanVideo-1.5-Diffusers-480p_t2v

# Enable for Wan (also needs the Wan hook explicit)
DIFFUSION_ATTENTION_BACKEND=LVSA \
LVSA_WAN_HOOK=1 \
LVSA_AUTO_KEYFRAMES=1 \
LVSA_REFERENCE_LATENT_FRAMES=21 \
vllm serve --omni --model Wan2.2-T2V-14B
```

The plugin entry point in `pyproject.toml` registers `lvsa_vllm_omni:register` under `vllm_omni.general_plugins`, which fires at vLLM-Omni startup.

## Core call path

```python
class LVSAAttentionImpl(AttentionImpl):
    """vLLM-Omni attention backend that dispatches to LVSA sparse attention."""

    def forward_cuda(
        self,
        query: torch.Tensor,       # [B, seq, H, D]
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: LVSAAttentionMetadata,
    ) -> torch.Tensor:
        return sparse_windowed_attention(query, key, value, attn_metadata)
```

QKV extraction and RoPE are already done by vLLM-Omni's transformer code; the plugin only replaces the attention dispatch.

## Per-step metadata

```python
@dataclass
class LVSAAttentionMetadata:
    total_latent_frames: int
    patches_per_frame: int
    window_size: int
    n_first_frames: int
    key_frame_interval: int
    step_idx: int                    # For rotating keyframes
    total_steps: int
    rotate_keyframes: bool
    kv_cache_active: bool
    expand_window: bool
```

Built per denoising step. Step index is auto-detected from vLLM-Omni's `denoise_step()` call sequence.

## Configuration

```python
@dataclass
class LVSAConfig:
    window_size: int = 32            # In video frames
    n_first_frames: int = 4
    auto_keyframes: bool = True      # Auto-compute KFI from frame count
    rotate_keyframes: bool = True    # Shift keyframe grid each step
    expand_window: bool = True
    backend: str = "auto"            # "flashinfer" > "sdpa"
    sparsity_scale: float = 1.0
    reference_latent_frames: int     # MUST be set per-model (21 / 33 / 13)
```

Every field has a `LVSA_<UPPERCASE>` env-var equivalent. See [`../lvsa-vllm-omni/README.md`](../lvsa-vllm-omni/README.md) for the full table.

## Auto-activation

For long-video workloads, the plugin can auto-engage only when generation exceeds the model's training horizon:

```bash
LVSA_AUTO=1 vllm serve --omni --model Wan2.2-T2V-14B
```

At request time, when `num_frames > training_horizon` the LVSA backend takes over; otherwise dense attention runs. Useful for production deployments that mix short and long workloads.

## Geometry overrides

For models running at non-default resolutions, set these so the plugin computes the correct patch count and rotation grid:

| Env var | Default | Meaning |
|---|---|---|
| `LVSA_PATCHES_PER_FRAME` | 1560 | Tokens per latent frame after VAE + patchify |
| `LVSA_VIDEO_HEIGHT` | 480 | Video height in pixels |
| `LVSA_VIDEO_WIDTH` | 832 | Video width in pixels |
| `LVSA_VAE_SPATIAL_FACTOR` | 8 | VAE spatial compression |
| `LVSA_PATCH_SIZE` | 2 | Patchify factor |
| `LVSA_VAE_TEMPORAL_FACTOR` | 4 | VAE temporal compression |

If the geometry detection at runtime fails, the plugin falls back to dense and emits `[LVSA-FALLBACK] origin=forward_cuda reason=geometry_detect ...`.

## Multi-GPU

Standard Ulysses context parallelism. Set `--ulysses-degree N` in the vLLM-Omni CLI. The plugin handles the K/V global gather automatically.

**Constraint**: `seq_len = T_lat × patches_per_frame` must be divisible by `N`.

## Diagnostics

```bash
grep -E "\[LVSA" run.log | head -30
```

Expected lines after a successful run:
```
[LVSA] Step counter: n_blocks=N (from env)
[LVSA-hook] Installed LVSA hook on HunyuanVideo15Attention ...
[LVSA] Geometry detected: T_lat=33 P=1560 text_tokens=512
[LVSA] reference_latent_frames=33 target_latent_frames=33 extension_ratio=1.00x
```

If you see fallback warnings instead, see [`troubleshooting.md`](troubleshooting.md).
