# Changelog

All notable changes to LVSA will be documented in this file. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-06-02

### vLLM-Omni plugin — upgraded to vllm-omni 0.22.0rc1

The plugin now targets **vllm-omni 0.22.0rc1** (paired with **vllm 0.22.0**,
torch 2.11 / CUDA 13). The previous `vllm-omni 0.18.0` line is maintained on the
`release/v0.18.x` branch.

**Breaking changes for users:**

- **Backend selection moved to per-role `AttentionConfig`.** vllm-omni 0.22
  removed the `DIFFUSION_ATTENTION_BACKEND` env var. Select LVSA per attention
  role instead — on the CLI:
  `--diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'`,
  or via the Python API:
  `Omni(..., diffusion_attention_config={"per_role": {"self": {"backend": "LVSA"}}})`.
  The `python -m lvsa_vllm_omni.serve` wrapper and the `offline_*.py` /
  `serve_*.sh` examples inject this for you.
- **Install is asymmetric and source-built.** vllm-omni 0.22.0rc1 is a
  pre-release not published to PyPI; install `vllm==0.22.0` from PyPI first,
  then build vllm-omni from the git tag
  (`vllm-omni @ git+…@v0.22.0rc1`). The versions intentionally differ.
- **Both Docker images now build on `nvidia/cuda:13.0.0`** (torch 2.11/2.12,
  flashinfer 0.6.11.post2).

**Internal changes:**

- `wan_hook.py`: `apply_rotary_emb_wan` (removed upstream) replaced by
  `RotaryEmbeddingWan`; patched-forward signature updated to the new
  `(hidden_states, rotary_emb, attn_metadata)`; dense fallback delegates to the
  original forward.
- `hunyuan_hook.py`: patched-forward gained `hidden_states_mask`; dense fallback
  delegates to the original forward (SP-aware).
- `attention_impl.py`: `LVSAAttentionImpl.__init__` accepts the new
  `qkv_layout` / `backend_kwargs` parameters.
- Enum registration via `aenum.extend_enum` continues to work against the 0.22
  `DiffusionAttentionBackendEnum`.

## [1.0.0] — 2026-05-27

Initial public release.

### Core algorithm

- Block-sparse attention with rotating keyframes, expanded window bounds, auto-keyframe scheduling.
- Two backends: SDPA (default; runs on CUDA + Ascend NPU via `torch_npu`) and FlashInfer (block-sparse CSR; CUDA-only).
- Single-GPU and multi-GPU (Ulysses) context-parallel via standard PyTorch distributed primitives.
- Optional [RIFLEx](https://arxiv.org/abs/2502.15894) RoPE rescaling for additional extrapolation headroom — composable with LVSA.

### Models

- **Stable**: Wan 2.1 (1.3B, 14B), Wan 2.2 (T2V-A14B, TI2V-5B), HunyuanVideo 1.5.
- **Experimental**: CogVideoX 5B (correctness only — no speedup due to joint-attention shared-QKV layout).

### vLLM-Omni plugin

- `LVSABackend` for HunyuanVideo (works through the generic attention plugin path).
- `wan_hook` for Wan 2.x (intercepts before vLLM-Omni's `_sp_plan` shards the sequence).
- `LVSAConfig` with environment-variable parsing (`LVSA_*` env vars), including geometry overrides (`LVSA_PATCHES_PER_FRAME`, `LVSA_VIDEO_HEIGHT`, `LVSA_VIDEO_WIDTH`, `LVSA_VAE_SPATIAL_FACTOR`, `LVSA_PATCH_SIZE`, `LVSA_VAE_TEMPORAL_FACTOR`) for non-standard resolutions.

### Configuration safety improvements

- `compute_auto_kfi` short-circuits to `kfi=1` when `T_lat ≤ reference_frames` to guarantee fully-dense attention at training reference.
- `reference_frames` is propagated through all call sites (`LVSAMetadata.build`, `DistributedLVSAProcessor.__init__`, `set_window_size`, `set_sparsity_scale`, `_rebuild_for_current_params`, plus vllm-omni hook paths).
- `LVSAConfig.reference_latent_frames` field with `LVSA_REFERENCE_LATENT_FRAMES` env-var support.
- Default `LVSA_SCHEDULE_START=0 LVSA_SCHEDULE_END=0` (graduated schedule disabled by default; users opt in by setting `> 0`). Soft-deprecated in favor of `sparsity_scale`.

### Bundled subpackages

- [`lvsa-vllm-omni/`](lvsa-vllm-omni/) — vLLM-Omni serving plugin.
- [`vqeval/`](vqeval/) — companion video-quality benchmark suite (composite + 6 dimensions: spatial, temporal, loop, artifacts, dynamic, text-alignment).

### Documentation

- README with install, quickstart SotA numbers, citation.
- `docs/install.md`, `docs/quickstart.md`, `docs/tuning.md`, `docs/troubleshooting.md`, `docs/architecture.md`, `docs/VLLM_OMNI_INTEGRATION.md`.
- `lvsa-vllm-omni/README.md` documents all `LVSA_*` environment variables.
- `benchmarks/README.md` describes the paper-reproduction recipe.
- Companion Claude Code skills shipped under [`skills/`](skills/).

### Tests

- CPU-only test suite covering windowed attention primitives, processor wiring, adapter contracts, device-detection helpers, RoPE math, RIFLEx, and the `reference_frames` propagation invariants.
- VQeval test suite at `vqeval/tests/`.

[1.1.0]: https://github.com/JiusiServe/LongVideoSparseAttention/releases/tag/v1.1.0
[1.0.0]: https://github.com/JiusiServe/LongVideoSparseAttention/releases/tag/v1.0.0
