# Changelog

All notable changes to LVSA will be documented in this file. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Batched-input guard on `Cosmos3LVSAAttnProcessor`** (PR #5 review follow-up).
  The processor mirrors the stock `Cosmos3AttnProcessor`, which flattens batch
  into sequence — only correct for one sample per call. `Cosmos3OmniPipeline`
  guarantees that (sequential CFG), but a batched (`B>1`) input now raises a
  clear `NotImplementedError` instead of silently cross-contaminating samples.

- **Standalone Cosmos 3.0 LVSA (experimental).** A diffusers, single-GPU path for
  NVIDIA Cosmos 3.0, independent of the vLLM-Omni plugin:
  - `lvsa/cosmos3.py` — geometry helpers (`cosmos3_latent_frames`,
    `cosmos3_patches_per_frame`, `COSMOS3_REFERENCE_LATENT_FRAMES=48`), the
    `Cosmos3LVSAAttnProcessor` (LVSA on the video `gen` pathway only; the text/VLM
    `und` causal path is byte-identical to diffusers' stock processor), and
    `install_cosmos3_lvsa(transformer, num_frames, height, width, ...)` which swaps
    the processor onto every `transformer.layers[i].self_attn`.
  - `examples/cosmos_generate.py` — end-to-end generation (`--lvsa`, `out/adhoc/`).
  - Engages via a **processor swap**, not the `ModelAdapter` ABC — Cosmos's
    separate-stream asymmetric attention doesn't fit the ABC. Needs **diffusers
    main** (`>=0.39.0.dev0`) for `Cosmos3OmniPipeline`. MVP: single-GPU, SDPA,
    fixed keyframes (FlashInfer + Ulysses are follow-ups).
  - Correctness validated: `tests/test_cosmos3_processor.py` (geometry,
    **1×==dense equivalence** vs the real diffusers `Cosmos3AttnProcessor`,
    sparse-engagement, installer-swap) + a 1-step GPU smoke across dense / 1× / 2×.
- **`build_global_kv`** hoisted into the model-agnostic core
  (`lvsa/sparse_attention.py`; previously only the plugin had a copy).

### Fixed

- `lvsa-vllm-omni/examples/offline_lvsa.py` raises a clear `RuntimeError` when the
  pipeline returns no frames (`result.images is None`, or 0 frames after
  conversion) instead of an opaque `AttributeError`.
- `examples/cosmos_generate.py` passes `enable_safety_checker=False` to
  `from_pretrained` so the pipeline doesn't construct `CosmosSafetyChecker` at load
  (which requires the external `cosmos_guardrail` package).

### Changed

- Docs updated for the Cosmos standalone path and the processor-swap pattern:
  `README.md` (supported models, examples), `examples/README.md`,
  `docs/architecture.md` (new "when a model doesn't fit the ABC" section),
  `docs/quickstart.md`.
- Skills bumped: `lvsa-quickstart` 1.2.0 → 1.3.0, `lvsa-add-model` 1.0.0 → 1.1.0
  (processor-swap path), `lvsa-vllm-omni` 1.3.0 → 1.4.0 (standalone pointer).
- **Deduped `build_global_kv`.** The plugin's `lvsa_vllm_omni/global_kv.py` now
  re-exports `build_global_kv` from the core (`lvsa.sparse_attention`) instead of
  carrying a byte-identical copy; existing `from lvsa_vllm_omni.global_kv import
  build_global_kv` imports (Wan / HunyuanVideo / Cosmos hooks, plugin tests) are
  unchanged.

## [1.2.0] — 2026-06-08

### Fixed

- **vLLM-Omni plugin: ~30 GB memory overhead eliminated.** The FlashInfer LVSA
  backend instantiated a runner *per attention layer*, each permanently caching
  a 128 MB workspace + compact-K/V scratch (~1 GB/layer → ~30 GB on a 40-layer
  model). A process-wide shared runner (`flashinfer_runner.get_shared_runner()`)
  is now reused across all layers. Plugin peak (Wan2.1-14B, 1×, 480p, FlashInfer)
  drops **74.3 → 44.0 GB**, matching the diffusers standalone; HunyuanVideo 1.5
  plugin at 1× goes from OOM to fitting (~61 GB).
- **Wan2.2-TI2V-5B LVSA geometry.** Both paths mishandled the model's new
  high-compression VAE (spatial 16 / temporal 4): the Wan adapter's default
  reference horizon (21) treated the native 121-frame 1× as a 1.48× *extension*
  (over-sparsifying at 1×), and the plugin's geometry detection lacked the 720p
  patches-per-frame (880, vs the 480p default 1560) → silently fell back to
  dense. Fixed via the new reference-horizon / patches-per-frame overrides below.

### Added

- **`lvsa-vllm-omni/examples/offline_lvsa.py`** — a single offline runner for all
  model families (`--family {wan,hunyuan,cosmos}`), superseding the per-family
  `offline_wan.py` / `offline_hunyuan.py`. Attention path via
  `--backend {flashinfer,sdpa,dense}` (`dense` = no-LVSA baseline); knobs
  `--sparsity-scale`, `--ref-lat`, `--patches-per-frame`, `--no-rotate`,
  `--offload`, `--eager`.
- **`--reference-latent-frames`** on `examples/wan_generate.py` and a matching
  `reference_latent_frames` override on `install_lvsa_processors`
  (`lvsa/parallel.py`) — set the training horizon in latent frames (e.g. 31 for
  Wan2.2-TI2V-5B vs the adapter default 21).
- `flashinfer_runner.get_shared_runner()` — process-wide shared FlashInfer LVSA runner.
- Cosmos 3.0 run guidance in the `lvsa-vllm-omni` skill (plugin-only; offline +
  cross-attention hook; vllm-omni-main venv, 720p, ~400-frame single-shot cap).

### Changed

- HunyuanVideo plugin horizon ceiling is **~1.5×**: vLLM-Omni runs the HV 3D VAE
  decode in fp32 (~+14 GB vs the bf16 standalone), so the plugin maxes at ~80 GB
  by 1.5×. The diffusers standalone carries HV ≥2× (ceiling ~3×).
- `lvsa-vllm-omni/scripts/integration_sweep.sh` migrated onto `offline_lvsa.py`
  (adds SDPA-backend coverage; `--no-lvsa` → `--backend dense`).
- **Pinned vLLM-Omni to `v0.22.0` stable** (was `v0.22.0rc1`). The pairing is now
  symmetric (`vllm==0.22.0` + `vllm-omni==0.22.0`). The plugin's dependency
  surface is byte-identical between rc1 and v0.22.0 (verified: imports +
  monkeypatch signatures + enum registration smoke-pass), so this is a pin bump
  with no code change. **Cosmos 3 is now included in v0.22.0 stable** — it no
  longer requires a `main` build, so one stable install covers Wan / HunyuanVideo
  / Cosmos.

### Deprecated

- `lvsa-vllm-omni/examples/offline_wan.py` and `offline_hunyuan.py` — superseded
  by `offline_lvsa.py`; slated for removal.

### Removed

- Dead `_build_flashinfer_args` path and its orphaned `_fi_*` scratch buffers in
  `attention_impl.py` (the pre-LSE-merge FlashInfer buffer builder, unused since
  the LSE-merge runner replaced it), plus several unused imports.

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
