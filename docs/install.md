# Installation

## Requirements

- **Linux** (Ubuntu 20.04+ recommended)
- **CUDA** 12.1+ (12.8 tested), **or** Ascend NPU with `torch_npu`
- **Python** 3.10+
- **PyTorch** 2.5+ with CUDA (or `torch_npu`) support
- A GPU with ≥ 24 GB VRAM (Wan 1.3B), ≥ 80 GB recommended (Wan 14B / HunyuanVideo)

## Option 1 — pip from source (development install)

```bash
git clone https://github.com/JiusiServe/LongVideoSparseAttention
cd LVSA

# Create a virtual environment (uv recommended; venv works too)
uv venv --python 3.12
source .venv/bin/activate

# Core library
uv pip install -e .

# vLLM-Omni plugin (only needed for serving)
uv pip install -e lvsa-vllm-omni/
uv pip install "vllm==0.18.0" "vllm-omni==0.18.0"   # validated pair

# Video-quality assessment (only needed for benchmarks/eval)
uv pip install -e vqeval/

# Verify
python -c "import lvsa; print(lvsa.__file__)"
python -c "from lvsa.adapters.wan import WanAdapter; print('OK')"
```

### Optional extras

```bash
# All attention backends + model deps
uv pip install -e ".[all]"

# Just FlashInfer (fast block-sparse on CUDA)
uv pip install -e ".[flashinfer]"

# Dev tools (pytest)
uv pip install -e ".[dev]"
```

## Option 2 — Docker (reproducible)

Two paths.

### A. Pre-built image (vLLM-Omni serving)

```bash
docker pull lvsa-vllm-omni:latest
```

The image contains:
- vLLM 0.18.0 + vLLM-Omni
- LVSA installed editably at `/workdir/code` (mount your repo to override)
- diffusers 0.38 with HunyuanVideo and Wan pipelines
- FlashInfer kernels precompiled

### B. Build your own

```bash
# Top-level Dockerfile builds an image with LVSA + diffusers (no vLLM-Omni)
docker build -t lvsa:latest -f Dockerfile .

# vLLM-Omni serving image
docker build -t lvsa-vllm-omni:latest -f lvsa-vllm-omni/Dockerfile lvsa-vllm-omni/
```

## Backends

LVSA supports two attention backends. They're selected per-run via CLI / env var:

| Backend | When to use | Install | Hardware |
|---|---|---|---|
| **SDPA** (default) | Always available; reasonable on short sequences | included with PyTorch | CUDA, Ascend NPU, CPU |
| **FlashInfer** | Fastest at `T_lat ≥ 49` (HunyuanVideo 1.5×+, Wan 2×+) | `pip install flashinfer-python flashinfer-cubin` | CUDA only |

If your install doesn't include FlashInfer, runs fall back to SDPA with a warning.

### Ascend NPU note

LVSA's SDPA path runs on Ascend NPU through `torch_npu`. The device helpers in `lvsa/device.py` auto-detect NPU and route memory probes / device placement appropriately. Distributed runs use `hccl` instead of `nccl`. The FlashInfer backend is CUDA-only and will fall back to SDPA on NPU.

Custom NPU-native kernels (single-call `npu_fusion_attention`, `npu_block_sparse_attention`) are not part of v1.0 — only the SDPA path is exercised on NPU. They may return in a future release.

## Verifying the install

```bash
# Run the unit tests (CPU-only, no model weights needed)
pytest tests/ lvsa-vllm-omni/tests/ vqeval/tests/ -v
# Should report 230+ tests passing
```

For an actual generation smoke test (requires a model checkpoint), see [`quickstart.md`](quickstart.md).

## Model weights

LVSA does not ship model weights. Download them separately:

| Model | HuggingFace |
|---|---|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Wan 2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| HunyuanVideo 1.5 480p T2V | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` |
| CogVideoX-5B | `THUDM/CogVideoX-5b` |

Each downloads under its own license — see the model cards.

## Troubleshooting installation

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'lvsa'` | re-run `pip install -e .` from the repo root |
| `flashinfer not found` warning | install `flashinfer-python flashinfer-cubin`, or rely on SDPA |
| FlashInfer install fails | use the SDPA backend (default); FlashInfer is optional |
| `OSError: libcuda.so.1` | not a CUDA-enabled environment; install NVIDIA drivers + CUDA runtime, OR use Ascend NPU via `torch_npu` |
| `torch_npu` not importable on Ascend | install Ascend's `torch_npu` matching your CANN + PyTorch version |

For runtime issues, see [`troubleshooting.md`](troubleshooting.md).
