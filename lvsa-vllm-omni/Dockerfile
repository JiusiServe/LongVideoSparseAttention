FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /workdir

ENV TORCH_CUDA_ARCH_LIST="8.0"

RUN apt-get update && apt-get install -y \
        curl wget git nano \
        ffmpeg

# Installing uv
ENV PATH=/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

RUN --mount=type=cache,target=/root/.cache/uv \
        uv venv --python 3.12 /venv --seed

# ── All from PyPI — no source compilation needed ──
RUN --mount=type=cache,target=/root/.cache/uv \
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/uv \
        uv pip install ftfy imageio imageio-ffmpeg

ENV HF_HUB_CACHE="/models"
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

RUN --mount=type=cache,target=/root/.cache/uv  \
    uv pip install "llvmlite>=0.40.0" "numba>=0.57.0"

RUN --mount=type=cache,target=/root/.cache/uv  \
    uv pip install accelerate numpy==1.26.4 pytorch-lightning \
    hf_xet "kernels==0.10.0" ninja wheel psutil
# kernels>=0.13 introduced PEP-604 union types (str | None) in its dataclasses,
# which huggingface_hub 0.36.x StrictDataclass validation rejects. Pin to the
# last compatible version. Revisit when bumping vllm/vllm-omni.

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[test]"

# ── vllm-omni + vllm from PyPI (must pin matching versions) ──
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "vllm==0.18.0" "vllm-omni==0.18.0"

# ── NO flash_attn — torch SDPA dispatches to FA2 internally ──

# ── Install lvsa + lvsa-vllm-omni from local source ──
COPY . /workdir/code
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e /workdir/code -e /workdir/code/lvsa-vllm-omni

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "flashinfer-python==0.6.6" "flashinfer-cubin==0.6.6"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "flashinfer-jit-cache==0.6.6" --index-url https://flashinfer.ai/whl/cu128

# LVSA registers via the `vllm_omni.general_plugins` entry point (pyproject.toml).
# vllm-omni calls `load_omni_general_plugins()` in the main process, engine,
# AND every diffusion worker, so the entry point alone is sufficient.
# `register_lvsa_backend()` also triggers the optional monkey-patches
# (ring processor, Hunyuan hook, Wan hook) — each env-var guarded.

RUN echo 'source /venv/bin/activate' >> /root/.bashrc

WORKDIR /workdir/code
