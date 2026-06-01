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

# torch pinned to 2.8.x for CUDA 12.8 compatibility (base image is cuda:12.8.0).
# The package itself declares torch>=2.1.0 in pyproject.toml; this Docker image
# ships a specific CUDA-matched build.
RUN --mount=type=cache,target=/root/.cache/uv \
        uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/uv \
        uv pip install ftfy imageio imageio-ffmpeg

# ENV HF_HOME="/models"
ENV HF_HUB_CACHE="/models"
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

# Install compatible versions of numba/llvmlite for Python 3.10+
RUN --mount=type=cache,target=/root/.cache/uv  \
    uv pip install "llvmlite>=0.40.0" \
    "numba>=0.57.0"

# Extra dependencies
RUN --mount=type=cache,target=/root/.cache/uv  \
    uv pip install accelerate \
    numpy==1.26.4 \
    pytorch-lightning \
    hf_xet \
    kernels \
    ninja wheel

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@main#egg=diffusers[test]"

# HunyuanVideo-1.5 text-encoder dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install sentencepiece qwen-vl-utils

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flash_attn --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flashinfer-python flashinfer-cubin

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128

RUN echo 'source /venv/bin/activate' >> /root/.bashrc