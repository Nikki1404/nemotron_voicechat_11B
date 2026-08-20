# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ARG DEBIAN_FRONTEND=noninteractive
ARG MODEL_ID=nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
ARG MODEL_DIR=/app/models/NVIDIA-NemotronLabs-VoiceChat-11B

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    MODEL_ID=${MODEL_ID} \
    MODEL_PATH=${MODEL_DIR} \
    DEVICE=cuda \
    NEMO_DIR=/opt/Speech \
    PYTHONPATH=/opt/Speech \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/opt/conda/bin:/usr/local/cuda-12.4/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    build-essential \
    ninja-build \
    cuda-toolkit-12-4 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# ---------------------------------------------------------------------------
# Python 3.12 environment
# ---------------------------------------------------------------------------

RUN curl -fsSL \
      https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh \
      -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -f /tmp/miniconda.sh \
    && python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Clone NVIDIA VoiceChat branch
# ---------------------------------------------------------------------------

WORKDIR /opt

RUN git clone \
    --branch nemotron-labs-voicechat \
    --depth 1 \
    https://github.com/NVIDIA-NeMo/Speech.git \
    /opt/Speech

WORKDIR /opt/Speech

# ---------------------------------------------------------------------------
# Install Torch first
#
# Torch is intentionally NOT in requirements.txt.
# causal-conv1d and mamba-ssm import Torch during build/metadata generation.
# ---------------------------------------------------------------------------

RUN python -m pip install \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio==2.10.0

RUN python -c "\
import torch; \
print('========================================'); \
print('Torch before NeMo:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('========================================')"

# ---------------------------------------------------------------------------
# Install NVIDIA NeMo VoiceChat stack
# ---------------------------------------------------------------------------

RUN python -m pip install -e ".[all]"

# This dependency is not required for our standalone app.
RUN python -m pip uninstall -y nvidia-resiliency-ext || true

# ---------------------------------------------------------------------------
# Force Torch versions again AFTER NeMo
#
# NeMo dependency resolution may install/change Torch-related packages.
# This guarantees the final Torch stack we want.
# ---------------------------------------------------------------------------

RUN python -m pip install \
    --upgrade \
    --force-reinstall \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio==2.10.0

RUN python -c "\
import torch, torchvision, torchaudio; \
print('========================================'); \
print('Torch after NeMo:', torch.__version__); \
print('Torchvision:', torchvision.__version__); \
print('Torchaudio:', torchaudio.__version__); \
print('Torch CUDA:', torch.version.cuda); \
print('========================================')"

# ---------------------------------------------------------------------------
# Application dependencies
# ---------------------------------------------------------------------------

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    -r /app/requirements.txt

# Make sure Torch is still available before building CUDA extensions.
RUN python -c "\
import torch; \
print('Torch before CUDA extensions:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda)"

# ---------------------------------------------------------------------------
# CUDA extensions
#
# Keep these OUT of requirements.txt.
# They need Torch already installed and must build against the current env.
# ---------------------------------------------------------------------------

RUN python -m pip install \
    --no-build-isolation \
    --no-deps \
    causal-conv1d==1.6.2.post1 \
    mamba-ssm==2.3.2.post1

# ---------------------------------------------------------------------------
# Final dependency verification
# ---------------------------------------------------------------------------

RUN python -c "\
import torch; \
import torchvision; \
import torchaudio; \
import huggingface_hub; \
import causal_conv1d; \
import mamba_ssm; \
print('========================================'); \
print('FINAL ENVIRONMENT'); \
print('========================================'); \
print('torch:', torch.__version__); \
print('torchvision:', torchvision.__version__); \
print('torchaudio:', torchaudio.__version__); \
print('torch CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('huggingface_hub:', huggingface_hub.__version__); \
print('causal_conv1d: OK'); \
print('mamba_ssm: OK'); \
print('========================================')"

# ---------------------------------------------------------------------------
# Download Nemotron VoiceChat model during Docker build
#
# No Hugging Face token.
# Use snapshot_download instead of relying on the `hf` CLI.
# ---------------------------------------------------------------------------

RUN mkdir -p ${MODEL_DIR} \
    && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='${MODEL_ID}', \
    local_dir='${MODEL_DIR}' \
)"

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

WORKDIR /app

COPY server.py /app/server.py

EXPOSE 8000

CMD ["python", "server.py"]
