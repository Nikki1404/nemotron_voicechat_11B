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

# System/runtime dependencies plus CUDA build tooling needed by
# causal-conv1d and mamba-ssm.
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

# NVIDIA's published VoiceChat setup uses Python 3.12.
RUN curl -fsSL \
      https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh \
      -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -f /tmp/miniconda.sh \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /opt

# NVIDIA's dedicated NemotronLabs VoiceChat branch.
RUN git clone \
    --branch nemotron-labs-voicechat \
    --depth 1 \
    https://github.com/NVIDIA-NeMo/Speech.git \
    /opt/Speech

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install the complete server/model Python stack from requirements.txt.
# --no-build-isolation lets the CUDA extensions build against installed Torch.
RUN python -m pip install \
      --no-build-isolation \
      -r /app/requirements.txt \
    && python -m pip uninstall -y nvidia-resiliency-ext || true

# Download the Hugging Face checkpoint into the Docker image.
# No Hugging Face token is used.
RUN mkdir -p ${MODEL_DIR} \
    && hf download ${MODEL_ID} \
       --local-dir ${MODEL_DIR}

COPY server.py /app/server.py

EXPOSE 8000

CMD ["python", "server.py"]
