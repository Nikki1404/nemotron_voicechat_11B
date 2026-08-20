# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

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
    PYTHONPATH=/opt/Speech

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    git git-lfs curl ffmpeg libsndfile1 \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /opt

# NVIDIA's dedicated VoiceChat branch.
RUN git clone --branch nemotron-labs-voicechat --depth 1 \
    https://github.com/NVIDIA-NeMo/Speech.git /opt/Speech

WORKDIR /opt/Speech

# Versions are intentionally aligned to NVIDIA's VoiceChat model-card instructions.
RUN python3 -m pip install --break-system-packages \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    && python3 -m pip install --break-system-packages -e ".[all]" \
    && python3 -m pip uninstall --break-system-packages -y nvidia-resiliency-ext || true \
    && python3 -m pip install --break-system-packages \
      transformers==4.56.0 tokenizers==0.22.0 lhotse==1.32.2 \
      huggingface-hub==0.34.4 hf-xet==1.1.9 torchcodec==0.10.0 \
      torch_audiomentations jinja2 ninja packaging wheel einops \
      fastapi "uvicorn[standard]" python-multipart soundfile websockets requests sounddevice \
    && python3 -m pip install --break-system-packages \
      --no-build-isolation --no-deps \
      causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1

# Download the Hugging Face checkpoint directly into the Docker image.
# No Hugging Face token is used by this project.
RUN mkdir -p ${MODEL_DIR} && \
    hf download ${MODEL_ID} --local-dir ${MODEL_DIR}

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000

CMD ["python3", "server.py"]
