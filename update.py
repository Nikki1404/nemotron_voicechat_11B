(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_voicechat_11B# docker run --rm -it   --gpus all   --ipc=host   --shm-size=8g  -e MIN_GPU_VRAM_GB=0 -p 8001:8001   nemotron-voicechat:latest

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/opt/conda/lib/python3.12/site-packages/torch/cuda/__init__.py:65: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
/opt/conda/lib/python3.12/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (1.26.20) or chardet (6.0.0.post1)/charset_normalizer (3.3.2) doesn't match a supported version!
  warnings.warn(
[NeMo I 2026-08-20 08:02:49 nemo_logging:394] Triton available & CUDA detected. Using Triton kernel for batch_matmul.
[NeMo W 2026-08-20 08:02:49 nemo_logging:406] /opt/Speech/nemo/collections/speechlm2/parts/optim_setup.py:93: SyntaxWarning: invalid escape sequence '\.'
      ... params = freeze_and_subset(model.named_parameters(), ['^llm\..+$'])

INFO:     Started server process [1]
INFO:     Waiting for application startup.
==========================================================================================
NVIDIA NemotronLabs VoiceChat 11B
==========================================================================================
MODEL_ID        : nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
MODEL_PATH      : /app/models/NVIDIA-NemotronLabs-VoiceChat-11B
DEVICE          : cuda
CUDA available  : True
GPU             : NVIDIA A10G
GPU VRAM        : 22.09 GB
Torch version   : 2.10.0+cu128
Torch CUDA      : 12.8

Loading Nemotron VoiceChat model...
This can take significant time for the 11B checkpoint.

config.json: 1.51kB [00:00, 7.41MB/s]
configuration_nemotron_h.py: 12.2kB [00:00, 47.5MB/s]
A new version of the following files was downloaded from https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2:
- configuration_nemotron_h.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
modeling_nemotron_h.py: 79.0kB [00:00, 8.97MB/s]
A new version of the following files was downloaded from https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2:
- modeling_nemotron_h.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
`torch_dtype` is deprecated! Use `dtype` instead!
