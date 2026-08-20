 => [ 8/10] RUN python -m pip install       --no-build-isolation       -r /app/requirements.txt     && python -m pip uninstall -y nvidia-resiliency-ext || true                                            4.2s
 => ERROR [ 9/10] RUN mkdir -p /app/models/NVIDIA-NemotronLabs-VoiceChat-11B     && hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B        --local-dir /app/models/NVIDIA-NemotronLabs-VoiceChat-11B  0.3s
------
 > [ 9/10] RUN mkdir -p /app/models/NVIDIA-NemotronLabs-VoiceChat-11B     && hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B        --local-dir /app/models/NVIDIA-NemotronLabs-VoiceChat-11B:
0.234 /bin/sh: 1: hf: not found
------
ERROR: failed to build: failed to solve: process "/bin/sh -c mkdir -p ${MODEL_DIR}     && hf download ${MODEL_ID}        --local-dir ${MODEL_DIR}" did not complete successfully: exit code: 127
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_voicechat_11B#
