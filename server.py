import asyncio
import base64
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, Response

from nemo.collections.speechlm2.inference.utils.offline_voicechat import (
    TARGET_SR,
    SOURCE_SR,
    build_model,
    encode_system_prompt,
    load_wav_16k_mono,
    run_offline_inference,
)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/models/NVIDIA-NemotronLabs-VoiceChat-11B",
)

DEVICE = os.getenv("DEVICE", "cuda")

HOST = "0.0.0.0"
PORT = 8001

# API-facing sample rate
CLIENT_SAMPLE_RATE = 24000

# Nemotron internal sample rates
MODEL_INPUT_SR = SOURCE_SR
MODEL_OUTPUT_SR = TARGET_SR

OUTPUT_CHUNK_MS = int(os.getenv("OUTPUT_CHUNK_MS", "80"))

# 11B model: keep concurrency low by default
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

# Fail early rather than hanging while attempting to load an obviously
# undersized GPU. You may override this if testing another loading strategy.
MIN_GPU_VRAM_GB = float(os.getenv("MIN_GPU_VRAM_GB", "40"))


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Answer naturally in a spoken conversational style. "
    "Be concise and helpful."
)


# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI(
    title="NVIDIA NemotronLabs VoiceChat 11B Speech-to-Speech API",
    version="1.0.0",
)


model = None
model_load_ms = None

inference_lock = asyncio.Semaphore(MAX_CONCURRENT)


# =============================================================================
# HELPERS
# =============================================================================


def now_ms():
    return time.perf_counter() * 1000.0


def make_event(event_type: str, **kwargs):
    return {
        "type": event_type,
        "event_id": str(uuid.uuid4()),
        **kwargs,
    }


def pcm16_bytes_to_wav_file(
    pcm_bytes: bytes,
    sample_rate: int = CLIENT_SAMPLE_RATE,
):
    """
    Convert raw mono PCM16 little-endian audio to a temporary WAV.
    """

    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]

    audio = np.frombuffer(
        pcm_bytes,
        dtype="<i2",
    ).astype(np.float32)

    audio /= 32768.0

    fd, path = tempfile.mkstemp(
        suffix=".wav",
    )

    os.close(fd)

    sf.write(
        path,
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    return path


def upload_to_temp_file(
    data: bytes,
    suffix: str = ".wav",
):
    fd, path = tempfile.mkstemp(
        suffix=suffix,
    )

    os.close(fd)

    Path(path).write_bytes(data)

    return path


def tensor_audio_to_pcm16_24k(
    audio_tensor: torch.Tensor,
    audio_len: int,
):
    """
    Nemotron outputs audio at MODEL_OUTPUT_SR.
    Convert to API-facing PCM16 24 kHz.
    """

    audio = (
        audio_tensor[:audio_len]
        .detach()
        .float()
        .cpu()
        .unsqueeze(0)
    )

    if MODEL_OUTPUT_SR != CLIENT_SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio,
            MODEL_OUTPUT_SR,
            CLIENT_SAMPLE_RATE,
        )

    audio = audio.squeeze(0)

    audio = audio.clamp(
        -1.0,
        1.0,
    )

    pcm = (
        audio.numpy() * 32767.0
    ).astype("<i2")

    return pcm.tobytes()


def tensor_audio_to_wav_bytes(
    audio_tensor: torch.Tensor,
    audio_len: int,
):
    audio = (
        audio_tensor[:audio_len]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        MODEL_OUTPUT_SR,
        format="WAV",
        subtype="PCM_16",
    )

    return buffer.getvalue()


# =============================================================================
# MODEL INFERENCE
# =============================================================================


def _infer_sync(
    wav_path: str,
    system_prompt: str,
):
    """
    Blocking Nemotron inference function.
    Runs inside asyncio.to_thread().
    """

    global model

    preprocess_start = now_ms()

    (
        wav_1d,
        input_signal,
        input_signal_lens,
    ) = load_wav_16k_mono(
        wav_path,
        device=DEVICE,
    )

    (
        prompt_tokens,
        prompt_token_lens,
    ) = encode_system_prompt(
        model,
        system_prompt,
        device=DEVICE,
    )

    prep_ms = (
        now_ms()
        - preprocess_start
    )

    inference_start = now_ms()

    result = run_offline_inference(
        model,
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
    )

    inference_ms = (
        now_ms()
        - inference_start
    )

    text = result.get(
        "text",
        [""],
    )[0]

    if result.get("audio") is None:
        raise RuntimeError(
            "Nemotron VoiceChat returned no audio."
        )

    audio_len = int(
        result["audio_len"][0].item()
    )

    audio = result["audio"][0]

    return {
        "text": text,
        "audio": audio,
        "audio_len": audio_len,
        "prep_ms": prep_ms,
        "inference_ms": inference_ms,
        "input_duration_s": (
            float(wav_1d.shape[0])
            / MODEL_INPUT_SR
        ),
        "output_duration_s": (
            float(audio_len)
            / MODEL_OUTPUT_SR
        ),
    }


async def infer(
    wav_path: str,
    system_prompt: str,
):
    async with inference_lock:
        return await asyncio.to_thread(
            _infer_sync,
            wav_path,
            system_prompt,
        )


# =============================================================================
# STARTUP
# =============================================================================


@app.on_event("startup")
async def startup_event():
    global model
    global model_load_ms

    print("=" * 90)
    print(
        "NVIDIA NemotronLabs VoiceChat 11B"
    )
    print("=" * 90)

    print(
        f"MODEL_ID        : {MODEL_ID}"
    )
    print(
        f"MODEL_PATH      : {MODEL_PATH}"
    )
    print(
        f"DEVICE          : {DEVICE}"
    )

    if DEVICE.startswith("cuda"):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no CUDA GPU "
                "is visible inside the container."
            )

        gpu_name = torch.cuda.get_device_name(
            0
        )

        props = (
            torch.cuda.get_device_properties(
                0
            )
        )

        total_vram_gb = (
            props.total_memory
            / (1024 ** 3)
        )

        print(
            "CUDA available  : True"
        )

        print(
            f"GPU             : {gpu_name}"
        )

        print(
            f"GPU VRAM        : "
            f"{total_vram_gb:.2f} GB"
        )

        print(
            f"Torch version   : "
            f"{torch.__version__}"
        )

        print(
            f"Torch CUDA      : "
            f"{torch.version.cuda}"
        )

        if total_vram_gb < MIN_GPU_VRAM_GB:

            raise RuntimeError(
                "\n"
                "=" * 90
                + "\n"
                "INSUFFICIENT GPU MEMORY\n"
                + "=" * 90
                + "\n"
                f"Detected GPU : {gpu_name}\n"
                f"Detected VRAM: "
                f"{total_vram_gb:.2f} GB\n"
                "\n"
                "NVIDIA-NemotronLabs-"
                "VoiceChat-11B is too large "
                "for the standard full-precision "
                "loading path on this GPU.\n"
                "\n"
                f"Configured minimum VRAM: "
                f"{MIN_GPU_VRAM_GB:.0f} GB\n"
                "\n"
                "Use a larger-memory GPU or a "
                "separately engineered quantized/"
                "offloaded loading path.\n"
                + "=" * 90
            )

    if not os.path.exists(
        MODEL_PATH
    ):
        raise RuntimeError(
            f"Model directory does not exist: "
            f"{MODEL_PATH}"
        )

    print()
    print(
        "Loading Nemotron VoiceChat model..."
    )
    print(
        "This can take significant time "
        "for the 11B checkpoint."
    )
    print()

    load_start = (
        time.perf_counter()
    )

    try:

        model = await asyncio.to_thread(
            build_model,
            MODEL_PATH,
            DEVICE,
        )

    except torch.cuda.OutOfMemoryError as exc:

        torch.cuda.empty_cache()

        raise RuntimeError(
            "CUDA out of memory while loading "
            "NVIDIA NemotronLabs VoiceChat 11B. "
            "The current GPU does not have "
            "enough VRAM for this loading path."
        ) from exc

    except Exception as exc:

        print()
        print("=" * 90)
        print(
            "MODEL LOADING FAILED"
        )
        print("=" * 90)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 90)

        raise

    model_load_ms = (
        time.perf_counter()
        - load_start
    ) * 1000.0

    print()
    print("=" * 90)
    print(
        "MODEL LOADED SUCCESSFULLY"
    )
    print("=" * 90)

    print(
        f"Model load time : "
        f"{model_load_ms:.2f} ms"
    )

    if torch.cuda.is_available():

        allocated_gb = (
            torch.cuda.memory_allocated(
                0
            )
            / (1024 ** 3)
        )

        reserved_gb = (
            torch.cuda.memory_reserved(
                0
            )
            / (1024 ** 3)
        )

        print(
            f"GPU allocated   : "
            f"{allocated_gb:.2f} GB"
        )

        print(
            f"GPU reserved    : "
            f"{reserved_gb:.2f} GB"
        )

    print(
        "Server ready."
    )

    print("=" * 90)


# =============================================================================
# BASIC ENDPOINTS
# =============================================================================


@app.get("/")
async def root():

    return {
        "service": (
            "nemotron-voicechat-standalone"
        ),
        "model": MODEL_ID,
        "status": (
            "ready"
            if model is not None
            else "loading"
        ),
        "websocket": (
            "/ws/speech_to_speech/"
        ),
        "openai_compatible": (
            "/openai-compatible/v1/"
            "audio/speech-to-speech"
        ),
        "port": PORT,
    }


@app.get("/health")
async def health():

    payload = {
        "status": (
            "ok"
            if model is not None
            else "not_ready"
        ),
        "model": MODEL_ID,
        "device": DEVICE,
        "model_load_ms": model_load_ms,
        "cuda_available": (
            torch.cuda.is_available()
        ),
    }

    if torch.cuda.is_available():

        payload["gpu"] = (
            torch.cuda.get_device_name(
                0
            )
        )

        payload["gpu_total_gb"] = round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3),
            3,
        )

        payload["gpu_allocated_gb"] = round(
            torch.cuda.memory_allocated(
                0
            )
            / (1024 ** 3),
            3,
        )

        payload["gpu_reserved_gb"] = round(
            torch.cuda.memory_reserved(
                0
            )
            / (1024 ** 3),
            3,
        )

    return JSONResponse(
        payload,
        status_code=(
            200
            if model is not None
            else 503
        ),
    )


@app.get(
    "/ws/speech_to_speech/health"
)
async def websocket_health():
    return await health()


# =============================================================================
# OPENAI-COMPATIBLE ENDPOINTS
# =============================================================================


@app.get(
    "/openai-compatible/v1/models"
)
async def list_models():

    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "nvidia",
            }
        ],
    }


@app.post(
    "/openai-compatible/v1/"
    "audio/speech-to-speech"
)
async def speech_to_speech(
    file: UploadFile = File(...),
    instructions: str = Form(
        DEFAULT_SYSTEM_PROMPT
    ),
    response_format: str = Form(
        "wav"
    ),
):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready.",
        )

    request_start = now_ms()

    suffix = (
        Path(
            file.filename
            or "input.wav"
        ).suffix
        or ".wav"
    )

    data = await file.read()

    input_path = (
        upload_to_temp_file(
            data,
            suffix=suffix,
        )
    )

    try:

        result = await infer(
            input_path,
            instructions,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:

        try:
            os.remove(
                input_path
            )
        except OSError:
            pass

    total_ms = (
        now_ms()
        - request_start
    )

    if (
        response_format.lower()
        == "json"
    ):

        wav_bytes = (
            tensor_audio_to_wav_bytes(
                result["audio"],
                result["audio_len"],
            )
        )

        return {
            "model": MODEL_ID,
            "transcript": (
                result["text"]
            ),
            "audio_base64": (
                base64.b64encode(
                    wav_bytes
                ).decode("ascii")
            ),
            "sample_rate": (
                MODEL_OUTPUT_SR
            ),
            "metrics": {
                "model_load_ms": (
                    model_load_ms
                ),
                "prep_ms": (
                    result["prep_ms"]
                ),
                "inference_ms": (
                    result[
                        "inference_ms"
                    ]
                ),
                "total_ms": total_ms,
                "input_duration_s": (
                    result[
                        "input_duration_s"
                    ]
                ),
                "output_duration_s": (
                    result[
                        "output_duration_s"
                    ]
                ),
            },
        }

    wav_bytes = (
        tensor_audio_to_wav_bytes(
            result["audio"],
            result["audio_len"],
        )
    )

    headers = {
        "X-Model-Load-Ms": (
            f"{model_load_ms:.2f}"
            if model_load_ms
            else "0"
        ),
        "X-Prep-Ms": (
            f"{result['prep_ms']:.2f}"
        ),
        "X-Inference-Ms": (
            f"{result['inference_ms']:.2f}"
        ),
        "X-Total-Ms": (
            f"{total_ms:.2f}"
        ),
        "X-Input-Duration-S": (
            f"{result['input_duration_s']:.3f}"
        ),
        "X-Output-Duration-S": (
            f"{result['output_duration_s']:.3f}"
        ),
        "X-Agent-Transcript": (
            result["text"][:1000]
        ),
    }

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers=headers,
    )


# =============================================================================
# WEBSOCKET SPEECH-TO-SPEECH
# =============================================================================


async def websocket_handler(
    websocket: WebSocket,
):

    await websocket.accept()

    if model is None:

        await websocket.send_json(
            make_event(
                "error",
                error={
                    "message": (
                        "Model is not ready."
                    )
                },
            )
        )

        await websocket.close(
            code=1013
        )

        return

    session_id = str(
        uuid.uuid4()
    )

    instructions = (
        DEFAULT_SYSTEM_PROMPT
    )

    audio_buffer = bytearray()

    connection_start = now_ms()

    await websocket.send_json(
        make_event(
            "session.created",
            session={
                "id": session_id,
                "model": MODEL_ID,
                "input_audio_format": (
                    "pcm16"
                ),
                "input_sample_rate": (
                    CLIENT_SAMPLE_RATE
                ),
                "output_audio_format": (
                    "pcm16"
                ),
                "output_sample_rate": (
                    CLIENT_SAMPLE_RATE
                ),
            },
        )
    )

    try:

        while True:

            message = (
                await websocket.receive()
            )

            if (
                message.get("type")
                == "websocket.disconnect"
            ):
                break

            # Optional binary PCM mode
            if (
                message.get("bytes")
                is not None
            ):

                audio_buffer.extend(
                    message["bytes"]
                )

                continue

            raw = message.get(
                "text"
            )

            if raw is None:
                continue

            try:

                payload = json.loads(
                    raw
                )

            except json.JSONDecodeError:

                await websocket.send_json(
                    make_event(
                        "error",
                        error={
                            "message": (
                                "Invalid JSON."
                            )
                        },
                    )
                )

                continue

            message_type = (
                payload.get("type")
            )

            # -------------------------------------------------------------
            # SESSION UPDATE
            # -------------------------------------------------------------

            if (
                message_type
                == "session.update"
            ):

                session = payload.get(
                    "session",
                    {},
                )

                if (
                    session.get(
                        "instructions"
                    )
                    is not None
                ):

                    instructions = (
                        session[
                            "instructions"
                        ]
                    )

                await websocket.send_json(
                    make_event(
                        "session.updated",
                        session={
                            "id": (
                                session_id
                            ),
                            "instructions": (
                                instructions
                            ),
                        },
                    )
                )

            # -------------------------------------------------------------
            # APPEND AUDIO
            # -------------------------------------------------------------

            elif (
                message_type
                == "input_audio_buffer.append"
            ):

                try:

                    encoded = (
                        payload["audio"]
                    )

                    chunk = (
                        base64.b64decode(
                            encoded
                        )
                    )

                    audio_buffer.extend(
                        chunk
                    )

                except Exception as exc:

                    await websocket.send_json(
                        make_event(
                            "error",
                            error={
                                "message": (
                                    "Invalid audio "
                                    f"payload: {exc}"
                                )
                            },
                        )
                    )

            # -------------------------------------------------------------
            # CLEAR
            # -------------------------------------------------------------

            elif (
                message_type
                == "input_audio_buffer.clear"
            ):

                audio_buffer.clear()

                await websocket.send_json(
                    make_event(
                        "input_audio_buffer."
                        "cleared"
                    )
                )

            # -------------------------------------------------------------
            # COMMIT / GENERATE
            # -------------------------------------------------------------

            elif message_type in (
                "input_audio_buffer.commit",
                "response.create",
            ):

                if not audio_buffer:

                    await websocket.send_json(
                        make_event(
                            "error",
                            error={
                                "message": (
                                    "Audio buffer "
                                    "is empty."
                                )
                            },
                        )
                    )

                    continue

                request_start = (
                    now_ms()
                )

                item_id = str(
                    uuid.uuid4()
                )

                response_id = str(
                    uuid.uuid4()
                )

                input_path = (
                    pcm16_bytes_to_wav_file(
                        bytes(
                            audio_buffer
                        ),
                        CLIENT_SAMPLE_RATE,
                    )
                )

                audio_buffer.clear()

                try:

                    result = await infer(
                        input_path,
                        instructions,
                    )

                except Exception as exc:

                    await websocket.send_json(
                        make_event(
                            "error",
                            error={
                                "message": (
                                    str(exc)
                                )
                            },
                        )
                    )

                    continue

                finally:

                    try:
                        os.remove(
                            input_path
                        )
                    except OSError:
                        pass

                # Transcript
                await websocket.send_json(
                    make_event(
                        "response."
                        "output_audio_"
                        "transcript.done",
                        response_id=(
                            response_id
                        ),
                        item_id=item_id,
                        transcript=(
                            result["text"]
                        ),
                    )
                )

                # Convert model output
                pcm = (
                    tensor_audio_to_pcm16_24k(
                        result["audio"],
                        result["audio_len"],
                    )
                )

                bytes_per_chunk = int(
                    CLIENT_SAMPLE_RATE
                    * 2
                    * OUTPUT_CHUNK_MS
                    / 1000
                )

                first_audio_ms = None

                for offset in range(
                    0,
                    len(pcm),
                    bytes_per_chunk,
                ):

                    chunk = pcm[
                        offset:
                        offset
                        + bytes_per_chunk
                    ]

                    if (
                        first_audio_ms
                        is None
                    ):

                        first_audio_ms = (
                            now_ms()
                            - request_start
                        )

                    await websocket.send_json(
                        make_event(
                            "response."
                            "output_audio.delta",
                            response_id=(
                                response_id
                            ),
                            item_id=item_id,
                            delta=(
                                base64.b64encode(
                                    chunk
                                ).decode(
                                    "ascii"
                                )
                            ),
                        )
                    )

                    await asyncio.sleep(0)

                total_ms = (
                    now_ms()
                    - request_start
                )

                await websocket.send_json(
                    make_event(
                        "response."
                        "output_audio.done",
                        response_id=(
                            response_id
                        ),
                        item_id=item_id,
                    )
                )

                await websocket.send_json(
                    make_event(
                        "response.done",
                        response={
                            "id": (
                                response_id
                            ),
                            "status": (
                                "completed"
                            ),
                            "transcript": (
                                result["text"]
                            ),
                        },
                        metrics={
                            "model_load_ms": (
                                model_load_ms
                            ),
                            "prep_ms": (
                                result[
                                    "prep_ms"
                                ]
                            ),
                            "inference_ms": (
                                result[
                                    "inference_ms"
                                ]
                            ),
                            "server_ttfa_ms": (
                                first_audio_ms
                            ),
                            "total_ms": (
                                total_ms
                            ),
                            "input_duration_s": (
                                result[
                                    "input_duration_s"
                                ]
                            ),
                            "output_duration_s": (
                                result[
                                    "output_duration_s"
                                ]
                            ),
                        },
                    )
                )

            # -------------------------------------------------------------
            # CLOSE
            # -------------------------------------------------------------

            elif (
                message_type
                == "session.close"
            ):

                await websocket.send_json(
                    make_event(
                        "session.end",
                        session_id=(
                            session_id
                        ),
                        duration_ms=(
                            now_ms()
                            - connection_start
                        ),
                    )
                )

                await websocket.close()

                break

            else:

                await websocket.send_json(
                    make_event(
                        "error",
                        error={
                            "message": (
                                "Unsupported "
                                "event type: "
                                f"{message_type}"
                            )
                        },
                    )
                )

    except WebSocketDisconnect:
        pass

    except Exception as exc:

        print(
            f"WebSocket error: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        try:

            await websocket.send_json(
                make_event(
                    "error",
                    error={
                        "message": str(
                            exc
                        )
                    },
                )
            )

        except Exception:
            pass


@app.websocket(
    "/ws/speech_to_speech/"
)
async def speech_to_speech_websocket(
    websocket: WebSocket,
):

    await websocket_handler(
        websocket
    )


# =============================================================================
# RUN
# =============================================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host=0.0.0.0,
        port=8001,
        log_level="info",
    )
