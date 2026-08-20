import asyncio
import base64
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from nemo.collections.speechlm2.inference.utils.offline_voicechat import (
    TARGET_SR,
    SOURCE_SR,
    build_model,
    encode_system_prompt,
    load_wav_16k_mono,
    run_offline_inference,
)

MODEL_ID = os.getenv("MODEL_ID", "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/NVIDIA-NemotronLabs-VoiceChat-11B")
DEVICE = os.getenv("DEVICE", "cuda")
CLIENT_SAMPLE_RATE = 24000
MODEL_INPUT_SR = SOURCE_SR       # 16000
MODEL_OUTPUT_SR = TARGET_SR      # 22050
OUTPUT_CHUNK_MS = int(os.getenv("OUTPUT_CHUNK_MS", "80"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again."
)

app = FastAPI(
    title="NemotronLabs VoiceChat 11B Speech-to-Speech API",
    version="1.0.0",
)

model = None
model_load_ms = None
inference_lock = asyncio.Semaphore(MAX_CONCURRENT)


def now_ms():
    return time.perf_counter() * 1000.0


def event(event_type: str, **kwargs):
    return {
        "type": event_type,
        "event_id": str(uuid.uuid4()),
        **kwargs,
    }


def pcm16_bytes_to_wav_file(pcm_bytes: bytes, sample_rate: int = CLIENT_SAMPLE_RATE) -> str:
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return path


def upload_to_wav_file(data: bytes, suffix: str = ".wav") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(path).write_bytes(data)
    return path


def tensor_audio_to_pcm16_24k(audio_tensor: torch.Tensor, audio_len: int) -> bytes:
    audio = audio_tensor[:audio_len].detach().float().cpu().unsqueeze(0)
    if MODEL_OUTPUT_SR != CLIENT_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, MODEL_OUTPUT_SR, CLIENT_SAMPLE_RATE)
    audio = audio.squeeze(0).clamp(-1.0, 1.0)
    pcm = (audio.numpy() * 32767.0).astype("<i2")
    return pcm.tobytes()


def tensor_audio_to_wav_bytes(audio_tensor: torch.Tensor, audio_len: int) -> bytes:
    audio = audio_tensor[:audio_len].detach().float().cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio, MODEL_OUTPUT_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _infer_sync(wav_path: str, system_prompt: str):
    global model
    prep_start = now_ms()

    wav_1d, input_signal, input_signal_lens = load_wav_16k_mono(
        wav_path, device=DEVICE
    )
    prompt_tokens, prompt_token_lens = encode_system_prompt(
        model, system_prompt, device=DEVICE
    )
    prep_ms = now_ms() - prep_start

    infer_start = now_ms()
    result = run_offline_inference(
        model,
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
    )
    inference_ms = now_ms() - infer_start

    text = result.get("text", [""])[0]
    if result.get("audio") is None:
        raise RuntimeError("Model returned no audio.")

    audio_len = int(result["audio_len"][0].item())
    audio = result["audio"][0]
    return {
        "text": text,
        "audio": audio,
        "audio_len": audio_len,
        "prep_ms": prep_ms,
        "inference_ms": inference_ms,
        "input_duration_s": float(wav_1d.shape[0]) / MODEL_INPUT_SR,
        "output_duration_s": float(audio_len) / MODEL_OUTPUT_SR,
    }


async def infer(wav_path: str, system_prompt: str):
    async with inference_lock:
        return await asyncio.to_thread(_infer_sync, wav_path, system_prompt)


@app.on_event("startup")
async def startup_event():
    global model, model_load_ms

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but no CUDA GPU is visible.")

    print("=" * 90)
    print("NVIDIA NemotronLabs VoiceChat 11B")
    print("=" * 90)
    print(f"MODEL_ID        : {MODEL_ID}")
    print(f"MODEL_PATH      : {MODEL_PATH}")
    print(f"DEVICE          : {DEVICE}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    start = now_ms()
    model = await asyncio.to_thread(build_model, MODEL_PATH, DEVICE)
    model_load_ms = now_ms() - start

    print(f"Model load time : {model_load_ms:.2f} ms")
    print("Server ready.")
    print("=" * 90)


@app.get("/")
async def root():
    return {
        "service": "nemotron-voicechat-standalone",
        "version": "1.0.0",
        "model": MODEL_ID,
        "websocket": "/ws/speech_to_speech/",
        "openai_compatible_s2s": "/openai-compatible/v1/audio/speech-to-speech",
        "openai_compatible_models": "/openai-compatible/v1/models",
        "health": "/health",
        "client_input": "PCM16 mono 24kHz for realtime websocket",
        "model_input_rate": MODEL_INPUT_SR,
        "model_output_rate": MODEL_OUTPUT_SR,
        "client_output_rate": CLIENT_SAMPLE_RATE,
    }


@app.get("/health")
@app.get("/ws/speech_to_speech/health")
async def health():
    ok = model is not None
    payload = {
        "status": "ok" if ok else "loading",
        "model": MODEL_ID,
        "device": DEVICE,
        "model_load_ms": model_load_ms,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        payload["gpu"] = torch.cuda.get_device_name(0)
        payload["gpu_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 2**30, 3)
        payload["gpu_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 2**30, 3)
    return JSONResponse(payload, status_code=200 if ok else 503)


@app.get("/openai-compatible/v1/models")
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


@app.post("/openai-compatible/v1/audio/speech-to-speech")
async def speech_to_speech(
    file: UploadFile = File(...),
    instructions: str = Form(DEFAULT_SYSTEM_PROMPT),
    response_format: str = Form("wav"),
):
    request_start = now_ms()
    suffix = Path(file.filename or "input.wav").suffix or ".wav"
    data = await file.read()
    wav_path = upload_to_wav_file(data, suffix=suffix)

    try:
        result = await infer(wav_path, instructions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    total_ms = now_ms() - request_start
    headers = {
        "X-Model-Load-Ms": f"{model_load_ms:.2f}" if model_load_ms else "0",
        "X-Prep-Ms": f"{result['prep_ms']:.2f}",
        "X-Inference-Ms": f"{result['inference_ms']:.2f}",
        "X-Total-Ms": f"{total_ms:.2f}",
        "X-Input-Duration-S": f"{result['input_duration_s']:.3f}",
        "X-Output-Duration-S": f"{result['output_duration_s']:.3f}",
        "X-Agent-Transcript": result["text"][:1000],
    }

    if response_format.lower() == "json":
        wav_bytes = tensor_audio_to_wav_bytes(result["audio"], result["audio_len"])
        return {
            "model": MODEL_ID,
            "transcript": result["text"],
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": MODEL_OUTPUT_SR,
            "metrics": {
                "model_load_ms": model_load_ms,
                "prep_ms": result["prep_ms"],
                "inference_ms": result["inference_ms"],
                "total_ms": total_ms,
                "input_duration_s": result["input_duration_s"],
                "output_duration_s": result["output_duration_s"],
            },
        }

    wav_bytes = tensor_audio_to_wav_bytes(result["audio"], result["audio_len"])
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


async def realtime_socket(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    instructions = DEFAULT_SYSTEM_PROMPT
    audio_buffer = bytearray()
    connection_start = now_ms()

    await ws.send_json(
        event(
            "session.created",
            session={
                "type": "realtime",
                "id": session_id,
                "model": MODEL_ID,
                "modalities": ["audio"],
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": CLIENT_SAMPLE_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": CLIENT_SAMPLE_RATE}},
                },
                "instructions": instructions,
            },
        )
    )

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                # Convenience mode: raw binary PCM16 24kHz.
                audio_buffer.extend(msg["bytes"])
                continue

            raw = msg.get("text")
            if raw is None:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json(event("error", error={"message": "Invalid JSON"}))
                continue

            msg_type = payload.get("type")

            if msg_type == "session.update":
                sess = payload.get("session", {})
                if sess.get("instructions") is not None:
                    instructions = sess["instructions"]
                await ws.send_json(
                    event(
                        "session.updated",
                        session={
                            "audio": {
                                "input": {"format": {"type": "audio/pcm", "rate": CLIENT_SAMPLE_RATE}},
                                "output": {"format": {"type": "audio/pcm", "rate": CLIENT_SAMPLE_RATE}},
                            },
                            "instructions": instructions,
                            "tools": sess.get("tools", []),
                        },
                    )
                )

            elif msg_type == "input_audio_buffer.append":
                try:
                    audio_buffer.extend(base64.b64decode(payload["audio"]))
                except Exception as exc:
                    await ws.send_json(
                        event("error", error={"message": f"Invalid audio payload: {exc}"})
                    )

            elif msg_type in ("input_audio_buffer.commit", "response.create"):
                if not audio_buffer:
                    await ws.send_json(
                        event("error", error={"message": "Audio buffer is empty."})
                    )
                    continue

                request_start = now_ms()
                item_id = str(uuid.uuid4())
                response_id = str(uuid.uuid4())

                await ws.send_json(
                    event(
                        "input_audio_buffer.speech_started",
                        audio_start_ms=0,
                        item_id=item_id,
                    )
                )

                wav_path = pcm16_bytes_to_wav_file(bytes(audio_buffer), CLIENT_SAMPLE_RATE)
                audio_buffer.clear()

                try:
                    result = await infer(wav_path, instructions)
                except Exception as exc:
                    await ws.send_json(event("error", error={"message": str(exc)}))
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                    continue
                finally:
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass

                first_response_ms = now_ms() - request_start

                await ws.send_json(
                    event(
                        "input_audio_buffer.speech_stopped",
                        audio_end_ms=int(result["input_duration_s"] * 1000),
                        item_id=item_id,
                    )
                )
                await ws.send_json(
                    event(
                        "response.created",
                        response={
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "in_progress",
                            "output": [],
                        },
                    )
                )
                await ws.send_json(
                    event(
                        "response.output_audio_transcript.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=result["text"],
                    )
                )

                pcm = tensor_audio_to_pcm16_24k(result["audio"], result["audio_len"])
                bytes_per_chunk = int(CLIENT_SAMPLE_RATE * 2 * OUTPUT_CHUNK_MS / 1000)

                audio_send_start = now_ms()
                first_audio_sent_ms = None
                for offset in range(0, len(pcm), bytes_per_chunk):
                    chunk = pcm[offset : offset + bytes_per_chunk]
                    if first_audio_sent_ms is None:
                        first_audio_sent_ms = now_ms() - request_start
                    await ws.send_json(
                        event(
                            "response.output_audio.delta",
                            response_id=response_id,
                            item_id=item_id,
                            output_index=0,
                            content_index=0,
                            delta=base64.b64encode(chunk).decode("ascii"),
                        )
                    )
                    await asyncio.sleep(0)

                await ws.send_json(
                    event(
                        "response.output_audio_transcript.done",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        transcript=result["text"],
                    )
                )
                await ws.send_json(
                    event(
                        "response.output_audio.done",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                    )
                )

                total_ms = now_ms() - request_start
                await ws.send_json(
                    event(
                        "response.done",
                        response={
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "completed",
                        },
                        metrics={
                            "model_load_ms": model_load_ms,
                            "prep_ms": result["prep_ms"],
                            "inference_ms": result["inference_ms"],
                            "server_ttft_ms": first_response_ms,
                            "server_ttfa_ms": first_audio_sent_ms,
                            "audio_send_ms": now_ms() - audio_send_start,
                            "total_ms": total_ms,
                            "input_duration_s": result["input_duration_s"],
                            "output_duration_s": result["output_duration_s"],
                        },
                    )
                )

            elif msg_type == "input_audio_buffer.clear":
                audio_buffer.clear()
                await ws.send_json(event("input_audio_buffer.cleared"))

            elif msg_type == "session.close":
                await ws.send_json(
                    event(
                        "session.end",
                        session_id=session_id,
                        duration_ms=now_ms() - connection_start,
                    )
                )
                await ws.close()
                break

            else:
                await ws.send_json(
                    event(
                        "error",
                        error={"message": f"Unsupported event type: {msg_type}"},
                    )
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json(event("error", error={"message": str(exc)}))
        except Exception:
            pass


@app.websocket("/ws/speech_to_speech/")
async def websocket_s2s(ws: WebSocket):
    await realtime_socket(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
