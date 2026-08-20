#!/usr/bin/env python3
import argparse
import asyncio
import base64
import io
import json
import os
import time
import uuid
import wave
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import websockets

CLIENT_SR = 24000
CHANNELS = 1
DTYPE = "int16"


def ms_since(t):
    return (time.perf_counter() - t) * 1000.0


def event(event_type, **kwargs):
    return {"type": event_type, "event_id": str(uuid.uuid4()), **kwargs}


def load_audio_pcm16_24k(path: str) -> bytes:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != CLIENT_SR:
        # dependency-light linear interpolation for client-side conversion
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_len = int(round(len(audio) * CLIENT_SR / sr))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        audio = np.interp(new_x, old_x, audio).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def record_mic(seconds: float) -> bytes:
    print(f"Recording microphone for {seconds:.1f} seconds...")
    samples = sd.rec(
        int(seconds * CLIENT_SR),
        samplerate=CLIENT_SR,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()
    print("Recording complete.")
    return samples.reshape(-1).astype("<i2").tobytes()


def save_pcm16_wav(path: str, pcm_bytes: bytes, sample_rate: int = CLIENT_SR):
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def play_pcm16(pcm_bytes: bytes, sample_rate: int = CLIENT_SR):
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    sd.play(samples, samplerate=sample_rate)
    sd.wait()


async def run_ws(args, pcm_bytes: bytes):
    overall_start = time.perf_counter()
    connect_start = time.perf_counter()

    output = bytearray()
    transcript = ""
    first_server_event = None
    first_audio = None
    response_done = None
    server_metrics = {}

    async with websockets.connect(
        args.server,
        max_size=None,
        ping_interval=20,
        ping_timeout=60,
    ) as ws:
        connection_ms = ms_since(connect_start)

        raw = await ws.recv()
        first_server_event = ms_since(overall_start)
        created = json.loads(raw)
        print(f"Connected. Session: {created.get('session', {}).get('id')}")

        await ws.send(
            json.dumps(
                event(
                    "session.update",
                    session={
                        "audio": {
                            "input": {"format": {"type": "audio/pcm", "rate": CLIENT_SR}},
                            "output": {"format": {"type": "audio/pcm", "rate": CLIENT_SR}},
                        },
                        "instructions": args.instructions,
                        "tools": [],
                    },
                )
            )
        )

        chunk_bytes = int(CLIENT_SR * 2 * args.chunk_ms / 1000)
        send_start = time.perf_counter()

        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset : offset + chunk_bytes]
            await ws.send(
                json.dumps(
                    event(
                        "input_audio_buffer.append",
                        audio=base64.b64encode(chunk).decode("ascii"),
                    )
                )
            )
            if args.realtime_send:
                await asyncio.sleep(args.chunk_ms / 1000.0)

        send_ms = ms_since(send_start)
        commit_start = time.perf_counter()
        await ws.send(json.dumps(event("input_audio_buffer.commit")))

        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            typ = msg.get("type")

            if typ == "response.output_audio.delta":
                if first_audio is None:
                    first_audio = ms_since(commit_start)
                output.extend(base64.b64decode(msg["delta"]))

            elif typ == "response.output_audio_transcript.done":
                transcript = msg.get("transcript", "")

            elif typ == "response.done":
                response_done = ms_since(commit_start)
                server_metrics = msg.get("metrics", {})
                break

            elif typ == "error":
                raise RuntimeError(msg.get("error", {}).get("message", str(msg)))

        await ws.send(json.dumps(event("session.close")))

    total_ms = ms_since(overall_start)
    save_pcm16_wav(args.output, bytes(output), CLIENT_SR)

    print("\n" + "=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)
    print(f"Connection latency : {connection_ms:.2f} ms")
    print(f"Send audio         : {send_ms:.2f} ms")
    print(f"Connection -> event: {first_server_event:.2f} ms")
    print(f"COMMIT -> TTFA      : {first_audio:.2f} ms" if first_audio is not None else "COMMIT -> TTFA      : n/a")
    print(f"COMMIT -> response  : {response_done:.2f} ms" if response_done is not None else "COMMIT -> response  : n/a")
    print(f"E2E TOTAL           : {total_ms:.2f} ms")

    print("\n" + "=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)
    for key, value in server_metrics.items():
        if isinstance(value, float):
            print(f"{key:22}: {value:.2f}")
        else:
            print(f"{key:22}: {value}")

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Transcript : {transcript}")
    print(f"Audio      : {args.output}")

    if args.play:
        play_pcm16(bytes(output), CLIENT_SR)


def ws_to_http(ws_url: str) -> str:
    p = urlparse(ws_url)
    scheme = "https" if p.scheme == "wss" else "http"
    host = p.netloc
    return f"{scheme}://{host}"


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = CLIENT_SR):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def run_rest(args, pcm_bytes: bytes):
    base = args.http_server or ws_to_http(args.server)
    url = base.rstrip("/") + "/openai-compatible/v1/audio/speech-to-speech"

    wav_bytes = pcm_to_wav_bytes(pcm_bytes)
    start = time.perf_counter()
    first_byte_start = time.perf_counter()

    resp = requests.post(
        url,
        files={"file": ("input.wav", wav_bytes, "audio/wav")},
        data={
            "instructions": args.instructions,
            "response_format": "wav",
        },
        timeout=args.timeout,
    )
    first_byte_ms = ms_since(first_byte_start)
    resp.raise_for_status()
    total_ms = ms_since(start)

    Path(args.output).write_bytes(resp.content)

    print("\n" + "=" * 80)
    print("OPENAI-STYLE REST LATENCY")
    print("=" * 80)
    print(f"Request -> response : {first_byte_ms:.2f} ms")
    print(f"E2E TOTAL           : {total_ms:.2f} ms")
    for h in [
        "X-Model-Load-Ms",
        "X-Prep-Ms",
        "X-Inference-Ms",
        "X-Total-Ms",
        "X-Input-Duration-S",
        "X-Output-Duration-S",
    ]:
        if h in resp.headers:
            print(f"{h:22}: {resp.headers[h]}")
    print(f"Transcript           : {resp.headers.get('X-Agent-Transcript', '')}")
    print(f"Audio                : {args.output}")

    if args.play:
        audio, sr = sf.read(args.output, dtype="float32")
        sd.play(audio, sr)
        sd.wait()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["ws", "rest"],
        default="ws",
        help="ws = native WebSocket S2S, rest = OpenAI-compatible HTTP S2S",
    )
    p.add_argument(
        "--server",
        default="ws://localhost:8000/ws/speech_to_speech/",
        help="WebSocket URL",
    )
    p.add_argument(
        "--http-server",
        default=None,
        help="Optional HTTP base URL, e.g. http://localhost:8000",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Input audio file")
    src.add_argument("--mic", action="store_true", help="Record from microphone")
    p.add_argument("--seconds", type=float, default=5.0, help="Mic recording duration")
    p.add_argument("--output", default="nemotron_response.wav")
    p.add_argument(
        "--instructions",
        default=(
            "You are a helpful AI voice assistant. "
            "Answer naturally and concisely in spoken English."
        ),
    )
    p.add_argument("--chunk-ms", type=int, default=80)
    p.add_argument(
        "--realtime-send",
        action="store_true",
        help="Pace websocket input chunks in real time",
    )
    p.add_argument("--play", action="store_true")
    p.add_argument("--timeout", type=float, default=600.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.file:
        pcm_bytes = load_audio_pcm16_24k(args.file)
    else:
        pcm_bytes = record_mic(args.seconds)

    if args.mode == "ws":
        asyncio.run(run_ws(args, pcm_bytes))
    else:
        run_rest(args, pcm_bytes)


if __name__ == "__main__":
    main()
