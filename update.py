import time
import requests

url = "https://qwen3-tts-150916788856.us-central1.run.app/v1/audio/speech"

payload = {
    "model": "qwen3-tts-0.6b",
    "input": "I cannot believe we finally made it!",
    "voice": "Aiden",
    "instructions": "Speak happily and with excitement.",
    "response_format": "wav",
    "speed": 1.0,
    "language": "English",
}

start = time.perf_counter()

response = requests.post(
    url,
    json=payload,
    stream=True,
    timeout=300,
)

headers_time = time.perf_counter()

response.raise_for_status()

first_audio_time = None

with open("openai_test.wav", "wb") as f:

    for chunk in response.iter_content(chunk_size=65536):

        if not chunk:
            continue

        if first_audio_time is None:
            first_audio_time = time.perf_counter()

        f.write(chunk)

end = time.perf_counter()


ttfb_ms = (headers_time - start) * 1000

ttfa_ms = (
    (first_audio_time - start) * 1000
    if first_audio_time
    else None
)

total_ms = (end - start) * 1000


print()
print("=" * 70)
print("CLIENT LATENCY")
print("=" * 70)

print(f"CLIENT TTFB       : {ttfb_ms:.2f} ms")

if ttfa_ms is not None:
    print(f"CLIENT TTFT/TTFA  : {ttfa_ms:.2f} ms")

print(f"CLIENT TOTAL      : {total_ms:.2f} ms")


print()
print("=" * 70)
print("SERVER LATENCY")
print("=" * 70)

print(
    f"SERVER INFERENCE  : "
    f"{response.headers.get('X-Server-Inference-MS')} ms"
)

print(
    f"SERVER ENCODING   : "
    f"{response.headers.get('X-Server-Encoding-MS')} ms"
)

print(
    f"SERVER TOTAL      : "
    f"{response.headers.get('X-Server-Total-MS')} ms"
)

print(
    f"AUDIO DURATION    : "
    f"{response.headers.get('X-Audio-Duration-S')} s"
)

print(
    f"RTF               : "
    f"{response.headers.get('X-RTF')}"
)

print()
print("Saved             : openai_test.wav")


curl -X POST "https://qwen3-tts-150916788856.us-central1.run.app/v1/audio/speech" -H "Content-Type: application/json" -d "{\"model\":\"qwen3-tts-0.6b\",\"input\":\"I cannot believe we finally made it!\",\"voice\":\"Aiden\",\"instructions\":\"Speak happily and with excitement.\",\"response_format\":\"wav\",\"speed\":1.0,\"language\":\"English\"}" --output openai_test.wav

{"message": "failed to synthesize speech: no audio frames were pushed for text: Hello, retrying in 2.0s", "level": "WARNING", "name": "livekit.agents", "tts": "livekit.plugins.openai.tts.TTS", "attempt": 3, "streamed": false, "eval_job_id": "run_af7ffa98", "scenario_id": "f29b843e-97c4-11f1-8d8d-42004e494300", "tenant_id": "default", "pid": 35, "job_id": "AJ_RYjUkiQ7ZSuH", "room": "aqa-f29b843e-97c4-11f1-8d8d-42004e494300-2b33d253", "timestamp": "2026-08-20T09:11:05.655372+00:00"}
 
