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

Major Optimization Opportunities
1. Early Response Streaming (TTFB Improvement)
Currently, the entire audio is generated and encoded before streaming starts. You can improve TTFB by starting the HTTP response earlier:


2. Remove Unnecessary asyncio.to_thread for Fast Operations
float_to_pcm16() is a quick NumPy operation that doesn't need thread offloading:


3. Conditional Encoding to Skip FFmpeg
When response_format="pcm" and speed=1.0, you're already returning raw PCM but still converting through WAV. Optimize this path:


4. Pre-compute Speaker Resolution
Move speaker validation before GPU lock acquisition:


5. Parallel Logging
Move logging to a background task so it doesn't block response:


6. Use Larger Audio Chunks for Network Efficiency

7. Streaming Inference (if model supports it)
Check if Qwen3TTSModel supports streaming generation:


Estimated Latency Impact
Optimization  TTFB Reduction  Total Latency Reduction
Early streaming  50-200ms  Minimal
Remove unnecessary threading  1-5ms  1-5ms
Skip FFmpeg for PCM  -  10-50ms
Parallel logging  -  2-10ms
Larger chunks  -  5-15ms
Streaming inference  500ms-2s  Variable
The biggest win would be implementing streaming inference if the model supports it, as it would drastically reduce TTFB for long text inputs.




Dia does not define [S1] = male and [S2] = female. They are speaker identities/turn markers, not gender selectors. The model card explicitly says Dia was not fine-tuned on specific voices, so voices can vary between generations.

Transcript stops early because max_new_tokens=1024 is much too small for the script we are giving , and Dia recommends avoiding very long single generations. Nari Labs says 256 tokens is only about ~2 seconds and recommends keeping generations roughly below 20 seconds for quality.
[S1] and [S2] do not mean female/male. Dia explicitly says voices can change between runs unless you use audio conditioning or, less reliably, a fixed seed.

Dia’s generation architecture is optimized for relatively short dialogue segments, not arbitrarily long scripts in one decode pass.

There are a few reasons:

Audio tokens grow very quickly. Dia generates audio-token sequences autoregressively. The longer the requested speech, the more tokens it must produce, and generation time grows accordingly.
max_new_tokens is a hard ceiling. If the model needs more tokens than the limit you set, generation stops even if the text is unfinished.
Quality degrades on very long contexts. The Dia authors note that very long generations can become faster, less stable, or otherwise degrade in quality. (github.com)
The model is dialogue-oriented. It was designed around [S1] / [S2] conversational turns. That works best when the model handles manageable spans of dialogue rather than several minutes of speech in one shot.
Memory and latency scale up. A long single generation keeps more autoregressive state around and delays the first audio until the whole generation completes.
reference-https://github.com/nari-labs/dia/issues/109
https://huggingface.co/nari-labs/Dia-1.6B-0626/blob/main/README.md
https://github.com/nari-labs/dia/blob/main/README.md

                                                                                                                                                                                                                                                                            
The problem is that long single-pass audio generation becomes token-limited, slow, and less stable, which is why chunking is the safer design which i am tryng now .                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                            

