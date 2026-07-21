"""
Benchmark Groq (hosted, free tier) vs. local Ollama latency for the same
generation calls. Run this locally — it needs GROQ_API_KEY set for the Groq
half and a running Ollama instance (`ollama serve`, with llama3.1 pulled)
for the Ollama half.

Usage:
    python -m app.benchmark
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from app.generation import generator as gen

TEST_PROMPTS = [
    "What is 2 + 2?",
    "Name one primary color.",
    "What is the capital of France?",
]

SYSTEM = "Answer briefly, in one sentence."


def time_calls(label: str, call_fn) -> None:
    times = []
    for prompt in TEST_PROMPTS:
        start = time.perf_counter()
        try:
            call_fn(SYSTEM, prompt)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  [{label}] {elapsed:.2f}s — {prompt!r}")
        except Exception as e:
            print(f"  [{label}] FAILED — {prompt!r} — {e}")

    if times:
        avg = sum(times) / len(times)
        print(f"  [{label}] average: {avg:.2f}s over {len(times)} calls\n")


if __name__ == "__main__":
    print("Benchmarking generation latency\n")

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        print("Groq (hosted, free tier):")
        time_calls("groq", lambda s, u: gen._call_groq(s, u, groq_key))
    else:
        print("Skipping Groq — GROQ_API_KEY not set in .env\n")

    print("Ollama (local):")
    try:
        time_calls("ollama", gen._call_ollama)
    except Exception as e:
        print(f"  Ollama unavailable — is it running? ('ollama serve') — {e}\n")

    print("Tip: Groq is almost always faster (hosted on dedicated inference")
    print("hardware) but Ollama has zero network dependency and zero rate limits.")
    print("Use Groq for demos/deployment, Ollama as an offline fallback.")