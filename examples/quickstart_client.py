"""
Quickstart client — exercises all three models on the quickstart server.

Run the server first:
    python examples/quickstart_server.py

Then run this:
    python examples/quickstart_client.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "http://127.0.0.1:8000"


def post(path: str, payload: dict) -> dict:
    r = requests.post(f"{BASE_URL}{path}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Standard batch demos
# ---------------------------------------------------------------------------

def demo_square():
    print("\n--- square: 16 concurrent requests ---")
    inputs = list(range(16))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(post, "/models/square/predict", {"data": x}): x for x in inputs}
        results = {}
        for f in as_completed(futures):
            x = futures[f]
            results[x] = f.result()["result"]
    elapsed = time.perf_counter() - start

    for x in inputs:
        print(f"  square({x:2d}) = {results[x]}")
    print(f"  Wall time: {elapsed:.3f}s")


def demo_word_count():
    print("\n--- word-count: 8 concurrent requests ---")
    sentences = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "smartbatch groups these into one batch",
        "reducing inference overhead significantly",
        "hello world",
        "one",
        "two words",
        "three little words",
    ]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(post, "/models/word-count/predict", {"data": {"text": s}}): s
            for s in sentences
        }
        for f in as_completed(futures):
            s = futures[f]
            print(f"  {f.result()['result']} words: \"{s}\"")
    elapsed = time.perf_counter() - start
    print(f"  Wall time: {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# Streaming demo
# ---------------------------------------------------------------------------

def stream_tokens(topic: str) -> list[str]:
    """
    Call POST /models/storyteller/stream and collect tokens via SSE.
    Returns the full list of tokens received.
    """
    tokens = []
    with requests.post(
        f"{BASE_URL}/models/storyteller/stream",
        json={"data": topic},
        stream=True,
        timeout=30,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode() if isinstance(line, bytes) else line
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(f"Stream error: {event['error']}")
            tokens.append(event["token"])
    return tokens


def demo_streaming():
    print("\n--- storyteller: 4 concurrent streaming requests ---")
    topics = ["dragons", "quantum computers", "the ocean", "ancient Rome"]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(stream_tokens, topic): topic for topic in topics}
        for f in as_completed(futures):
            topic = futures[f]
            tokens = f.result()
            story = "".join(tokens)
            print(f"\n  [{topic}]\n  {story}")
    elapsed = time.perf_counter() - start
    print(f"\n  Wall time: {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def demo_metrics():
    print("\n--- /metrics ---")
    r = requests.get(f"{BASE_URL}/metrics", timeout=5)
    print(json.dumps(r.json(), indent=2))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_square()
    demo_word_count()
    demo_streaming()
    demo_metrics()
