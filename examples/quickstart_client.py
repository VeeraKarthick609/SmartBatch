"""
Quickstart client — sends concurrent requests to the quickstart server.

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
        print(f"  square({x}) = {results[x]}")
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
        futures = {ex.submit(post, "/models/word-count/predict", {"data": {"text": s}}): s for s in sentences}
        for f in as_completed(futures):
            s = futures[f]
            print(f"  {f.result()['result']} words: \"{s}\"")
    elapsed = time.perf_counter() - start
    print(f"  Wall time: {elapsed:.3f}s")


def demo_metrics():
    print("\n--- /metrics ---")
    r = requests.get(f"{BASE_URL}/metrics", timeout=5)
    print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    demo_square()
    demo_word_count()
    demo_metrics()
