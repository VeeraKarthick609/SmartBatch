"""
Dynamic registration example.

Shows how to register, call, and deregister a model at runtime via the
admin API — no server restart needed.

Prerequisites:
    Server must be running:
        python examples/quickstart_server.py   (or any SmartBatch server)

    The server's Python path must include the examples/ folder so it can
    import 'dynamic_model'. Easiest way: start the server from the repo root.

Run:
    python examples/dynamic_registration.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000"
MODEL_NAME = "sentiment"
MODEL_VERSION = "v1"


def step(msg: str):
    print(f"\n{'─' * 50}")
    print(f"  {msg}")
    print('─' * 50)


def register_model():
    step("1. Register 'sentiment' model dynamically")
    payload = {
        "module": "examples.dynamic_model",
        "function": "sentiment_score",
        "version": MODEL_VERSION,
        "max_batch_size": 16,
        "max_wait_time": 0.01,
    }
    r = requests.post(f"{BASE_URL}/admin/models/{MODEL_NAME}", json=payload, timeout=10)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


def list_models():
    step("2. List all registered models")
    r = requests.get(f"{BASE_URL}/admin/models", timeout=5)
    print(json.dumps(r.json(), indent=2))


def run_inference():
    step("3. Run 10 concurrent inference requests")
    texts = [
        "this product is great and awesome",
        "absolutely terrible experience",
        "it was okay nothing special",
        "love it fantastic quality",
        "horrible waste of money",
        "good value for the price",
        "poor customer service awful",
        "excellent highly recommend",
        "not bad not great either",
        "fantastic and excellent work",
    ]

    endpoint = f"{BASE_URL}/models/{MODEL_NAME}/predict?version={MODEL_VERSION}"

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {
            ex.submit(
                requests.post,
                endpoint,
                json={"data": text},
                timeout=10,
            ): text
            for text in texts
        }
        for f in as_completed(futures):
            text = futures[f]
            result = f.result().json()["result"]
            print(f"  [{result['label']:>8} {result['score']:+.3f}]  \"{text}\"")
    elapsed = time.perf_counter() - start
    print(f"\n  Wall time: {elapsed:.3f}s")


def deregister_model():
    step(f"4. Deregister '{MODEL_NAME}' version '{MODEL_VERSION}'")
    r = requests.delete(
        f"{BASE_URL}/admin/models/{MODEL_NAME}/{MODEL_VERSION}", timeout=5
    )
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


def verify_gone():
    step("5. Confirm model is gone (expect 404)")
    r = requests.post(
        f"{BASE_URL}/models/{MODEL_NAME}/predict?version={MODEL_VERSION}",
        json={"data": "test"},
        timeout=5,
    )
    print(f"  Status: {r.status_code}  (expected 404)")
    print(f"  Body:   {r.json()['detail']}")


if __name__ == "__main__":
    register_model()
    list_models()
    run_inference()
    deregister_model()
    verify_gone()
