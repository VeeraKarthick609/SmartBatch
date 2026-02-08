import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import msgpack
import numpy as np
import requests


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


def generate_payload(seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    image = rng.random((3, 224, 224), dtype=np.float32).tolist()
    return {"data": {"image": image}}


def send_json(url: str, payload: Dict[str, object], timeout: float) -> Tuple[int, float, str]:
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    latency = time.perf_counter() - start
    return response.status_code, latency, response.text[:200]


def send_msgpack(url: str, payload: Dict[str, object], timeout: float) -> Tuple[int, float, str]:
    start = time.perf_counter()
    packed = msgpack.packb(payload, use_bin_type=True)
    response = requests.post(
        url,
        data=packed,
        headers={"Content-Type": "application/msgpack"},
        timeout=timeout,
    )
    latency = time.perf_counter() - start
    return response.status_code, latency, response.text[:200]


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartBatch ResNet example client/load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for SmartBatch API")
    parser.add_argument("--model", default="resnet18", help="Registered model name")
    parser.add_argument("--version", default="v2", help="Model version query parameter")
    parser.add_argument("--requests", type=int, default=32, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent request workers")
    parser.add_argument("--format", choices=["json", "msgpack"], default="json", help="Request payload format")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    args = parser.parse_args()

    endpoint = f"{args.base_url.rstrip('/')}/models/{args.model}/predict?version={args.version}"
    payloads = [generate_payload(seed=i) for i in range(args.requests)]

    request_fn = send_json if args.format == "json" else send_msgpack

    print(f"Target endpoint: {endpoint}")
    print(f"Requests: {args.requests}, Concurrency: {args.concurrency}, Format: {args.format}")

    latencies: List[float] = []
    status_counts: Dict[int, int] = {}
    failures: List[str] = []
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(request_fn, endpoint, payload, args.timeout) for payload in payloads
        ]
        for future in as_completed(futures):
            try:
                status, latency, body_preview = future.result()
                latencies.append(latency)
                status_counts[status] = status_counts.get(status, 0) + 1
                if status >= 400:
                    failures.append(f"status={status}, body={body_preview}")
            except Exception as exc:
                failures.append(str(exc))

    total_time = time.perf_counter() - started
    success = sum(count for status, count in status_counts.items() if 200 <= status < 300)
    rps = (len(latencies) / total_time) if total_time > 0 else 0.0

    print("\nSummary")
    print(f"- Successful responses: {success}/{args.requests}")
    print(f"- Status counts: {status_counts}")
    print(f"- Total wall time: {total_time:.3f}s")
    print(f"- Throughput: {rps:.2f} req/s")
    if latencies:
        print(f"- Latency mean: {statistics.mean(latencies):.4f}s")
        print(f"- Latency p50: {percentile(latencies, 50):.4f}s")
        print(f"- Latency p95: {percentile(latencies, 95):.4f}s")
        print(f"- Latency p99: {percentile(latencies, 99):.4f}s")

    if failures:
        print("\nFailures (first 5)")
        for item in failures[:5]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
