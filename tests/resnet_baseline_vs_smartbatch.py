"""
Benchmark baseline (request-per-inference) vs SmartBatch using ResNet18.

Outputs:
- latency_comparison.png
- throughput_comparison.png
- p50_comparison.png
- p95_comparison.png
- p99_comparison.png
- summary.json
- samples.csv

Usage:
    python tests/resnet_baseline_vs_smartbatch.py --requests 64 --concurrency 16
"""

import argparse
import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from torchvision.models import resnet18
except ImportError as exc:
    raise RuntimeError(
        "torchvision is required. Install with: pip install -r .examples/requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from smartbatch import batch


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def throughput_series(completions: List[float], bin_size: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    if not completions:
        return np.array([]), np.array([])
    max_t = max(completions)
    bins = np.arange(bin_size, max_t + bin_size, bin_size)
    counts = []
    prev = 0.0
    for current in bins:
        count = sum(1 for t in completions if prev < t <= current)
        counts.append(count / bin_size)
        prev = current
    return bins, np.array(counts, dtype=np.float64)


@dataclass
class RunStats:
    name: str
    latencies: List[float]
    completions: List[float]
    total_time: float
    success: int
    failures: int

    @property
    def throughput(self) -> float:
        return (self.success / self.total_time) if self.total_time > 0 else 0.0

    @property
    def p50(self) -> float:
        return percentile(self.latencies, 50)

    @property
    def p95(self) -> float:
        return percentile(self.latencies, 95)

    @property
    def p99(self) -> float:
        return percentile(self.latencies, 99)

    @property
    def mean(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else 0.0


@dataclass
class WorkerBundle:
    model: torch.nn.Module
    device: torch.device
    lock: Lock


def build_worker_bundle(worker_id: int) -> WorkerBundle:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    model = resnet18(weights=None)
    model.eval()
    model.to(device)
    return WorkerBundle(model=model, device=device, lock=Lock())


def generate_inputs(total_requests: int, seed: int, image_size: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((3, image_size, image_size), dtype=np.float32) for _ in range(total_requests)]


def run_model_sync(bundle: WorkerBundle, batch_inputs: List[np.ndarray]) -> List[int]:
    tensor_batch = torch.stack([torch.from_numpy(arr) for arr in batch_inputs]).to(bundle.device)
    with bundle.lock:
        with torch.inference_mode():
            logits = bundle.model(tensor_batch)
    top1 = torch.argmax(logits, dim=1)
    return top1.cpu().tolist()


def build_baseline_handler(bundle: WorkerBundle) -> Callable[[np.ndarray], Any]:
    async def handler(image: np.ndarray) -> int:
        result = run_model_sync(bundle, [image])
        return result[0]

    return handler


def build_smartbatch_handler(bundles: Dict[int, WorkerBundle], max_batch_size: int, max_wait_time: float) -> Any:
    @batch(
        max_batch_size=max_batch_size,
        max_wait_time=max_wait_time,
        workers=len(bundles),
        max_queue_size=2048,
        target_latency=0.2,
    )
    async def infer(batch_inputs: List[np.ndarray], worker_id: int = 0) -> List[int]:
        bundle = bundles[worker_id % len(bundles)]
        return run_model_sync(bundle, batch_inputs)

    return infer


async def run_workload(
    name: str,
    handler: Callable[[np.ndarray], Any],
    inputs: List[np.ndarray],
    concurrency: int,
) -> RunStats:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    completions: List[float] = []
    failures = 0
    started = time.perf_counter()

    async def one_request(item: np.ndarray):
        nonlocal failures
        async with semaphore:
            t0 = time.perf_counter()
            try:
                await handler(item)
            except Exception:
                failures += 1
            finally:
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                completions.append(t1 - started)

    await asyncio.gather(*(one_request(item) for item in inputs))
    total_time = time.perf_counter() - started
    success = len(inputs) - failures
    return RunStats(
        name=name,
        latencies=latencies,
        completions=completions,
        total_time=total_time,
        success=success,
        failures=failures,
    )


def save_summary(summary_path: Path, baseline: RunStats, smartbatch_stats: RunStats) -> None:
    payload = {
        "baseline": {
            "total_time": baseline.total_time,
            "throughput": baseline.throughput,
            "latency_mean": baseline.mean,
            "latency_p50": baseline.p50,
            "latency_p95": baseline.p95,
            "latency_p99": baseline.p99,
            "success": baseline.success,
            "failures": baseline.failures,
        },
        "smartbatch": {
            "total_time": smartbatch_stats.total_time,
            "throughput": smartbatch_stats.throughput,
            "latency_mean": smartbatch_stats.mean,
            "latency_p50": smartbatch_stats.p50,
            "latency_p95": smartbatch_stats.p95,
            "latency_p99": smartbatch_stats.p99,
            "success": smartbatch_stats.success,
            "failures": smartbatch_stats.failures,
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2))


def save_samples_csv(csv_path: Path, baseline: RunStats, smartbatch_stats: RunStats) -> None:
    max_len = max(len(baseline.latencies), len(smartbatch_stats.latencies))
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "request_index",
                "baseline_latency_s",
                "smartbatch_latency_s",
                "baseline_completion_s",
                "smartbatch_completion_s",
            ]
        )
        for idx in range(max_len):
            writer.writerow(
                [
                    idx,
                    baseline.latencies[idx] if idx < len(baseline.latencies) else "",
                    smartbatch_stats.latencies[idx] if idx < len(smartbatch_stats.latencies) else "",
                    baseline.completions[idx] if idx < len(baseline.completions) else "",
                    smartbatch_stats.completions[idx] if idx < len(smartbatch_stats.completions) else "",
                ]
            )


def plot_latency(output_dir: Path, baseline: RunStats, smartbatch_stats: RunStats) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(baseline.latencies)), baseline.latencies, label="Baseline", alpha=0.8)
    ax.plot(range(len(smartbatch_stats.latencies)), smartbatch_stats.latencies, label="SmartBatch", alpha=0.8)
    ax.set_title("Per-request Latency")
    ax.set_xlabel("Request Completion Index")
    ax.set_ylabel("Latency (s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", dpi=160)
    plt.close(fig)


def plot_throughput(output_dir: Path, baseline: RunStats, smartbatch_stats: RunStats) -> None:
    baseline_x, baseline_y = throughput_series(baseline.completions)
    smart_x, smart_y = throughput_series(smartbatch_stats.completions)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(baseline_x, baseline_y, label="Baseline", alpha=0.8)
    ax.plot(smart_x, smart_y, label="SmartBatch", alpha=0.8)
    ax.set_title("Throughput Over Time")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_comparison.png", dpi=160)
    plt.close(fig)


def plot_single_percentile(
    output_dir: Path,
    baseline: RunStats,
    smartbatch_stats: RunStats,
    p_name: str,
    baseline_value: float,
    smartbatch_value: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Baseline", "SmartBatch"], [baseline_value, smartbatch_value], color=["#e76f51", "#2a9d8f"])
    ax.set_title(f"{p_name} Latency Comparison")
    ax.set_ylabel("Latency (s)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{p_name.lower()}_comparison.png", dpi=160)
    plt.close(fig)


def print_summary(baseline: RunStats, smartbatch_stats: RunStats, output_dir: Path) -> None:
    print("\nBenchmark Summary")
    print(f"- Baseline success/fail: {baseline.success}/{baseline.failures}")
    print(f"- SmartBatch success/fail: {smartbatch_stats.success}/{smartbatch_stats.failures}")
    print(f"- Baseline throughput: {baseline.throughput:.2f} req/s")
    print(f"- SmartBatch throughput: {smartbatch_stats.throughput:.2f} req/s")
    print(f"- Baseline p50/p95/p99: {baseline.p50:.4f}s / {baseline.p95:.4f}s / {baseline.p99:.4f}s")
    print(
        "- SmartBatch p50/p95/p99: "
        f"{smartbatch_stats.p50:.4f}s / {smartbatch_stats.p95:.4f}s / {smartbatch_stats.p99:.4f}s"
    )
    print(f"- Graph output dir: {output_dir}")


async def benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = generate_inputs(total_requests=args.requests, seed=args.seed, image_size=args.image_size)

    baseline_bundle = build_worker_bundle(worker_id=0)
    baseline_handler = build_baseline_handler(baseline_bundle)

    smartbatch_bundles = {i: build_worker_bundle(worker_id=i) for i in range(args.workers)}
    smartbatch_handler = build_smartbatch_handler(
        bundles=smartbatch_bundles,
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time,
    )

    baseline_stats = await run_workload(
        name="baseline",
        handler=baseline_handler,
        inputs=inputs,
        concurrency=args.concurrency,
    )

    smartbatch_stats = await run_workload(
        name="smartbatch",
        handler=smartbatch_handler,
        inputs=inputs,
        concurrency=args.concurrency,
    )

    if hasattr(smartbatch_handler, "batcher"):
        await smartbatch_handler.batcher.stop()

    plot_latency(output_dir, baseline_stats, smartbatch_stats)
    plot_throughput(output_dir, baseline_stats, smartbatch_stats)
    plot_single_percentile(
        output_dir,
        baseline_stats,
        smartbatch_stats,
        "p50",
        baseline_stats.p50,
        smartbatch_stats.p50,
    )
    plot_single_percentile(
        output_dir,
        baseline_stats,
        smartbatch_stats,
        "p95",
        baseline_stats.p95,
        smartbatch_stats.p95,
    )
    plot_single_percentile(
        output_dir,
        baseline_stats,
        smartbatch_stats,
        "p99",
        baseline_stats.p99,
        smartbatch_stats.p99,
    )

    save_summary(output_dir / "summary.json", baseline_stats, smartbatch_stats)
    save_samples_csv(output_dir / "samples.csv", baseline_stats, smartbatch_stats)
    print_summary(baseline_stats, smartbatch_stats, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs SmartBatch ResNet18 benchmark")
    parser.add_argument("--requests", type=int, default=64, help="Total requests per mode")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent in-flight requests")
    parser.add_argument("--workers", type=int, default=2, help="SmartBatch internal worker loops")
    parser.add_argument("--max-batch-size", type=int, default=32, help="SmartBatch max batch size")
    parser.add_argument("--max-wait-time", type=float, default=0.02, help="SmartBatch max wait time in seconds")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic image generation")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for synthetic inputs (e.g., 224 for standard ResNet input)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/resnet_baseline_vs_smartbatch",
        help="Directory where graphs and summary files are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))
