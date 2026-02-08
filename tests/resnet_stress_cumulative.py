"""
Cumulative stress benchmark for Baseline vs SmartBatch using ResNet18.

This benchmark is designed for high request volumes (e.g., 1k -> 100k) and
reports metrics at cumulative request checkpoints.

Outputs (in --output-dir):
- cumulative_latency_mean.png
- cumulative_throughput.png
- cumulative_p50.png
- cumulative_p95.png
- cumulative_p99.png
- cumulative_metrics.csv
- summary.json

Usage:
    ./venv/bin/python tests/resnet_stress_cumulative.py \
      --checkpoints 1000,5000,10000,25000,50000,75000,100000 \
      --concurrency 256 \
      --workers 4
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
from typing import Any, Callable, Dict, List, Sequence

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


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def parse_checkpoints(raw: str) -> List[int]:
    checkpoints = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("checkpoints must be positive integers")
        checkpoints.append(value)
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    return sorted(set(checkpoints))


@dataclass
class WorkerBundle:
    model: torch.nn.Module
    device: torch.device
    lock: Lock


@dataclass
class RunTrace:
    name: str
    latencies: List[float]          # ordered by completion index
    completion_elapsed: List[float] # ordered by completion index
    total_time: float
    success: int
    failures: int


def build_worker_bundle(worker_id: int) -> WorkerBundle:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    model = resnet18(weights=None)
    model.eval()
    model.to(device)
    return WorkerBundle(model=model, device=device, lock=Lock())


def make_input_pool(pool_size: int, image_size: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((3, image_size, image_size), dtype=np.float32) for _ in range(pool_size)]


def run_model_sync(bundle: WorkerBundle, batch_inputs: List[np.ndarray]) -> List[int]:
    tensor_batch = torch.stack([torch.from_numpy(arr) for arr in batch_inputs]).to(bundle.device)
    with bundle.lock:
        with torch.inference_mode():
            logits = bundle.model(tensor_batch)
    top1 = torch.argmax(logits, dim=1)
    return top1.cpu().tolist()


def build_baseline_handler(bundle: WorkerBundle) -> Callable[[np.ndarray], Any]:
    async def infer_single(image: np.ndarray) -> int:
        result = run_model_sync(bundle, [image])
        return result[0]

    return infer_single


def build_smartbatch_handler(
    bundles: Dict[int, WorkerBundle],
    max_batch_size: int,
    max_wait_time: float,
) -> Any:
    @batch(
        max_batch_size=max_batch_size,
        max_wait_time=max_wait_time,
        workers=len(bundles),
        max_queue_size=4096,
        target_latency=0.2,
    )
    async def infer_batch(batch_inputs: List[np.ndarray], worker_id: int = 0) -> List[int]:
        bundle = bundles[worker_id % len(bundles)]
        return run_model_sync(bundle, batch_inputs)

    return infer_batch


async def run_parallel_workload(
    name: str,
    handler: Callable[[np.ndarray], Any],
    total_requests: int,
    concurrency: int,
    input_pool: List[np.ndarray],
) -> RunTrace:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for index in range(total_requests):
        queue.put_nowait(index)

    latencies: List[float] = []
    completion_elapsed: List[float] = []
    failures = 0
    started = time.perf_counter()

    async def worker() -> None:
        nonlocal failures
        while True:
            try:
                req_idx = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            payload = input_pool[req_idx % len(input_pool)]
            t0 = time.perf_counter()
            try:
                await handler(payload)
            except Exception:
                failures += 1
            finally:
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                completion_elapsed.append(t1 - started)
                queue.task_done()

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - started
    success = total_requests - failures

    return RunTrace(
        name=name,
        latencies=latencies,
        completion_elapsed=completion_elapsed,
        total_time=total_time,
        success=success,
        failures=failures,
    )


def build_cumulative_rows(trace: RunTrace, checkpoints: List[int]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    available = len(trace.latencies)
    for checkpoint in checkpoints:
        if checkpoint > available:
            break
        sample = trace.latencies[:checkpoint]
        elapsed = trace.completion_elapsed[checkpoint - 1]
        throughput = (checkpoint / elapsed) if elapsed > 0 else 0.0
        row = {
            "checkpoint": float(checkpoint),
            "mean_latency": float(np.mean(sample)),
            "p50_latency": percentile(sample, 50),
            "p95_latency": percentile(sample, 95),
            "p99_latency": percentile(sample, 99),
            "throughput_rps": throughput,
        }
        rows.append(row)
    return rows


def rows_to_map(rows: List[Dict[str, float]], key: str) -> Dict[int, float]:
    return {int(row["checkpoint"]): float(row[key]) for row in rows}


def plot_metric(
    output_path: Path,
    checkpoints: List[int],
    baseline_rows: List[Dict[str, float]],
    smartbatch_rows: List[Dict[str, float]],
    key: str,
    title: str,
    y_label: str,
) -> None:
    baseline_map = rows_to_map(baseline_rows, key)
    smartbatch_map = rows_to_map(smartbatch_rows, key)
    x = [cp for cp in checkpoints if cp in baseline_map and cp in smartbatch_map]
    baseline_y = [baseline_map[cp] for cp in x]
    smartbatch_y = [smartbatch_map[cp] for cp in x]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, baseline_y, marker="o", label="Baseline", color="#e76f51")
    ax.plot(x, smartbatch_y, marker="o", label="SmartBatch", color="#2a9d8f")
    ax.set_title(title)
    ax.set_xlabel("Cumulative Completed Requests")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_cumulative_csv(
    output_path: Path,
    checkpoints: List[int],
    baseline_rows: List[Dict[str, float]],
    smartbatch_rows: List[Dict[str, float]],
) -> None:
    baseline_by_cp = {int(row["checkpoint"]): row for row in baseline_rows}
    smartbatch_by_cp = {int(row["checkpoint"]): row for row in smartbatch_rows}

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "baseline_mean_latency",
                "baseline_p50",
                "baseline_p95",
                "baseline_p99",
                "baseline_throughput_rps",
                "smartbatch_mean_latency",
                "smartbatch_p50",
                "smartbatch_p95",
                "smartbatch_p99",
                "smartbatch_throughput_rps",
            ]
        )

        for checkpoint in checkpoints:
            if checkpoint not in baseline_by_cp or checkpoint not in smartbatch_by_cp:
                continue
            b = baseline_by_cp[checkpoint]
            s = smartbatch_by_cp[checkpoint]
            writer.writerow(
                [
                    checkpoint,
                    b["mean_latency"],
                    b["p50_latency"],
                    b["p95_latency"],
                    b["p99_latency"],
                    b["throughput_rps"],
                    s["mean_latency"],
                    s["p50_latency"],
                    s["p95_latency"],
                    s["p99_latency"],
                    s["throughput_rps"],
                ]
            )


def write_summary_json(output_path: Path, baseline_trace: RunTrace, smartbatch_trace: RunTrace) -> None:
    summary = {
        "baseline": {
            "total_time": baseline_trace.total_time,
            "success": baseline_trace.success,
            "failures": baseline_trace.failures,
            "throughput_rps": (baseline_trace.success / baseline_trace.total_time)
            if baseline_trace.total_time > 0
            else 0.0,
        },
        "smartbatch": {
            "total_time": smartbatch_trace.total_time,
            "success": smartbatch_trace.success,
            "failures": smartbatch_trace.failures,
            "throughput_rps": (smartbatch_trace.success / smartbatch_trace.total_time)
            if smartbatch_trace.total_time > 0
            else 0.0,
        },
    }
    output_path.write_text(json.dumps(summary, indent=2))


async def benchmark(args: argparse.Namespace) -> None:
    checkpoints = parse_checkpoints(args.checkpoints)
    max_requests = max(checkpoints)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.torch_num_threads))
    if args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    input_pool = make_input_pool(
        pool_size=args.pool_size,
        image_size=args.image_size,
        seed=args.seed,
    )

    baseline_bundle = build_worker_bundle(worker_id=0)
    baseline_handler = build_baseline_handler(baseline_bundle)

    smartbatch_bundles = {idx: build_worker_bundle(worker_id=idx) for idx in range(args.workers)}
    smartbatch_handler = build_smartbatch_handler(
        bundles=smartbatch_bundles,
        max_batch_size=args.max_batch_size,
        max_wait_time=args.max_wait_time,
    )

    print(f"Running baseline workload: total_requests={max_requests}, concurrency={args.concurrency}")
    baseline_trace = await run_parallel_workload(
        name="baseline",
        handler=baseline_handler,
        total_requests=max_requests,
        concurrency=args.concurrency,
        input_pool=input_pool,
    )

    print(f"Running smartbatch workload: total_requests={max_requests}, concurrency={args.concurrency}")
    smartbatch_trace = await run_parallel_workload(
        name="smartbatch",
        handler=smartbatch_handler,
        total_requests=max_requests,
        concurrency=args.concurrency,
        input_pool=input_pool,
    )

    if hasattr(smartbatch_handler, "batcher"):
        await smartbatch_handler.batcher.stop()

    baseline_rows = build_cumulative_rows(baseline_trace, checkpoints)
    smartbatch_rows = build_cumulative_rows(smartbatch_trace, checkpoints)

    plot_metric(
        output_dir / "cumulative_latency_mean.png",
        checkpoints,
        baseline_rows,
        smartbatch_rows,
        key="mean_latency",
        title="Mean Latency vs Cumulative Requests",
        y_label="Latency (s)",
    )
    plot_metric(
        output_dir / "cumulative_throughput.png",
        checkpoints,
        baseline_rows,
        smartbatch_rows,
        key="throughput_rps",
        title="Throughput vs Cumulative Requests",
        y_label="Throughput (req/s)",
    )
    plot_metric(
        output_dir / "cumulative_p50.png",
        checkpoints,
        baseline_rows,
        smartbatch_rows,
        key="p50_latency",
        title="P50 Latency vs Cumulative Requests",
        y_label="Latency (s)",
    )
    plot_metric(
        output_dir / "cumulative_p95.png",
        checkpoints,
        baseline_rows,
        smartbatch_rows,
        key="p95_latency",
        title="P95 Latency vs Cumulative Requests",
        y_label="Latency (s)",
    )
    plot_metric(
        output_dir / "cumulative_p99.png",
        checkpoints,
        baseline_rows,
        smartbatch_rows,
        key="p99_latency",
        title="P99 Latency vs Cumulative Requests",
        y_label="Latency (s)",
    )

    write_cumulative_csv(
        output_path=output_dir / "cumulative_metrics.csv",
        checkpoints=checkpoints,
        baseline_rows=baseline_rows,
        smartbatch_rows=smartbatch_rows,
    )
    write_summary_json(
        output_path=output_dir / "summary.json",
        baseline_trace=baseline_trace,
        smartbatch_trace=smartbatch_trace,
    )

    print("\nCumulative Stress Summary")
    print(f"- Checkpoints: {checkpoints}")
    print(
        f"- Baseline: success={baseline_trace.success}, failures={baseline_trace.failures}, "
        f"total_time={baseline_trace.total_time:.2f}s"
    )
    print(
        f"- SmartBatch: success={smartbatch_trace.success}, failures={smartbatch_trace.failures}, "
        f"total_time={smartbatch_trace.total_time:.2f}s"
    )
    print(f"- Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cumulative stress benchmark (Baseline vs SmartBatch)")
    parser.add_argument(
        "--checkpoints",
        default="1000,5000,10000,25000,50000,75000,100000",
        help="Comma-separated cumulative request checkpoints.",
    )
    parser.add_argument("--concurrency", type=int, default=256, help="Parallel in-flight requests.")
    parser.add_argument("--workers", type=int, default=4, help="SmartBatch internal worker loops.")
    parser.add_argument("--max-batch-size", type=int, default=32, help="SmartBatch max batch size.")
    parser.add_argument("--max-wait-time", type=float, default=0.02, help="SmartBatch max wait time.")
    parser.add_argument("--pool-size", type=int, default=64, help="Reusable synthetic input pool size.")
    parser.add_argument("--image-size", type=int, default=224, help="Synthetic image H/W for ResNet input.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="torch intra-op thread count for more stable stress runs.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=0,
        help="torch inter-op thread count (0 means leave default).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/resnet_stress_cumulative",
        help="Directory for graphs and metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))
