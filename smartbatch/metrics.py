import time
from collections import deque


class SystemMetrics:
    def __init__(self, window_size: int = 200):
        self._latencies: deque = deque(maxlen=window_size)
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.time()
        self._batch_sizes: deque = deque(maxlen=window_size)
        self._batch_latencies: deque = deque(maxlen=window_size)
        self._total_batches: int = 0

    def record_request(self, latency: float, status: str = "success"):
        self._latencies.append(latency)
        self._total_requests += 1
        if status != "success":
            self._total_errors += 1

    def record_batch(self, batch_size: int, inference_time: float):
        self._batch_sizes.append(batch_size)
        self._batch_latencies.append(inference_time)
        self._total_batches += 1

    def get_stats(self) -> dict:
        latencies = sorted(self._latencies)
        n = len(latencies)

        def percentile(vals: list, p: float) -> float:
            if not vals:
                return 0.0
            idx = min(int(p * len(vals)), len(vals) - 1)
            return round(vals[idx], 4)

        batch_latencies = sorted(self._batch_latencies)
        avg_batch_size = round(sum(self._batch_sizes) / len(self._batch_sizes), 1) if self._batch_sizes else 0.0

        uptime = round(time.time() - self._start_time, 1)
        return {
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "latency_p50": percentile(latencies, 0.50),
            "latency_p95": percentile(latencies, 0.95),
            "latency_p99": percentile(latencies, 0.99),
            "total_batches": self._total_batches,
            "avg_batch_size": avg_batch_size,
            "batch_latency_p50": percentile(batch_latencies, 0.50),
            "batch_latency_p95": percentile(batch_latencies, 0.95),
        }


metrics = SystemMetrics()


def get_metrics() -> SystemMetrics:
    return metrics
