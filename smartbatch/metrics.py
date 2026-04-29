import time
from collections import deque


class SystemMetrics:
    def __init__(self, window_size: int = 200):
        self._latencies: deque = deque(maxlen=window_size)
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.time()

    def record_request(self, latency: float, status: str = "success"):
        self._latencies.append(latency)
        self._total_requests += 1
        if status != "success":
            self._total_errors += 1

    def record_batch(self, batch_size: int, inference_time: float):
        pass  # kept for API compatibility; batch stats live in Batcher.processing_latencies

    def get_stats(self) -> dict:
        latencies = sorted(self._latencies)
        n = len(latencies)

        def percentile(p: float) -> float:
            if not latencies:
                return 0.0
            idx = min(int(p * n), n - 1)
            return round(latencies[idx], 4)

        uptime = round(time.time() - self._start_time, 1)
        return {
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "latency_p50": percentile(0.50),
            "latency_p95": percentile(0.95),
            "latency_p99": percentile(0.99),
        }


metrics = SystemMetrics()


def get_metrics() -> SystemMetrics:
    return metrics
