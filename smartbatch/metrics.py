import time
from collections import deque
from dataclasses import dataclass, field
from typing import List
import threading

@dataclass
class SystemMetrics:
    total_requests: int = 0
    total_batches: int = 0
    total_inference_time: float = 0.0
    batch_size_sum: int = 0
    
    # Keeping mostly aggregates to avoid memory leak over millions of reqs
    request_latencies: List[float] = field(default_factory=list) # CAUTION: Unlimited growth
    # Better: Use reservoir sampling or histograms? 
    # For now, just simplistic counters + recent latencies (deque)
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=10000)) 

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self, latency: float):
        with self._lock:
            self.total_requests += 1
            self.recent_latencies.append(latency)

    def record_batch(self, batch_size: int, inference_time: float):
        with self._lock:
            self.total_batches += 1
            self.batch_size_sum += batch_size
            self.total_inference_time += inference_time

    def get_stats(self):
        with self._lock:
            avg_batch = self.batch_size_sum / self.total_batches if self.total_batches > 0 else 0
            avg_inf_time = self.total_inference_time / self.total_batches if self.total_batches > 0 else 0
            
            p50 = 0.0
            p95 = 0.0
            if self.recent_latencies:
                sorted_lats = sorted(self.recent_latencies)
                n = len(sorted_lats)
                p50 = sorted_lats[int(n * 0.5)]
                p95 = sorted_lats[int(n * 0.95)]

            return {
                "total_requests": self.total_requests,
                "total_batches": self.total_batches,
                "avg_batch_size": round(avg_batch, 2),
                "avg_inference_time": round(avg_inf_time, 4),
                "p50_latency": round(p50, 4),
                "p95_latency": round(p95, 4),
                "rps_approx": 0 # TODO: Needs windowed counter
            }

# Global singleton
metrics = SystemMetrics()

def get_metrics():
    return metrics
