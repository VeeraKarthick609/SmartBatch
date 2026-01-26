from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

# --- Prometheus Metrics Definitions ---

# Total number of requests received
REQUEST_COUNT = Counter(
    'smartbatch_requests_total', 
    'Total number of inference requests',
    ['status'] # generic status label (success/error)
)

# Latency of individual requests (End-to-End)
REQUEST_LATENCY = Histogram(
    'smartbatch_request_duration_seconds', 
    'End-to-end request latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# Batch size distribution
BATCH_SIZE = Histogram(
    'smartbatch_batch_size', 
    'Distribution of batch sizes processed',
    buckets=[1, 4, 8, 16, 32, 64, 128]
)

# Processing time for the actual model inference (GPU time)
BATCH_LATENCY = Histogram(
    'smartbatch_batch_processing_seconds', 
    'Time taken for model inference batch processing'
)

class SystemMetrics:
    """
    Wrapper to maintain backward compatibility or helper methods
    But primarily just writes to the global Prometheus registry.
    """
    
    def record_request(self, latency: float, status: str = "success"):
        REQUEST_COUNT.labels(status=status).inc()
        REQUEST_LATENCY.observe(latency)

    def record_batch(self, batch_size: int, inference_time: float):
        BATCH_SIZE.observe(batch_size)
        BATCH_LATENCY.observe(inference_time)

    def get_stats(self):
        """
        Returns Prometheus formatted metrics string.
        """
        # Note: In a real app, you return Response(content, media_type)
        # But our api.py endpoint handles the wrapping, we just return raw bytes or wrapper
        return generate_latest()

# Global singleton
metrics = SystemMetrics()

def get_metrics():
    return metrics
