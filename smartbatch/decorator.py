import asyncio
import time
import logging
import inspect
from typing import Callable, List, Any, Optional, Union
from functools import wraps
from dataclasses import dataclass, field
from collections import deque
from smartbatch.exceptions import OverloadedError

logger = logging.getLogger("smartbatch.decorator")

@dataclass(order=True)
class _BatchedRequest:
    priority: int
    enqueue_time: float
    # Future is not comparable, so we exclude it from ordering or place it last
    # We use field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)

logging.basicConfig(level=logging.INFO)

# Constants
RETRY_SLEEP_INTERVAL = 0.05
RETRY_TIMEOUT = 1.0
WORKER_SHUTDOWN_TIMEOUT = 5.0
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 10.0

class MetricTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.times = deque(maxlen=window_size)
    
    def add(self, value: float, count: int = 1):
        """Add value(s). For rate calculation, count represents number of events."""
        now = time.time()
        for _ in range(count):
            self.values.append(value)
            self.times.append(now)

    def get_rate(self) -> float:
        """Calculate rate (items/sec) based on window."""
        if len(self.times) < 2:
            return 0.0
        duration = self.times[-1] - self.times[0]
        if duration <= 1e-6: # Avoid div by zero
            return 0.0
        return len(self.times) / duration

    def get_p95(self) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        index = int(0.95 * len(sorted_vals))
        return sorted_vals[min(index, len(sorted_vals) - 1)]

class CircuitBreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD, 
                 recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failures = 0
        self.last_failure_time = 0.0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit Breaker OPENED after {self.failures} failures.")

    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failures = 0
            logger.info("Circuit Breaker CLOSED (Recovery successful).")
        elif self.state == CircuitBreakerState.CLOSED:
             self.failures = 0

    def can_proceed(self) -> bool:
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.warning("Circuit Breaker HALF_OPEN (Probing).")
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Simple allow single request or all? Let's allow all in half-open for now, 
            # but usually you'd want rate limiting here.
            return True
        return True

class Batcher:
    """
    Manages the batching logic for a specific target function.
    Uses PriorityQueue for adaptive backpressure.
    """
    def __init__(self, 
                 func: Callable[[List[Any]], Any], 
                 max_batch_size: int, 
                 max_wait_time: float,
                 input_schema: Optional[Any] = None,
                 max_queue_size: int = 128,
                 workers: int = 1,
                 target_latency: Optional[float] = None):
        self.func = func
        self.max_batch_size = max_batch_size
        self.current_batch_size_limit = max_batch_size # Adaptive limit
        self.max_wait_time = max_wait_time
        self.input_schema = input_schema
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.target_latency = target_latency
        
        # PriorityQueue: (priority, timestamp, item)
        # Priority 0 = High (Retry/VIP)
        # Priority 10 = Normal
        # Per-GPU/Worker queues
        # Handle 0 workers case carefully (for testing backpressure)
        self.queues: List[asyncio.PriorityQueue[_BatchedRequest]] = [
            asyncio.PriorityQueue(maxsize=max_queue_size) for _ in range(max(1, workers))
        ]

        self.shutdown_event = asyncio.Event()
        self.worker_tasks: List[asyncio.Task] = []
        # Metrics for adaptive batching and admission control
        self.processing_latencies = MetricTracker(window_size=20)
        self.throughput_tracker = MetricTracker(window_size=50) # Track items processed per second
        
        # Fault Isolation
        self.circuit_breaker = CircuitBreaker()

        
    async def start(self):
        # Clean up finished tasks
        self.worker_tasks = [t for t in self.worker_tasks if not t.done()]
        
        if not self.worker_tasks:
            self.shutdown_event.clear()
            for i in range(self.workers):
                t = asyncio.create_task(self._worker_loop(worker_id=i))
                self.worker_tasks.append(t)
            
    async def stop(self):
        self.shutdown_event.set()
        if self.worker_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*self.worker_tasks), timeout=WORKER_SHUTDOWN_TIMEOUT)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            
    async def process_single(self, item: Any, priority: int = 10):
        """
        Public entry point: Enqueue a single item and wait for the batch result.
        """
        # --- Validation Layer ---
        if self.input_schema:
            if isinstance(item, dict):
                try:
                    if hasattr(self.input_schema, "model_validate"):
                        item = self.input_schema.model_validate(item)
                    else:
                        item = self.input_schema.parse_obj(item)
                except Exception as e:
                    raise ValueError(f"Input validation failed: {e}")
            elif not isinstance(item, self.input_schema):
                 raise TypeError(f"Expected {self.input_schema.__name__}, got {type(item)}")

        # Ensure worker is running
        if not self.circuit_breaker.can_proceed():
             raise OverloadedError("Service temporarily unavailable (Circuit Breaker OPEN)")

        # Admission Control (Little's Law)
        # Estimated Wait = Total Queue Size / Throughput
        current_throughput = self.throughput_tracker.get_rate()
        total_queued = sum(q.qsize() for q in self.queues)
        
        if current_throughput > 0:
            sla_limit = self.max_wait_time + (self.target_latency or 0.1)
            estimated_wait = total_queued / current_throughput
            
            # 1.5x grace factor to accommodate bursts
            if estimated_wait > sla_limit * 1.5:
                # Allow priority 0 (retry/high) to pass? 
                # For now strict rejection to protect system.
                if priority > 0:
                    raise OverloadedError(f"System overloaded. Est wait {estimated_wait:.3f}s > limit {sla_limit*1.5:.3f}s")

        if not self.worker_tasks:
            await self.start()
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        req = _BatchedRequest(
            priority=priority,
            enqueue_time=time.time(),
            payload=item,
            future=future
        )
        
        # --- Backpressure / Retry Logic ---
        
        # Load Balancing: Find shortest queue
        # For approximation, we just check qsize. 
        # In high concurrency, this is racy but acceptable.
        
        # If workers=0 (testing), we just use queue[0]
        # But we need to simulate load balancing if workers > 1.
        
        best_queue_idx = 0
        if self.workers > 1:
             # Find index with min size
             best_queue_idx = min(range(len(self.queues)), key=lambda i: self.queues[i].qsize())
        
        target_queue = self.queues[best_queue_idx]

        try:
            # Try to enqueue immediately
            target_queue.put_nowait(req)
        except asyncio.QueueFull:
            # Queue is full. Back off and retry with higher priority.
            logger.warning("Queue full, retrying with high priority...")
            await asyncio.sleep(RETRY_SLEEP_INTERVAL) # Wait 50ms
            
            # Retry with Priority 0 (High)
            req.priority = 0
            try:
                # Wait up to 1s to squeeze in
                await asyncio.wait_for(target_queue.put(req), timeout=RETRY_TIMEOUT)
            except asyncio.TimeoutError:
                # Still full after retry -> Hard Failure
                raise OverloadedError("Service overloaded: Queue full after retry")
        
        return await future

    async def _worker_loop(self, worker_id: int):
        logger.info(f"Batch worker {worker_id} started for {self.func.__name__}")
        batch: List[_BatchedRequest] = []
        
        # Determine which queue this worker listens to. 
        # If workers match queues, 1:1 map.
        my_queue = self.queues[worker_id % len(self.queues)]
        
        while not self.shutdown_event.is_set():
            try:
                # 1. Fetch first item (blocking wait)
                if not batch:
                    try:
                        req = await asyncio.wait_for(my_queue.get(), timeout=1.0)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        continue
                
                # 2. Accumulate
                # Use current time for deadline to maximize batching opportunity, 
                # instead of enqueue time which might be stale.
                # Or use enqueue time if we want to enforce strict total latency.
                # Given "max_wait_time" usually implies "wait for more items", base it on first item fetch.
                deadline = time.time() + self.max_wait_time
                
                while len(batch) < self.current_batch_size_limit:
                    now = time.time()
                    remaining = deadline - now
                    
                    if remaining <= 0:
                        # Try to fill batch from queue without waiting
                        try:
                            while len(batch) < self.current_batch_size_limit:
                                req = my_queue.get_nowait()
                                batch.append(req)
                        except asyncio.QueueEmpty:
                            pass
                        break
                        
                    try:
                        req = await asyncio.wait_for(my_queue.get(), timeout=remaining)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break
                        
                # 3. Process
                if batch:
                    inputs = [b.payload for b in batch]
                    try:
                        start_time = time.time()
                        # Call user function with optional worker_id injection
                        
                        # Inspect function signature once (or cache it if performance is critical, 
                        # but for batch processing overhead is negligible)
                        sig = inspect.signature(self.func)
                        accepts_worker_id = "worker_id" in sig.parameters

                        if asyncio.iscoroutinefunction(self.func):
                            if accepts_worker_id:
                                results = await self.func(inputs, worker_id=worker_id)
                            else:
                                results = await self.func(inputs)
                        else:
                            if accepts_worker_id:
                                results = await asyncio.to_thread(self.func, inputs, worker_id=worker_id)
                            else:
                                results = await asyncio.to_thread(self.func, inputs)

                        exec_duration = time.time() - start_time
                        logger.debug(f"Batch size: {len(batch)}, Duration: {exec_duration:.4f}s, Target: {self.target_latency}")

                        # --- Adaptive Batching Logic ---
                        # Update metrics
                        self.processing_latencies.add(exec_duration)
                        self.throughput_tracker.add(1, count=len(batch))

                        if self.target_latency:
                             p95_latency = self.processing_latencies.get_p95()
                             
                             # Multiplicative Decrease
                             if p95_latency > self.target_latency:
                                 old_limit = self.current_batch_size_limit
                                 self.current_batch_size_limit = max(1, int(self.current_batch_size_limit * 0.8))
                                 if self.current_batch_size_limit != old_limit:
                                     logger.info(f"High P95 latency ({p95_latency:.3f}s). Reducing batch size to {self.current_batch_size_limit}")
                                     # We don't clear metrics immediately to allow stabilization
                             
                             # Additive Increase
                             # Only increase if we are meeting SLA comfortably AND utilizing current batch size
                             elif p95_latency < self.target_latency * 0.8:
                                  if len(batch) >= self.current_batch_size_limit * 0.8:
                                      if self.current_batch_size_limit < self.max_batch_size:
                                          self.current_batch_size_limit += 1 
                                          logger.debug(f"Low latency. Increasing batch size to {self.current_batch_size_limit}")
                            
                        if len(results) != len(inputs):
                             raise ValueError(f"Batch function returned {len(results)} items, expected {len(inputs)}")
                             
                        for req, res in zip(batch, results):
                            if not req.future.done():
                                req.future.set_result(res)
                        
                        # Successful batch
                        self.circuit_breaker.record_success()
                                
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}. Attempting recovery/fallback.")
                        # Partial Failure / Fallback Strategy
                        # We suspect one item might be causing the issue.
                        # Strategy: Retry items individually (or in smaller chunks).
                        
                        # But first, record failure in circuit breaker
                        self.circuit_breaker.record_failure()
                        
                        await self._process_fallback_individually(batch, worker_id)

                    finally:
                         batch = []
                         
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} crashed: {e}")
                await asyncio.sleep(1)

    async def _process_fallback_individually(self, batch: List[_BatchedRequest], worker_id: int):
        """
        Fallback mechanism: Process each item in the failed batch individually.
        This isolates the failure to the specific item(s) causing it.
        """
        logger.info(f"Fallback: Processing {len(batch)} items individually.")
        for req in batch:
            if req.future.done():
                continue
                
            try:
                # Process single item
                single_batch = [req.payload]
                
                # We need to call the user function.
                # NOTE: This assumes the user function handles list of length 1 correctly.
                # Most batch functions should.
                
                # .. duplicate call logic ..
                sig = inspect.signature(self.func)
                accepts_worker_id = "worker_id" in sig.parameters

                if asyncio.iscoroutinefunction(self.func):
                    if accepts_worker_id:
                        results = await self.func(single_batch, worker_id=worker_id)
                    else:
                        results = await self.func(single_batch)
                else:
                    if accepts_worker_id:
                        results = await asyncio.to_thread(self.func, single_batch, worker_id=worker_id)
                    else:
                        results = await asyncio.to_thread(self.func, single_batch)

                if len(results) != 1:
                     raise ValueError(f"Fallback expected 1 result, got {len(results)}")
                
                req.future.set_result(results[0])
                
            except Exception as e:
                logger.error(f"Fallback failed for specific item: {e}")
                req.future.set_exception(e)

def batch(max_batch_size: int = 32, 
          max_wait_time: float = 0.01, 
          input_schema: Optional[Any] = None,
          max_queue_size: int = 128,
          workers: int = 1,
          target_latency: Optional[float] = None):
    """
    Decorator to convert a List->List function into a Single->Single async function 
    that automatically batches requests in the background.
    """
    def decorator(func):
        # Create a single Batcher instance for this function definition
        batcher = Batcher(func, max_batch_size, max_wait_time, 
                          input_schema=input_schema, max_queue_size=max_queue_size, workers=workers,
                          target_latency=target_latency)
        
        @wraps(func)
        async def wrapper(item: Any):
            return await batcher.process_single(item)
            
        # Expose batcher control? e.g. wrapper.batcher = batcher
        wrapper.batcher = batcher
        return wrapper
    return decorator
