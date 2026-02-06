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
        # Moving average for adaptive batching
        self.recent_latencies: deque = deque(maxlen=10)

        
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
                        if self.target_latency:
                             self.recent_latencies.append(exec_duration)
                             avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
                             
                             # Multiplicative Decrease
                             if avg_latency > self.target_latency:
                                 old_limit = self.current_batch_size_limit
                                 self.current_batch_size_limit = max(1, int(self.current_batch_size_limit * 0.8))
                                 if self.current_batch_size_limit != old_limit:
                                     logger.info(f"High latency (avg {avg_latency:.3f}s). Reducing batch size to {self.current_batch_size_limit}")
                                     self.recent_latencies.clear() # Reset to avoid double penalizing
                             
                             # Additive Increase
                             elif avg_latency < self.target_latency * 0.8:
                                  if self.current_batch_size_limit < self.max_batch_size:
                                      self.current_batch_size_limit += 1 
                                      logger.debug(f"Low latency. Increasing batch size to {self.current_batch_size_limit}")
                                      # No clear here, gradual increase is fine
                            
                        if len(results) != len(inputs):
                             raise ValueError(f"Batch function returned {len(results)} items, expected {len(inputs)}")
                             
                        for req, res in zip(batch, results):
                            if not req.future.done():
                                req.future.set_result(res)
                                
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        for req in batch:
                            if not req.future.done():
                                req.future.set_exception(e)
                    finally:
                         batch = []
                         
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} crashed: {e}")
                await asyncio.sleep(1)

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
