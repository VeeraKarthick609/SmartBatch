import asyncio
import time
import logging
from typing import Callable, List, Any, Optional, Union
from functools import wraps
from dataclasses import dataclass, field

logger = logging.getLogger("smartbatch.decorator")

@dataclass(order=True)
class _BatchedRequest:
    priority: int
    enqueue_time: float
    # Future is not comparable, so we exclude it from ordering or place it last
    # We use field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)

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
                 workers: int = 1):
        self.func = func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.input_schema = input_schema
        self.max_queue_size = max_queue_size
        self.workers = workers
        
        # PriorityQueue: (priority, timestamp, item)
        # Priority 0 = High (Retry/VIP)
        # Priority 10 = Normal
        self.queue: asyncio.PriorityQueue[_BatchedRequest] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.shutdown_event = asyncio.Event()
        self.worker_tasks: List[asyncio.Task] = []
        
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
                await asyncio.wait_for(asyncio.gather(*self.worker_tasks), timeout=5.0)
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
        try:
            # Try to enqueue immediately
            self.queue.put_nowait(req)
        except asyncio.QueueFull:
            # Queue is full. Back off and retry with higher priority.
            # logger.warning("Queue full, retrying with high priority...")
            await asyncio.sleep(0.05) # Wait 50ms
            
            # Retry with Priority 0 (High)
            req.priority = 0
            try:
                # Wait up to 1s to squeeze in
                await asyncio.wait_for(self.queue.put(req), timeout=1.0)
            except asyncio.TimeoutError:
                # Still full after retry -> Hard Failure
                raise RuntimeError("Service overloaded: Queue full after retry")
        
        return await future

    async def _worker_loop(self, worker_id: int):
        logger.info(f"Batch worker {worker_id} started for {self.func.__name__}")
        batch: List[_BatchedRequest] = []
        
        while not self.shutdown_event.is_set():
            try:
                # 1. Fetch first item (blocking wait)
                if not batch:
                    try:
                        req = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        continue
                
                # 2. Accumulate
                deadline = batch[0].enqueue_time + self.max_wait_time
                
                while len(batch) < self.max_batch_size:
                    now = time.time()
                    remaining = deadline - now
                    
                    if remaining <= 0:
                        break
                        
                    try:
                        req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break
                        
                # 3. Process
                if batch:
                    inputs = [b.payload for b in batch]
                    try:
                        # Call user function with optional worker_id injection
                        try:
                            if asyncio.iscoroutinefunction(self.func):
                                results = await self.func(inputs, worker_id=worker_id)
                            else:
                                results = await asyncio.to_thread(self.func, inputs, worker_id=worker_id)
                        except TypeError as te:
                             if "worker_id" in str(te):
                                 # Function doesn't accept worker_id
                                 if asyncio.iscoroutinefunction(self.func):
                                     results = await self.func(inputs)
                                 else:
                                     results = await asyncio.to_thread(self.func, inputs)
                             else:
                                 raise te
                            
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
          workers: int = 1):
    """
    Decorator to convert a List->List function into a Single->Single async function 
    that automatically batches requests in the background.
    """
    def decorator(func):
        # Create a single Batcher instance for this function definition
        batcher = Batcher(func, max_batch_size, max_wait_time, 
                          input_schema=input_schema, max_queue_size=max_queue_size, workers=workers)
        
        @wraps(func)
        async def wrapper(item: Any):
            return await batcher.process_single(item)
            
        # Expose batcher control? e.g. wrapper.batcher = batcher
        wrapper.batcher = batcher
        return wrapper
    return decorator
