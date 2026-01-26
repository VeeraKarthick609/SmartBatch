import asyncio
import time
import logging
from typing import Callable, List, Any, Optional, Union
from functools import wraps
from dataclasses import dataclass

logger = logging.getLogger("smartbatch.decorator")

@dataclass
class _BatchedRequest:
    payload: Any
    future: asyncio.Future
    enqueue_time: float

class Batcher:
    """
    Manages the batching logic for a specific target function.
    """
    def __init__(self, 
                 func: Callable[[List[Any]], Any], 
                 max_batch_size: int, 
                 max_wait_time: float,
                 input_schema: Optional[Any] = None):
        self.func = func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.input_schema = input_schema
        self.queue: asyncio.Queue[_BatchedRequest] = asyncio.Queue()
        self.shutdown_event = asyncio.Event()
        self.worker_task: Optional[asyncio.Task] = None
        
    async def start(self):
        if self.worker_task is None or self.worker_task.done():
            self.shutdown_event.clear()
            self.worker_task = asyncio.create_task(self._worker_loop())
            
    async def stop(self):
        self.shutdown_event.set()
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            
    async def process_single(self, item: Any):
        """
        Public entry point: Enqueue a single item and wait for the batch result.
        """
        # --- Validation Layer ---
        if self.input_schema:
            # If item is a dict, convert to Pydantic model
            if isinstance(item, dict):
                try:
                    # Support Pydantic V2 and V1
                    if hasattr(self.input_schema, "model_validate"):
                        item = self.input_schema.model_validate(item)
                    else:
                        item = self.input_schema.parse_obj(item)
                except Exception as e:
                    # Re-raise as ValueError or let it bubble up (FastAPI handles it)
                    raise ValueError(f"Input validation failed: {e}")
            elif not isinstance(item, self.input_schema):
                 # Strict type check if not dict
                 raise TypeError(f"Expected {self.input_schema.__name__}, got {type(item)}")

        # Ensure worker is running (lazy start)
        if self.worker_task is None:
            await self.start()
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = _BatchedRequest(
            payload=item,
            future=future,
            enqueue_time=time.time()
        )
        
        await self.queue.put(req)
        return await future

    async def _worker_loop(self):
        logger.info(f"Batch worker started for {self.func.__name__}")
        batch: List[_BatchedRequest] = []
        
        while not self.shutdown_event.is_set():
            try:
                # 1. Fetch first item (blocking wait)
                if not batch:
                    # calculating wait based on nothing, just wait standard
                    try:
                        # If we have a shutdown signal, we might still want to process remaining items?
                        # For now, simple check
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
                        # Call the user's batched function
                        # Detect if it's a coroutine or regular func? 
                        # We assume async for now as per design
                        if asyncio.iscoroutinefunction(self.func):
                            results = await self.func(inputs)
                        else:
                            # Run sync function in thread pool to avoid blocking loop
                            results = await asyncio.to_thread(self.func, inputs)
                            
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
                logger.error(f"Worker crashed: {e}")
                await asyncio.sleep(1)

def batch(max_batch_size: int = 32, max_wait_time: float = 0.01, input_schema: Optional[Any] = None):
    """
    Decorator to convert a List->List function into a Single->Single async function 
    that automatically batches requests in the background.
    """
    def decorator(func):
        # Create a single Batcher instance for this function definition
        # Note: This means all calls to this decorated function share the same queue
        batcher = Batcher(func, max_batch_size, max_wait_time, input_schema=input_schema)
        
        @wraps(func)
        async def wrapper(item: Any):
            return await batcher.process_single(item)
            
        # Expose batcher control? e.g. wrapper.batcher = batcher
        wrapper.batcher = batcher
        return wrapper
    return decorator
