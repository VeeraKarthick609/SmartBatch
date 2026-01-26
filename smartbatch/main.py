import asyncio
import logging
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI
from smartbatch.api import router
from smartbatch.batching import init_request_queue, get_request_queue, InferenceRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartbatch")

import os

# Configuration from Environment Variables
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
MAX_WAIT_TIME = float(os.getenv("MAX_WAIT_TIME", "0.01")) # 10ms
MODEL_PATH = os.getenv("MODEL_PATH", None)

async def batch_worker():
    """
    Core Batch Processing Loop.
    Accumulates requests until MAX_BATCH_SIZE or MAX_WAIT_TIME is reached.
    """
    logger = logging.getLogger("smartbatch.worker")
    logger.info(f"Batch worker started with BatchSize={MAX_BATCH_SIZE}, Wait={MAX_WAIT_TIME}")
    
    # Initialize Model
    from smartbatch.model import ModelWrapper
    # Lazy init will happen inside model wrapper, but we pass path here
    model = ModelWrapper(model_path=MODEL_PATH)
    # Trigger load immediately to be ready (optional, but good for logs)
    # model.load() 
    
    queue = get_request_queue()
    
    batch: List[InferenceRequest] = []
    from smartbatch.batching import get_shutdown_event
    shutdown_event = get_shutdown_event()
    
    # Loop while running OR (Shutting down AND Queue has items)
    while not shutdown_event.is_set() or not queue.empty() or batch:
        try:
            # 1. Fetch first request (Blocking)
            # If batch is empty, we wait indefinitely for the first item
            if not batch:
                if shutdown_event.is_set() and queue.empty():
                    break # Done
                
                try:
                    # If shutting down, don't wait forever
                    timeout = 1.0 if shutdown_event.is_set() else None
                    if timeout:
                         req = await asyncio.wait_for(queue.get(), timeout=timeout)
                    else:
                         req = await queue.get()
                         
                    batch.append(req)
                    # Start the timer for this batch
                    deadline = asyncio.get_running_loop().time() + MAX_WAIT_TIME
                except asyncio.TimeoutError:
                     # Queue empty and ensure shutdown check loop
                     continue

            # 2. Accumulate more requests until triggered
            
            # 2. Accumulate more requests until triggered
            while len(batch) < MAX_BATCH_SIZE:
                now = asyncio.get_running_loop().time()
                remaining = deadline - now
                
                if remaining <= 0:
                    logger.debug("Batch timeout triggered")
                    break
                
                try:
                    # Wait for next item with timeout
                    req = await asyncio.wait_for(queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    logger.debug("Batch timeout reached during wait")
                    break
            
            # 3. Process Batch
            if batch:
                logger.info(f"Processing batch of size {len(batch)}")
                
                # Extract payloads
                inputs = [req.payload for req in batch]
                
                try:
                    # Run Inference
                    t0_inf = asyncio.get_running_loop().time()
                    results = model.infer(inputs)
                    t1_inf = asyncio.get_running_loop().time()
                    
                    # Record Batch Metrics
                    from smartbatch.metrics import get_metrics
                    get_metrics().record_batch(len(batch), t1_inf - t0_inf)

                    # Set Results
                    for req, res in zip(batch, results):
                        if not req.future.done():
                            req.future.set_result(res)
                            
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}")
                    # Fail all requests in this batch
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(e)
                finally:
                    # Mark tasks as done in queue
                    for _ in batch:
                        queue.task_done()
                    
                    # Reset batch
                    batch = []

        except asyncio.CancelledError:
            logger.info("Worker cancelled")
            break
        except Exception as e:
            logger.error(f"Worker critical error: {e}")
            # Prevent busy loop if something goes really wrong
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing SmartBatch...")
    init_request_queue(maxsize=100)
    
    # Start worker
    worker_task = asyncio.create_task(batch_worker())
    
    yield
    
    # Shutdown
    logger.info("Shutting down... Draining queue.")
    
    # 1. Signal shutdown to API (stop accepting new reqs)
    from smartbatch.batching import get_shutdown_event
    get_shutdown_event().set()
    
    # 2. Wait for worker to finish (it should Exit only when queue is empty)
    # We might need to cancel it if it takes too long, but let's try graceful first
    try:
        # Give it some time to drain? Or wait indefinitely?
        # For Day 3 Goal: "Drain queue".
        # We need to update worker loop condition first!
        # The worker currently loops `while True`. 
        # We need to signal the worker to stop? 
        # Actually, if we set the event, we can change the worker loop condition.
        await asyncio.wait_for(worker_task, timeout=10.0) # 10s grace
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out, cancelling worker...")
        worker_task.cancel()
    except asyncio.CancelledError:
        pass

app = FastAPI(title="SmartBatch", lifespan=lifespan)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
