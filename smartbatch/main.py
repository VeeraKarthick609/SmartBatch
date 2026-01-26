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
    
    while True:
        try:
            # 1. Fetch first request (Blocking)
            # If batch is empty, we wait indefinitely for the first item
            if not batch:
                req = await queue.get()
                batch.append(req)
                
                # Start the timer for this batch
                deadline = asyncio.get_running_loop().time() + MAX_WAIT_TIME
            
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
                    results = model.infer(inputs)
                    
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
    logger.info("Shutting down...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="SmartBatch", lifespan=lifespan)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
