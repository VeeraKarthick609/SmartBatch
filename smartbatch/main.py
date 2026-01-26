import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from smartbatch.api import router
from smartbatch.batching import init_request_queue, get_request_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartbatch")

async def batch_worker():
    """
    Dummy worker for Day 1.
    Dequeues requests and returns immediately.
    """
    logger.info("Batch worker started")
    queue = get_request_queue()
    
    while True:
        try:
            req = await queue.get()
            
            # Day 1 Logic: Immediate dummy result
            # Log enqueue/dequeue latency
            now = asyncio.get_running_loop().time()
            # Note: req.enqueue_time is from time.time(), so we mix clocks slightly but okay for logging
            
            logger.info(f"Processing request {req.request_id}")
            
            # Simulate work? No, requirement says "Immediate future.set_result"
            # But making it async/await safe
            req.future.set_result([0.99] * len(req.payload) if isinstance(req.payload, list) else "dummy_result")
            
            queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("Worker cancelled")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")

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
