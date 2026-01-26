import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from smartbatch.api import router, run_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartbatch")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing SmartBatch...")
    # Trigger model load (optional, but good for cold start)
    # await run_inference.batcher.start() 
    
    yield
    
    # Shutdown
    logger.info("Shutting down... Stopping batcher.")
    await run_inference.batcher.stop()

app = FastAPI(title="SmartBatch", lifespan=lifespan)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
