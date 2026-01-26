import logging
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from smartbatch.api import router, run_inference
from smartbatch.registry import register

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartbatch")

# --- Register Default Model ---
# We register the existing 'run_inference' handler as 'default'
# enabling POST /models/default/predict
# The original POST /predict still works by calling run_inference directly
register("default")(run_inference)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing SmartBatch...")
    
    # Optional: We could iterate over registry and pre-warm models here
    
    yield
    
    # Shutdown
    logger.info("Shutting down... Stopping batcher.")
    await run_inference.batcher.stop() 
    # TODO: In a real multi-model setup, we should iterate _registry and stop all batchers
    # For now, we only have one main one active in this demo.

app = FastAPI(title="SmartBatch", lifespan=lifespan)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
