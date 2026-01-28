import logging
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from smartbatch.api import router
from smartbatch.registry import register

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("smartbatch")

# --- Registry Initialization ---
# The registry starts empty. Users import 'smartbatch' and register their own models.
# Example: 
# @register(name="my-model")
# @batch
# def my_func(batch): ...

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing SmartBatch...")
    yield
    # Shutdown
    logger.info("Shutting down...")
    # TODO: Signal all active batchers to stop?
    # For now, individual batchers manage their own tasks/daemons or rely on process exit.

app = FastAPI(title="SmartBatch", lifespan=lifespan)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
