import asyncio
import uuid
import time
import os
import logging
from typing import List, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from smartbatch.decorator import batch
from smartbatch.model import ModelWrapper
from smartbatch.metrics import get_metrics
from smartbatch.exceptions import OverloadedError

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Configuration ---
# API configuration can go here (e.g. auth middleware)

# --- API Endpoints ---

from starlette.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

@router.get("/metrics")
def metrics_endpoint():
    data = get_metrics().get_stats()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

class PredictRequest(BaseModel):
    data: Any 

class PredictResponse(BaseModel):
    result: Any
    request_id: str
    processing_time: float

from fastapi import Request
import msgpack

@router.post("/predict")
async def predict_deprecated(request: Request):
    """
    Deprecated: Use POST /models/{name}/predict
    """
    raise HTTPException(status_code=400, detail="Use /models/{name}/predict")

@router.post("/models/{model_name}/predict", response_model=PredictResponse)
async def predict_model(model_name: str, request: PredictRequest):
    """
    Dynamic endpoint for registered models.
    """
    from smartbatch.registry import get_model
    
    handler = get_model(model_name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Call the registered handler (which should be decorated with @batch)
        result = await handler(request.data)
        
        duration = time.time() - start_time
        get_metrics().record_request(duration)

        return PredictResponse(
            result=result,
            request_id=request_id,
            processing_time=duration
        )
        
    except Exception as e:
        logger.error(f"Inference failed for {request_id} on model {model_name}: {e}")
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(e, OverloadedError):
             status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif "Server is shutting down" in str(e):
             status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=str(e))
