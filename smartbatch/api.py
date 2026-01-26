import asyncio
import uuid
import time
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Any
import logging

from smartbatch.batching import InferenceRequest, get_request_queue
from smartbatch.metrics import get_metrics

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/metrics")
def metrics_endpoint():
    return get_metrics().get_stats()


class PredictRequest(BaseModel):
    data: Any # Allow nested lists for tensors

class PredictResponse(BaseModel):
    result: Any
    request_id: str
    processing_time: float

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Enqueue inference request and await result.
    """
    from smartbatch.batching import get_shutdown_event
    if get_shutdown_event().is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down"
        )

    request_id = str(uuid.uuid4())
    queue = get_request_queue()
    
    # Create valid inference request
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    inference_req = InferenceRequest(
        request_id=request_id,
        payload=request.data,
        future=future
    )
    
    try:
        # Enqueue with timeout/backpressure check could happen here
        # For now, standard put (will block if full, or we can use put_nowait)
        if queue.full():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Server is busy, try again later."
            )
        
        queue.put_nowait(inference_req)
        
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is busy"
        )
        
    try:
        # Await the result
        result = await future
        
        # Calculate processing time
        duration = time.time() - inference_req.enqueue_time
        
        # Record metric
        get_metrics().record_request(duration)

        return PredictResponse(
            result=result,
            request_id=request_id,
            processing_time=duration
        )
        
    except Exception as e:
        logger.error(f"Inference failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
