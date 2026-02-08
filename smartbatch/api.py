import uuid
import time
import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, status, Query, Request
from pydantic import BaseModel, ValidationError
from starlette.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST
import msgpack
from smartbatch.metrics import get_metrics
from smartbatch.exceptions import OverloadedError

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Configuration ---
# API configuration can go here (e.g. auth middleware)

# --- API Endpoints ---

@router.get("/admin/models")
def list_models():
    """
    List all registered models and their versions.
    """
    from smartbatch.registry import get_all_models
    return get_all_models()

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

async def _extract_request_data(request: Request) -> Any:
    """
    Supports JSON payloads {"data": ...} and MsgPack payloads:
    - {"data": ...}
    - raw value (e.g. list) for high-performance clients.
    """
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    if content_type == "application/msgpack":
        raw_body = await request.body()
        if not raw_body:
            raise HTTPException(status_code=400, detail="Empty MsgPack body")
        try:
            payload = msgpack.unpackb(raw_body, raw=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid MsgPack payload: {exc}") from exc

        if isinstance(payload, dict):
            try:
                return PredictRequest.model_validate(payload).data
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return payload

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

    try:
        return PredictRequest.model_validate(payload).data
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

@router.post("/predict")
async def predict_deprecated(request: Request):
    """
    Deprecated: Use POST /models/{name}/predict
    """
    raise HTTPException(status_code=400, detail="Use /models/{name}/predict")

@router.post("/models/{model_name}/predict", response_model=PredictResponse)
async def predict_model(model_name: str, request: Request, version: Optional[str] = Query(default=None)):
    """
    Dynamic endpoint for registered models.
    Optional 'version' query parameter to target specific version.
    """
    from smartbatch.registry import get_model
    
    handler = get_model(model_name, version=version)
    if not handler:
        detail = f"Model '{model_name}'"
        if version:
            detail += f" version '{version}'"
        detail += " not found"
        raise HTTPException(status_code=404, detail=detail)
        
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Call the registered handler (which should be decorated with @batch)
        input_data = await _extract_request_data(request)
        result = await handler(input_data)
        
        duration = time.time() - start_time
        get_metrics().record_request(duration)

        return PredictResponse(
            result=result,
            request_id=request_id,
            processing_time=duration
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed for {request_id} on model {model_name}: {e}")
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(e, OverloadedError):
             status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif "Server is shutting down" in str(e):
             status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=str(e))
