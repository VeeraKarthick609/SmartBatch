import importlib
import json
import uuid
import time
import logging
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, status, Query, Request
from pydantic import BaseModel, ValidationError
from starlette.responses import StreamingResponse
import msgpack
from smartbatch.metrics import get_metrics
from smartbatch.exceptions import OverloadedError
from smartbatch.decorator import _STREAM_DONE, _STREAM_ERROR

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Configuration ---
# API configuration can go here (e.g. auth middleware)

# --- API Endpoints ---

@router.get("/admin/models")
def list_models():
    from smartbatch.registry import get_all_models
    return get_all_models()


class RegisterRequest(BaseModel):
    module: str
    function: str
    version: str = "v1"
    max_batch_size: int = 32
    max_wait_time: float = 0.01
    max_queue_size: int = 128
    workers: int = 1
    target_latency: Optional[float] = None


@router.post("/admin/models/{name}", status_code=status.HTTP_201_CREATED)
def register_model(name: str, req: RegisterRequest):
    """
    Dynamically register a batch function from an importable module.

    The function must accept List[Any] and return List[Any] of the same length.
    Example body:
        {"module": "myapp.models", "function": "infer", "version": "v2"}
    """
    from smartbatch.registry import register
    from smartbatch.decorator import batch

    try:
        mod = importlib.import_module(req.module)
    except ModuleNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"Cannot import module '{req.module}': {e}")

    func = getattr(mod, req.function, None)
    if func is None:
        raise HTTPException(status_code=400, detail=f"Function '{req.function}' not found in '{req.module}'")
    if not callable(func):
        raise HTTPException(status_code=400, detail=f"'{req.function}' is not callable")

    batched = batch(
        max_batch_size=req.max_batch_size,
        max_wait_time=req.max_wait_time,
        max_queue_size=req.max_queue_size,
        workers=req.workers,
        target_latency=req.target_latency,
    )(func)

    register(name=name, version=req.version)(batched)

    logger.info(f"Dynamically registered '{name}' version '{req.version}' from {req.module}.{req.function}")
    return {"name": name, "version": req.version, "module": req.module, "function": req.function}


@router.delete("/admin/models/{name}/{version}", status_code=status.HTTP_200_OK)
def deregister_model(name: str, version: str):
    from smartbatch.registry import deregister
    removed = deregister(name, version)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Model '{name}' version '{version}' not found")
    logger.info(f"Deregistered '{name}' version '{version}'")
    return {"name": name, "version": version, "status": "removed"}

@router.get("/metrics")
def metrics_endpoint():
    return get_metrics().get_stats()

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


@router.post("/models/{model_name}/stream")
async def stream_model(model_name: str, request: Request, version: Optional[str] = Query(default=None)):
    """
    Streaming inference endpoint. Returns a Server-Sent Events (SSE) stream.

    The registered handler must be decorated with @streaming_batch.

    Each SSE event carries one token:
        data: {"token": "...", "request_id": "..."}

    The stream ends with:
        data: [DONE]

    On error:
        data: {"error": "..."}
    """
    from smartbatch.registry import get_model

    handler = get_model(model_name, version=version)
    if not handler:
        detail = f"Model '{model_name}'"
        if version:
            detail += f" version '{version}'"
        detail += " not found"
        raise HTTPException(status_code=404, detail=detail)

    if not getattr(handler, "is_streaming", False):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' does not support streaming. "
                   "Use POST /models/{name}/predict instead.",
        )

    request_id = str(uuid.uuid4())

    try:
        input_data = await _extract_request_data(request)
        token_queue = await handler(input_data)
    except OverloadedError as e:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def event_stream():
        start_time = time.time()
        try:
            while True:
                item = await token_queue.get()

                if item is _STREAM_DONE:
                    yield "data: [DONE]\n\n"
                    break

                if isinstance(item, tuple) and len(item) == 2 and item[0] is _STREAM_ERROR:
                    _, exc = item
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                    break

                yield f"data: {json.dumps({'token': item, 'request_id': request_id})}\n\n"

        finally:
            get_metrics().record_request(time.time() - start_time)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
