import asyncio
import json
from typing import List

import msgpack
import pytest
from pydantic import BaseModel
from starlette.requests import Request

from smartbatch.api import predict_model
from smartbatch.decorator import batch
from smartbatch.registry import get_model, register, reset_registry


@pytest.fixture(autouse=True)
def reset_model_registry():
    reset_registry()
    yield
    reset_registry()


def _build_request(body: bytes, content_type: str) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/models/model-a/predict",
        "raw_path": b"/models/model-a/predict",
        "query_string": b"",
        "headers": [(b"content-type", content_type.encode()), (b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    return Request(scope, receive)


def test_register_supports_versions_and_latest_lookup():
    @register(name="model-a", version="v9")
    async def model_v9(batch):
        return batch

    @register(name="model-a", version="v10")
    async def model_v10(batch):
        return batch

    assert get_model("model-a") is model_v10
    assert get_model("model-a", version="v9") is model_v9


def test_batch_handles_concurrency_and_restart_after_stop():
    @batch(max_batch_size=4, max_wait_time=0.01, workers=1)
    async def add_one(batch):
        return [item + 1 for item in batch]

    async def scenario():
        try:
            first = await asyncio.gather(*(add_one(i) for i in range(16)))
            assert first == [i + 1 for i in range(16)]

            await add_one.batcher.stop()

            second = await asyncio.wait_for(add_one(41), timeout=2.0)
            assert second == 42
        finally:
            await add_one.batcher.stop()

    asyncio.run(scenario())


def test_batch_input_schema_validation():
    class Payload(BaseModel):
        value: int

    @batch(max_batch_size=2, max_wait_time=0.01, input_schema=Payload)
    async def multiply(batch: List[Payload]):
        return [item.value * 2 for item in batch]

    async def scenario():
        try:
            ok = await multiply({"value": 12})
            assert ok == 24

            with pytest.raises(ValueError):
                await multiply({"value": "bad"})

            with pytest.raises(TypeError):
                await multiply(5)
        finally:
            await multiply.batcher.stop()

    asyncio.run(scenario())


def test_predict_model_accepts_json_and_msgpack():
    @register(name="model-a", version="v1")
    @batch(max_batch_size=2, max_wait_time=0.01)
    async def echo(batch):
        return batch

    async def scenario():
        try:
            json_request = _build_request(
                body=json.dumps({"data": [1, 2, 3]}).encode(),
                content_type="application/json",
            )
            json_response = await predict_model("model-a", json_request, version="v1")
            assert json_response.result == [1, 2, 3]

            msgpack_request = _build_request(
                body=msgpack.packb({"data": {"x": 7}}, use_bin_type=True),
                content_type="application/msgpack",
            )
            msgpack_response = await predict_model("model-a", msgpack_request, version="v1")
            assert msgpack_response.result == {"x": 7}
        finally:
            await echo.batcher.stop()

    asyncio.run(scenario())


def test_batch_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="workers must be >= 1"):
        @batch(workers=0)
        async def invalid(batch):
            return batch
