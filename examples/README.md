# SmartBatch Examples

Three self-contained examples, ordered by complexity. Start with the quickstart.

---

## Prerequisites

```bash
pip install smartbatch
```

For the ResNet example only:
```bash
pip install -r examples/requirements.txt
```

---

## 1. Quickstart (no GPU required)

The fastest way to see SmartBatch working. No ML dependencies needed.

Registers three models:
- **square** — numeric batch inference
- **word-count** — text processing with Pydantic input validation
- **storyteller** — streaming text generation (LLM-style, word by word)

**Terminal 1 — start the server:**
```bash
python examples/quickstart_server.py
```

**Terminal 2 — run all demos:**
```bash
python examples/quickstart_client.py
```

What you'll see:
- 16 concurrent `square` requests batched into a single call
- 8 concurrent `word-count` requests with schema validation
- 4 concurrent streaming responses via SSE, tokens arriving word by word
- `/metrics` output showing request counts and latency percentiles

---

## 2. Dynamic Registration

Shows how to register and deregister a model **at runtime** via the admin API,
without restarting the server.

**Terminal 1 — start any SmartBatch server (from repo root):**
```bash
python examples/quickstart_server.py
```

**Terminal 2:**
```bash
python examples/dynamic_registration.py
```

What it demonstrates:
- `POST /admin/models/{name}` — load a function from any importable module
- `DELETE /admin/models/{name}/{version}` — remove it
- Inference on the dynamically registered model
- Confirming the model returns 404 after removal

> **Note**: Run the server from the repo root so `examples/` is on the
> Python path and `examples.dynamic_model` can be imported.

---

## 3. ResNet (GPU optional, production-realistic)

A production-realistic example using ResNet18. Runs on CPU if no GPU is available.

**Terminal 1:**
```bash
python examples/resnet_server.py
```

**Terminal 2:**
```bash
# 32 requests, 8 concurrent, JSON format
python examples/resnet_client.py --requests 32 --concurrency 8 --format json

# MsgPack (lower serialization overhead)
python examples/resnet_client.py --requests 32 --concurrency 8 --format msgpack

# Pin to a specific version
python examples/resnet_client.py --version v1
```

What it demonstrates:
- Multi-worker batching across GPUs (`workers=2`, `worker_id` injection)
- Versioned models: `v1` returns top-1 class, `v2` returns top-5 with scores
- JSON and MsgPack transport
- Adaptive batching with `target_latency`

Environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SB_EXAMPLE_WORKERS` | `2` | Number of batch workers |
| `SB_EXAMPLE_HOST` | `0.0.0.0` | Server bind host |
| `SB_EXAMPLE_PORT` | `8000` | Server bind port |

---

## Streaming: how it works

`POST /models/{name}/stream` returns a [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) stream.

Each event carries one token:
```
data: {"token": "Once ", "request_id": "abc123"}
data: {"token": "upon ", "request_id": "abc123"}
data: {"token": "a ",    "request_id": "abc123"}
...
data: [DONE]
```

On error:
```
data: {"error": "something went wrong"}
```

**Reading the stream in Python:**
```python
import json, requests

with requests.post(
    "http://localhost:8000/models/storyteller/stream",
    json={"data": "dragons"},
    stream=True,
) as resp:
    for line in resp.iter_lines():
        line = line.decode()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        print(json.loads(payload)["token"], end="", flush=True)
```

**Writing a streaming model:**
```python
from smartbatch import streaming_batch, register
from typing import List, Optional

@register(name="my-llm", version="v1")
@streaming_batch(max_batch_size=8, max_wait_time=0.05)
async def generate(batch: List[str]):
    # batch = list of prompts from concurrent requests
    # yield List[Optional[str]]: one token per sequence, None when that sequence is done
    streams = [llm.astream(prompt) for prompt in batch]
    active = [True] * len(batch)

    while any(active):
        tokens = []
        for i, stream in enumerate(streams):
            if active[i]:
                try:
                    tokens.append(await anext(stream))
                except StopAsyncIteration:
                    tokens.append(None)
                    active[i] = False
            else:
                tokens.append(None)
        yield tokens
```

---

## All available endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | JSON: request counts, latency p50/p95/p99, batch stats |
| `GET` | `/admin/models` | List registered models and versions |
| `POST` | `/admin/models/{name}` | Dynamically register a model |
| `DELETE` | `/admin/models/{name}/{version}` | Deregister a model version |
| `POST` | `/models/{name}/predict` | Standard inference (add `?version=` to pin) |
| `POST` | `/models/{name}/stream` | Streaming inference via SSE (add `?version=` to pin) |
