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

The fastest way to see SmartBatch working. Registers two models — a numeric
function and a text processor — with no ML dependencies.

**Terminal 1 — start the server:**
```bash
python examples/quickstart_server.py
```

**Terminal 2 — send concurrent requests:**
```bash
python examples/quickstart_client.py
```

What it demonstrates:
- `@batch` decorator basics
- `@register` for named model endpoints
- Pydantic input schema validation
- Concurrent requests being grouped into batches
- `/metrics` output

---

## 2. Dynamic Registration

Shows how to register and deregister a model **at runtime** via the admin API,
without restarting the server.

**Terminal 1 — start any SmartBatch server (from repo root):**
```bash
python examples/quickstart_server.py
```

**Terminal 2 — run the registration demo:**
```bash
python examples/dynamic_registration.py
```

What it demonstrates:
- `POST /admin/models/{name}` to load a function from an importable module
- `DELETE /admin/models/{name}/{version}` to remove it
- Inference on the dynamically registered model
- Verifying the model is gone after deregistration

The model being loaded lives in `examples/dynamic_model.py`. Swap in your own
module and function name to register real models the same way.

> **Note**: Run the server from the repo root so that `examples/` is on the
> Python path and `examples.dynamic_model` can be imported.

---

## 3. ResNet (GPU optional)

A production-realistic example using ResNet18 from torchvision. Runs on CPU
if no GPU is available.

**Terminal 1 — start the ResNet server:**
```bash
python examples/resnet_server.py
```

**Terminal 2 — run the load client:**
```bash
# JSON format, 32 requests, 8 concurrent
python examples/resnet_client.py --requests 32 --concurrency 8 --format json

# MsgPack format (lower overhead)
python examples/resnet_client.py --requests 32 --concurrency 8 --format msgpack

# Target a specific version
python examples/resnet_client.py --version v1
```

What it demonstrates:
- Multi-worker batching across GPUs (`workers=2`, `worker_id` injection)
- Versioned models (`v1` returns top-1 class, `v2` returns top-5 with scores)
- JSON and MsgPack transport
- Adaptive batching with `target_latency`
- Large queue handling under concurrent load

Environment variables:
| Variable | Default | Description |
| :--- | :--- | :--- |
| `SB_EXAMPLE_WORKERS` | `2` | Number of batch workers |
| `SB_EXAMPLE_HOST` | `0.0.0.0` | Server bind host |
| `SB_EXAMPLE_PORT` | `8000` | Server bind port |

---

## Useful endpoints (all examples)

| Endpoint | Description |
| :--- | :--- |
| `GET /health` | Health check |
| `GET /metrics` | Request counts, latency p50/p95/p99, batch stats |
| `GET /admin/models` | List registered models and versions |
| `POST /admin/models/{name}` | Dynamically register a model |
| `DELETE /admin/models/{name}/{version}` | Deregister a model version |
| `POST /models/{name}/predict` | Run inference (add `?version=` to pin a version) |
