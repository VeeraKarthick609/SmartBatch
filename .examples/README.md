# SmartBatch ResNet Examples

This folder contains end-to-end examples that exercise most SmartBatch features using a ResNet model.

## What these examples cover

- Dynamic batching (`max_batch_size`, `max_wait_time`)
- Adaptive batching (`target_latency`)
- Input validation with Pydantic (`input_schema`)
- Multi-worker execution with `worker_id`
- Versioned model registry (`v1`, `v2`)
- JSON and MsgPack request formats
- Metrics/admin endpoints exposed by `smartbatch.main`

## Prerequisites

Install project deps plus torchvision:

```bash
pip install -e .
pip install torchvision
```

## 1. Start the ResNet service

```bash
python .examples/resnet_server.py
```

Server defaults:

- Host: `0.0.0.0`
- Port: `8000`
- Workers inside SmartBatch batcher: `2` (configurable)

Environment variables:

- `SB_EXAMPLE_WORKERS` (default `2`)
- `SB_EXAMPLE_HOST` (default `0.0.0.0`)
- `SB_EXAMPLE_PORT` (default `8000`)

## 2. Run the concurrent client/load test

JSON mode:

```bash
python .examples/resnet_client.py --requests 32 --concurrency 8 --version v2 --format json
```

MsgPack mode:

```bash
python .examples/resnet_client.py --requests 32 --concurrency 8 --version v2 --format msgpack
```

## Useful endpoints

- `GET /health`
- `GET /metrics`
- `GET /admin/models`
- `POST /models/resnet18/predict?version=v1`
- `POST /models/resnet18/predict?version=v2`
