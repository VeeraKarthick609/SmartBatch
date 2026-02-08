# Test Suite

## 1) Syntax + Expected Behavior Tests

Run:

```bash
./venv/bin/pytest -q tests/test_syntax_and_behavior.py
```

Covers:

- Decorator behavior under concurrency
- Worker restart after stop
- Schema validation path
- Versioned model registry behavior
- JSON and MsgPack request handling
- Invalid worker count rejection

## 2) Baseline vs SmartBatch ResNet Benchmark

Run:

```bash
./venv/bin/python tests/resnet_baseline_vs_smartbatch.py --requests 64 --concurrency 16 --workers 2
```

Outputs (default `results/resnet_baseline_vs_smartbatch/`):

- `latency_comparison.png`
- `throughput_comparison.png`
- `p50_comparison.png`
- `p95_comparison.png`
- `p99_comparison.png`
- `summary.json`
- `samples.csv`

Notes:

- Uses `torchvision.models.resnet18(weights=None)` (no checkpoint download).
- For quick smoke runs on slower machines, lower image size:

```bash
./venv/bin/python tests/resnet_baseline_vs_smartbatch.py --requests 8 --concurrency 4 --image-size 64
```

## 3) Cumulative Stress Test (1k -> 100k)

Run:

```bash
./venv/bin/python tests/resnet_stress_cumulative.py \
  --checkpoints 1000,5000,10000,25000,50000,75000,100000 \
  --concurrency 256 \
  --workers 4
```

Outputs (default `results/resnet_stress_cumulative/`):

- `cumulative_latency_mean.png`
- `cumulative_throughput.png`
- `cumulative_p50.png`
- `cumulative_p95.png`
- `cumulative_p99.png`
- `cumulative_metrics.csv`
- `summary.json`

X-axis in all graphs is cumulative completed requests.
