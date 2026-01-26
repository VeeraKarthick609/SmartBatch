# SmartBatch: High-Throughput Async Inference Middleware

**SmartBatch** is a production-grade inference serving system designed to maximize GPU utilization and throughput for PyTorch models. It implements **Dynamic Batching** to group incoming requests on-the-fly, significantly reducing overhead compared to naive request-per-inference processing.

## 🚀 Key Features

*   **Dynamic Batching**: Automatically groups requests into batches (up to `MAX_BATCH_SIZE`) or processes them after a timeout (`MAX_WAIT_TIME`), striking the perfect balance between throughput and latency.
*   **Asynchronous API**: Built on `FastAPI` and `asyncio` to handle thousands of concurrent connections efficiently.
*   **Production Robustness**: Includes graceful shutdown, proper error handling, and thread-safe metrics.
*   **Real-World Load Testing**: Benchmarking suite included to simulate high-concurrency traffic with realistic payloads.
*   **Observability**: `/metrics` endpoint for real-time monitoring of latency, batch sizes, and throughput.

## 📊 Benchmark Results (Stress Test)

We conducted a rigorous 1-hour stress test comparing **SmartBatch** against a **Baseline** (no batching) implementation.

**Hardware**: Single Node (Simulated Production Environment)
**Load**: 200-1000 Concurrent Users
**Payload**: Real-world images (ResNet18 inputs)

| Metric | Baseline (Sequential) | SmartBatch (Batched) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput (RPS)** | ~0.67 req/s | **~2.22 req/s** | **3.3x Higher** |
| **Median Latency (p50)** | > 1000s (Collapsed) | **~13s** (Stable) | **~99% Reduction** |
| **Tail Latency (p95)** | Unstable / Timeouts | Controlled by Batching | **Stabilized** |

### Performance Visualization

**1. Throughput Stability**
SmartBatch maintains significantly higher requests-per-second (RPS) under heavy load compared to the baseline, which collapses due to contention.
![Throughput Graph](production_throughput.png)

**2. Latency Control**
While the Baseline's latency explodes exponentially as the queue fills, SmartBatch keeps latency stable by processing efficiently in batches.
![Latency Graph](production_latency_p50.png)

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/SmartBatch.git
    cd SmartBatch
    ```

2.  **Create a virtual environment**:
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install .
    ```

## 🏃 Usage

### 1. Start the Server
Run the production server with customizable batch settings:
```bash
# Example: Batch size 32, Max wait 10ms
export MAX_BATCH_SIZE=32
export MAX_WAIT_TIME=0.01
export MODEL_PATH="resnet18" # Uses torchvision default

uvicorn smartbatch.main:app --host 0.0.0.0 --port 8000
```

### 2. Send Requests
You can send requests containing image data (nested lists or flattened arrays):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [[0.1, 0.2, ...]]}'
```

### 3. Check Metrics
Monitor system health and performance in real-time:
```bash
curl http://localhost:8000/metrics
```

## 🧪 Benchmarking

Reproduce the performance results yourself using the included benchmark suite.

**1. Run the Comparison Benchark**
This script runs both Baseline and SmartBatch scenarios sequentially and generates plots.
```bash
# usage: --users [NUM] --duration [TIME]
venv/bin/python scripts/production_benchmark.py --users 200 --duration 1h
```

**2. View Results**
The script will generate these files in your root directory:
*   `production_throughput.png`
*   `production_latency_p50.png`
*   `production_latency_p95.png`

## 📂 Project Structure

```
SmartBatch/
├── smartbatch/          # Core Package
│   ├── main.py          # Entry point & App lifecycle
│   ├── api.py           # FastAPI endpoints
│   ├── batching.py      # Batch logic & Queue management
│   ├── model.py         # PyTorch Model Wrapper
│   └── metrics.py       # Thread-safe metrics collection
├── scripts/             # Utilities
│   ├── production_benchmark.py # A/B Stress Test Script
│   └── study_params.py  # Hyperparameter optimization
├── tests/               # Testing
│   ├── locustfile.py    # Load generator
│   └── data/            # Test images (for realistic load)
└── README.md            # Documentation
```
