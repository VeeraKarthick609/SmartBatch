import asyncio
from smartbatch import batch
from typing import List

# Mock "Models" on different devices
class MockModel:
    def __init__(self, device):
        self.device = device
    
    def __call__(self, batch):
        return [f"processed_on_{self.device}" for _ in batch]

# Load models (Simulating loading on cuda:0, cuda:1, etc.)
models = {
    0: MockModel("gpu:0"),
    1: MockModel("gpu:1"),
    2: MockModel("gpu:2")
}

# 3 Workers for 3 Devices
@batch(max_batch_size=5, workers=3)
async def multi_gpu_infer(batch: List[int], worker_id=0):
    # SmartBatch injects the worker_id (0, 1, or 2)
    print(f"Worker {worker_id} (Device {models[worker_id].device}) got batch: {batch}")
    
    # Use the correct model instance
    model = models[worker_id]
    
    await asyncio.sleep(0.5) 
    return model(batch)

async def main():
    print("--- Multi-GPU Worker Pool Demo ---")
    # Send requests. They should be distributed across workers.
    # Note: Distribution depends on who picks from the queue first.
    tasks = [multi_gpu_infer(i) for i in range(15)]
    results = await asyncio.gather(*tasks)
    print("Sample Results:", results[:3])

if __name__ == "__main__":
    asyncio.run(main())
