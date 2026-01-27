import asyncio
import random
import time
from smartbatch import batch
from typing import List

# Simulate a model where execution time scales linearly with batch size
# e.g., 100ms per item. 
# If batch size is 5 -> 500ms.
# If batch size is 10 -> 1s.

@batch(
    max_batch_size=16, 
    max_wait_time=0.2, 
    target_latency=0.3, # We want to keep latency under 300ms
    workers=1
)
async def adaptive_model(batch: List[int]):
    # Allow injection of worker_id if passed, but we don't need it here.
    # Note: decorator passes worker_id if function accepts it.
    
    # Linear slowdown
    current_latency = len(batch) * 0.05
    await asyncio.sleep(current_latency)
    
    print(f"[Batch Size {len(batch)}] Latency: {current_latency:.2f}s (Limit: {adaptive_model.batcher.current_batch_size_limit})")
    
    return [x * 2 for x in batch]

async def main():
    print("--- Adaptive Batching Demo ---")
    print("Target Latency: 0.3s. Model speed: 0.05s per item.")
    print("Ideal Batch Size: 0.3 / 0.05 = 6 items.")
    print("Max Batch Size: 16 items.")
    print("sending bursts...")
    
    # Send a burst of 50 requests. 
    # Logic should effectively reduce batch size to around 6.
    
    tasks = []
    for i in range(50):
        tasks.append(asyncio.create_task(adaptive_model(i)))
        # Slight delay to allow accumulation but not instant dump
        if i % 5 == 0:
            await asyncio.sleep(0.01)
            
    await asyncio.gather(*tasks)
    
    print("\nFinal Batch Size Limit:", adaptive_model.batcher.current_batch_size_limit)

if __name__ == "__main__":
    asyncio.run(main())
