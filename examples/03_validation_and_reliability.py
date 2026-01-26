import asyncio
import time
from typing import List
from pydantic import BaseModel
from smartbatch import batch

# 1. Define Input Schema
class DataItem(BaseModel):
    value: int
    tag: str

# 2. Batched Function with Validation and Low Capacity (for testing backpressure)
@batch(
    max_batch_size=2,          # Very small batches
    max_wait_time=0.5,         # Slow processing
    input_schema=DataItem,     # Enforce Pydantic
    max_queue_size=4           # Tiny queue to force backpressure
)
async def flexible_worker(batch: List[DataItem]):
    print(f"Worker: Processing {[x.value for x in batch]}")
    await asyncio.sleep(0.5) # Slow processing
    return [x.value for x in batch]

async def success_case():
    print("\n--- Valid Request ---")
    try:
        # Valid input (dict or object)
        res = await flexible_worker({"value": 100, "tag": "test"})
        print(f"Success: {res}")
    except Exception as e:
        print(f"Error: {e}")

async def validation_failure():
    print("\n--- Invalid Request (Schema) ---")
    try:
        # Missing 'tag' -> Schema Failure
        await flexible_worker({"value": 999}) 
    except ValueError as e:
        print(f"Caught Expected Validation Error: {e}")

async def backpressure_test():
    print("\n--- Backpressure Test (Burst) ---")
    # Queue size is 4. We send 10 requests rapidly.
    # Worker takes 0.5s per batch of 2.
    # Expect: Some might fail or retry logic kicks in.
    
    tasks = []
    for i in range(10):
        tasks.append(flexible_worker(DataItem(value=i, tag="burst")))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            print(f"Req {i}: Failed ({res})")
        else:
            print(f"Req {i}: Success")

async def main():
    await success_case()
    await validation_failure()
    await backpressure_test()

if __name__ == "__main__":
    asyncio.run(main())
