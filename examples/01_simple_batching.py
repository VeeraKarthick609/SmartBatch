import asyncio
from smartbatch import batch
from typing import List

# 1. Define your backend processing function (batches of inputs)
# This simulates a model inference that takes multiple inputs at once.
@batch(max_batch_size=4, max_wait_time=0.1)
async def process_batch(batch: List[int]) -> List[int]:
    print(f"Server: Processing batch of size {len(batch)}: {batch}")
    await asyncio.sleep(0.1) # Simulate inference time
    return [x * 2 for x in batch]

# 2. Simulate concurrent clients
async def client_request(i: int):
    # Each client just calls the function with a SINGLE item
    # They don't know batching is happening!
    print(f"Client {i}: Sending request...")
    result = await process_batch(i)
    print(f"Client {i}: Received result: {result}")

async def main():
    print("--- Starting Basic Batching Demo ---")
    # Launch 10 concurrent requests
    await asyncio.gather(*[client_request(i) for i in range(10)])
    print("--- Done ---")

if __name__ == "__main__":
    asyncio.run(main())
