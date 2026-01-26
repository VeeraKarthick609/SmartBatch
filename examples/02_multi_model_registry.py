import asyncio
from smartbatch import batch, register
from smartbatch import registry
from typing import List

# 1. Register multiple "models"
@register(name="doubler")
@batch(max_batch_size=5)
async def double_model(batch: List[int]):
    print(f"[Doubler] Processing batch: {batch}")
    return [x * 2 for x in batch]

@register(name="squarer")
@batch(max_batch_size=2)
async def square_model(batch: List[int]):
    print(f"[Squarer] Processing batch: {batch}")
    return [x * x for x in batch]

async def main():
    print("--- Multi-Model Registry Demo ---")
    
    # Simulate API router looking up models dynamically
    handler_doubler = registry.get_model("doubler")
    handler_squarer = registry.get_model("squarer")
    
    # Simulate requests
    t1 = handler_doubler(10)
    t2 = handler_squarer(5)
    t3 = handler_doubler(20)
    
    results = await asyncio.gather(t1, t2, t3)
    print(f"Results: {results}") # [20, 25, 40]

if __name__ == "__main__":
    asyncio.run(main())
