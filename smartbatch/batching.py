import asyncio
from dataclasses import dataclass
from typing import List, Union
import time

@dataclass
class InferenceRequest:
    request_id: str
    payload: Union[List[float], "torch.Tensor"]
    future: asyncio.Future
    enqueue_time: float = 0.0

    def __post_init__(self):
        self.enqueue_time = time.time()

# Global queue for inference requests
# We'll initialize this with a specific size in the app startup
# but declaring it here for type hinting/global access pattern
request_queue: asyncio.Queue = None

def get_request_queue() -> asyncio.Queue:
    global request_queue
    if request_queue is None:
        raise RuntimeError("Request queue not initialized")
    return request_queue

def init_request_queue(maxsize: int = 100):
    global request_queue
    request_queue = asyncio.Queue(maxsize=maxsize)
