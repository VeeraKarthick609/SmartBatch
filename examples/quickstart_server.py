"""
Quickstart server — no GPU or ML dependencies required.

Demonstrates:
- @batch decorator
- @register for named model endpoints
- Input schema validation with Pydantic

Run:
    pip install smartbatch
    python examples/quickstart_server.py

Then in another terminal:
    python examples/quickstart_client.py
"""

import sys
from pathlib import Path
from typing import List

import uvicorn
from pydantic import BaseModel

# Use local workspace if running from the repo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartbatch import batch, register
from smartbatch.main import app


# --- Model 1: simple arithmetic ---
# No schema — accepts any numeric input

@register(name="square", version="v1")
@batch(max_batch_size=16, max_wait_time=0.01)
def square(batch: List[float]) -> List[float]:
    return [x ** 2 for x in batch]


# --- Model 2: text processing with input validation ---

class SentenceInput(BaseModel):
    text: str

@register(name="word-count", version="v1")
@batch(max_batch_size=32, max_wait_time=0.01, input_schema=SentenceInput)
def word_count(batch: List[SentenceInput]) -> List[int]:
    return [len(item.text.split()) for item in batch]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
