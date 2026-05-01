"""
Quickstart server — no GPU or ML dependencies required.

Demonstrates:
- @batch for standard request/response inference
- @streaming_batch for token-by-token streaming (LLM-style)
- @register for named model endpoints
- Input schema validation with Pydantic

Run:
    pip install smartbatch
    python examples/quickstart_server.py

Then in another terminal:
    python examples/quickstart_client.py
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import uvicorn
from pydantic import BaseModel

# Use local workspace if running from the repo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smartbatch import batch, register, streaming_batch
from smartbatch.main import app


# ---------------------------------------------------------------------------
# Model 1: simple arithmetic (standard batch)
# ---------------------------------------------------------------------------

@register(name="square", version="v1")
@batch(max_batch_size=16, max_wait_time=0.01)
def square(batch: List[float]) -> List[float]:
    return [x ** 2 for x in batch]


# ---------------------------------------------------------------------------
# Model 2: text processing with input validation (standard batch)
# ---------------------------------------------------------------------------

class SentenceInput(BaseModel):
    text: str

@register(name="word-count", version="v1")
@batch(max_batch_size=32, max_wait_time=0.01, input_schema=SentenceInput)
def word_count(batch: List[SentenceInput]) -> List[int]:
    return [len(item.text.split()) for item in batch]


# ---------------------------------------------------------------------------
# Model 3: streaming text generator (LLM-style)
#
# Simulates token-by-token generation without any ML dependency.
# In production replace this with your LLM's async streaming call.
#
# The function must be an async generator that yields List[Optional[str]]:
#   - One entry per sequence in the batch
#   - None means that sequence has finished generating
# ---------------------------------------------------------------------------

@register(name="storyteller", version="v1")
@streaming_batch(max_batch_size=8, max_wait_time=0.05)
async def storyteller(batch: List[str]):
    """
    Given a topic, stream a short made-up story word by word.
    Each yield is one word for every active sequence in the batch.
    """
    stories = [
        f"Once upon a time, a brave explorer discovered {topic}. "
        f"It changed everything about how we see {topic} forever."
        for topic in batch
    ]
    word_lists = [story.split() for story in stories]
    max_len = max(len(words) for words in word_lists)
    active = [True] * len(batch)

    for i in range(max_len):
        tokens = []
        for j, words in enumerate(word_lists):
            if not active[j]:
                tokens.append(None)
            elif i < len(words):
                tokens.append(words[i] + (" " if i < len(words) - 1 else ""))
            else:
                active[j] = False
                tokens.append(None)
        yield tokens
        await asyncio.sleep(0.02)   # simulate generation latency


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
