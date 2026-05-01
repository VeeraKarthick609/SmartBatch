import asyncio
import json
from typing import List, Optional

import pytest

from smartbatch.decorator import _STREAM_DONE, _STREAM_ERROR, StreamingBatcher, streaming_batch
from smartbatch.registry import register, reset_registry


@pytest.fixture(autouse=True)
def reset_model_registry():
    reset_registry()
    yield
    reset_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _drain(queue: asyncio.Queue) -> list:
    """Read all tokens from a stream queue until DONE."""
    tokens = []
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=5.0)
        if item is _STREAM_DONE:
            break
        if isinstance(item, tuple) and item[0] is _STREAM_ERROR:
            raise item[1]
        tokens.append(item)
    return tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_streaming_batch_single_sequence():
    @streaming_batch(max_batch_size=4, max_wait_time=0.01)
    async def char_stream(batch: List[str]):
        for word in batch:
            chars = list(word)
            done = [False] * len(batch)
            done[batch.index(word)] = False

        # Yield char by char for each sequence
        max_len = max(len(w) for w in batch)
        for i in range(max_len):
            yield [w[i] if i < len(w) else None for w in batch]

    async def scenario():
        try:
            queue = await char_stream("hello")
            tokens = await _drain(queue)
            assert tokens == list("hello")
        finally:
            await char_stream.batcher.stop()

    asyncio.run(scenario())


def test_streaming_batch_multiple_concurrent_sequences():
    @streaming_batch(max_batch_size=8, max_wait_time=0.02)
    async def word_stream(batch: List[str]):
        max_len = max(len(s.split()) for s in batch)
        split = [s.split() for s in batch]
        for i in range(max_len):
            yield [words[i] if i < len(words) else None for words in split]

    async def scenario():
        try:
            sentences = ["one two three", "a b", "x y z w"]
            queues = await asyncio.gather(*(word_stream(s) for s in sentences))
            results = await asyncio.gather(*(_drain(q) for q in queues))
            assert results[0] == ["one", "two", "three"]
            assert results[1] == ["a", "b"]
            assert results[2] == ["x", "y", "z", "w"]
        finally:
            await word_stream.batcher.stop()

    asyncio.run(scenario())


def test_streaming_batch_rejects_non_async_gen():
    with pytest.raises(TypeError, match="async generator"):
        @streaming_batch()
        def not_a_gen(batch):
            return batch


def test_streaming_batch_rejects_invalid_worker_count():
    with pytest.raises(ValueError, match="workers must be >= 1"):
        @streaming_batch(workers=0)
        async def bad(batch):
            yield []


def test_streaming_batch_error_propagated():
    @streaming_batch(max_batch_size=2, max_wait_time=0.01)
    async def exploding(batch: List[str]):
        yield ["token"] * len(batch)
        raise RuntimeError("boom")

    async def scenario():
        try:
            queue = await exploding("test")
            first = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert first == "token"
            second = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert isinstance(second, tuple) and second[0] is _STREAM_ERROR
            assert "boom" in str(second[1])
        finally:
            await exploding.batcher.stop()

    asyncio.run(scenario())


def test_streaming_batch_wrapper_has_is_streaming_flag():
    @streaming_batch()
    async def gen(batch):
        yield [None] * len(batch)

    assert getattr(gen, "is_streaming", False) is True


def test_streaming_batch_restart_after_stop():
    @streaming_batch(max_batch_size=4, max_wait_time=0.01)
    async def toggling(batch: List[int]):
        yield [x * 2 for x in batch]

    async def scenario():
        try:
            q = await toggling(3)
            tokens = await _drain(q)
            assert tokens == [6]

            await toggling.batcher.stop()

            q2 = await toggling(5)
            tokens2 = await _drain(q2)
            assert tokens2 == [10]
        finally:
            await toggling.batcher.stop()

    asyncio.run(scenario())
