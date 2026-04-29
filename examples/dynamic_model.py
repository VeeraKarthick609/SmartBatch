"""
A standalone batch function that can be dynamically registered via the admin API.

This module must be importable from wherever the SmartBatch server runs.
The function must accept List[Any] and return List[Any] of the same length.
"""

from typing import List


def sentiment_score(batch: List[str]) -> List[dict]:
    """
    Toy sentiment scorer. Replace with a real model in production.
    Returns a score in [-1.0, 1.0] and a label for each input string.
    """
    positive_words = {"good", "great", "excellent", "love", "awesome", "fantastic"}
    negative_words = {"bad", "terrible", "awful", "hate", "horrible", "poor"}

    results = []
    for text in batch:
        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg or 1
        score = round((pos - neg) / total, 3)
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        results.append({"score": score, "label": label})
    return results
