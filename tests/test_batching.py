"""Cross-request micro-batching coalesces concurrent requests (no model load)."""

from __future__ import annotations

import asyncio

from certa.batching import MicroBatcher
from certa.contract import PickOne


class RecordingEngine:
    """Stand-in for ModelEngine: returns uniform distributions and records batch sizes."""

    name = "fake"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def distributions(self, plans):
        self.batch_sizes.append(len(plans))
        return [[1.0 / p.width] * p.width for p in plans], 0


async def _fire(n: int):
    engine = RecordingEngine()
    batcher = MicroBatcher(engine, max_delay=0.05, max_batch=256)
    question = {"t": PickOne(type="choice", instructions="x", criteria={"a": "A", "b": "B"})}
    results = await asyncio.gather(*[batcher.decide("state", question) for _ in range(n)])
    return engine.batch_sizes, results


def test_concurrent_requests_are_coalesced():
    sizes, results = asyncio.run(_fire(10))
    assert len(results) == 10
    assert all(r.answers["t"].type == "choice" for r in results)
    assert sum(sizes) == 10  # every plan was scored exactly once
    assert len(sizes) < 10  # ...but in fewer than 10 forward passes (coalesced)


def test_single_request_still_works():
    sizes, results = asyncio.run(_fire(1))
    assert len(results) == 1 and sizes == [1]
