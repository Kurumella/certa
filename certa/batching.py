"""Cross-request micro-batching for the gateway.

Certa's decode is prefill-bound with ~zero generation, so throughput comes from packing many
concurrent requests into one forward pass. This coalesces the option-scoring of requests that
arrive within a few milliseconds of each other into a single ``engine.distributions`` call,
then routes the per-request slices back. It is transparent: each caller still gets exactly the
answer it would have gotten alone.
"""

from __future__ import annotations

import asyncio
from typing import Mapping

from .contract import DecisionResponse, Question, TokenUsage, assemble_answer, compile_question


class MicroBatcher:
    """Batches concurrent decision requests at the option-scoring level.

    ``max_delay`` is how long the worker waits gathering more requests before flushing;
    ``max_batch`` caps the number of decode plans per forward pass.
    """

    def __init__(self, engine, *, name: str | None = None, max_delay: float = 0.008, max_batch: int = 64) -> None:
        self._engine = engine
        self.name = name or getattr(engine, "name", "certa")
        self._max_delay = max_delay
        self._max_batch = max_batch
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        self._ensure_worker()
        qids = list(questions)
        plans = [compile_question(state, questions[qid]) for qid in qids]  # may raise CompileError → 422
        distributions = await self._submit(plans)
        answers = {qid: assemble_answer(plan, dist) for qid, plan, dist in zip(qids, plans, distributions)}
        return DecisionResponse(model=self.name, answers=answers, usage=TokenUsage(output_tokens=len(plans)))

    async def _submit(self, plans) -> list[list[float]]:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((plans, future))
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            pending = [await self._queue.get()]  # block until at least one request
            total = len(pending[0][0])
            deadline = loop.time() + self._max_delay
            # Gather more requests until the batch is full or the short window elapses.
            while total < self._max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                pending.append(item)
                total += len(item[0])

            flat = [plan for plans, _ in pending for plan in plans]
            try:
                # Run the (synchronous) GPU work off the event loop.
                distributions, _ = await loop.run_in_executor(None, self._engine.distributions, flat)
            except Exception as exc:  # propagate to every waiter in the batch
                for _, future in pending:
                    if not future.done():
                        future.set_exception(exc)
                continue

            offset = 0
            for plans, future in pending:
                if not future.done():
                    future.set_result(distributions[offset : offset + len(plans)])
                offset += len(plans)
