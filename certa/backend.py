"""Deployment backends behind one interface.

A caller always does ``backend.decide(state, questions)``; *where* the model runs is a
configuration choice, not a code fork:

* :class:`LocalBackend`  — model in-process, same machine as the caller (lowest latency).
* :class:`RemoteBackend` — HTTP to a Certa (or Jev-compatible) gateway on another machine.
* :class:`VLLMBackend`   — opt-in adapter for teams already running vLLM (advanced).

:class:`CachedBackend` wraps any of them; :func:`from_env` builds one from environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from typing import Mapping, Protocol

from .contract import DecisionResponse, Question


class DecisionBackend(Protocol):
    """Anything that can answer a decision request. Implementations are interchangeable."""

    name: str

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse: ...


class LocalBackend:
    """Run the model in-process. Best when the caller and the GPU share a machine."""

    def __init__(self, checkpoint: str | None = None, **engine_kwargs) -> None:
        from .contract.constants import BASE_CHECKPOINT
        from .engine import ModelEngine

        self._engine = ModelEngine(checkpoint or BASE_CHECKPOINT, **engine_kwargs)
        self.name = self._engine.name

    @property
    def engine(self):
        """The underlying :class:`~certa.engine.ModelEngine` (used by the micro-batcher)."""
        return self._engine

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        return self._engine.decide(state, questions)


class RemoteBackend:
    """Call a Certa gateway over HTTP. Best when many light clients share one GPU box."""

    def __init__(self, base_url: str | None = None, *, api_key: str | None = None,
                 model: str | None = None, timeout: float = 10.0) -> None:
        from .client import Client

        self._client = Client(base_url, api_key=api_key, model=model, timeout=timeout)
        self.name = model or "certa-remote"

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        return self._client.decide(state, questions)

    def close(self) -> None:
        self._client.close()


class VLLMBackend:
    """Opt-in adapter for scoring option tokens via a vLLM server (advanced, unimplemented).

    For Certa's single-forward-pass decode the batched :class:`LocalBackend` / gateway is
    simpler and preserves the exact ``softmax over declared options`` semantics that make the
    probabilities calibrated. vLLM targets long autoregressive generation, which this workload
    does not do — so use this only if you already run vLLM and want a single serving fabric.

    Recipe to implement against a vLLM OpenAI-compatible server: request the decode prompt with
    ``max_tokens=1`` and ``logprobs`` wide enough to include every option-label token, pull those
    tokens' logprobs, and re-normalize over the declared options (mirroring ``ModelEngine``).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "VLLMBackend is an opt-in stub. Use LocalBackend/RemoteBackend, or implement the "
            "logprobs recipe in this class's docstring against your vLLM server."
        )


class CachedBackend:
    """LRU cache over any backend. Safe because a decision is deterministic (argmax/softmax,
    no sampling): the same (state, questions) always yields the same answer."""

    def __init__(self, inner: DecisionBackend, maxsize: int = 4096) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._cache: "OrderedDict[str, DecisionResponse]" = OrderedDict()
        self.name = inner.name

    @staticmethod
    def _key(state, questions: Mapping[str, Question]) -> str:
        payload = {"state": state, "questions": {q: v.model_dump() for q, v in questions.items()}}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        key = self._key(state, questions)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)  # LRU touch
            return hit
        result = self._inner.decide(state, questions)
        self._cache[key] = result
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict least-recently-used
        return result


def from_env() -> DecisionBackend:
    """Build a backend from the environment.

    ``CERTA_BACKEND`` = ``local`` (default) | ``remote`` | ``vllm``. Local reads
    ``CERTA_CHECKPOINT``; remote reads ``CERTA_BASE_URL`` / ``CERTA_API_KEY`` /
    ``CERTA_DEFAULT_MODEL``. Set ``CERTA_CACHE=1`` to wrap the result in a cache.
    """
    kind = os.environ.get("CERTA_BACKEND", "local").lower()
    if kind == "local":
        backend: DecisionBackend = LocalBackend(os.environ.get("CERTA_CHECKPOINT"))
    elif kind == "remote":
        backend = RemoteBackend(
            os.environ.get("CERTA_BASE_URL"),
            api_key=os.environ.get("CERTA_API_KEY"),
            model=os.environ.get("CERTA_DEFAULT_MODEL"),
        )
    elif kind == "vllm":
        backend = VLLMBackend()
    else:
        raise ValueError(f"unknown CERTA_BACKEND: {kind!r} (expected local|remote|vllm)")

    if os.environ.get("CERTA_CACHE", "").lower() in {"1", "true", "yes"}:
        backend = CachedBackend(backend)
    return backend
