"""Thin HTTP client for a Certa (or any Jev-compatible) decision endpoint.

Provides small builders for the three question kinds and a client that posts a decision
request and parses the typed response. Requires the ``client`` extra (httpx).
"""

from __future__ import annotations

import os
from typing import Mapping

from .contract import (
    BooleanLabels,
    DecisionResponse,
    PickOne,
    Question,
    Rate,
    Verify,
)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_TIMEOUT = 10.0


def choice(instructions, criteria: Mapping[str, object | None]) -> PickOne:
    return PickOne(type="choice", instructions=instructions, criteria=dict(criteria))


def score(instructions, criteria: list) -> Rate:
    return Rate(type="score", instructions=instructions, criteria=list(criteria))


def verify(instructions, *, true=None, false=None) -> Verify:
    labels = BooleanLabels(true=true, false=false) if (true or false) else None
    return Verify(type="noul", instructions=instructions, criteria=labels)


class Client:
    """Posts decisions to a running endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        import httpx  # local import keeps httpx an optional dependency

        self._model = model
        headers = {}
        key = api_key or os.environ.get("CERTA_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=base_url or os.environ.get("CERTA_BASE_URL", _DEFAULT_BASE_URL),
            headers=headers,
            timeout=timeout,
        )

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        payload = {
            "state": state,
            "questions": {qid: q.model_dump() for qid, q in questions.items()},
        }
        if self._model:
            payload["model"] = self._model
        reply = self._http.post("/v1/systemone", json=payload)
        reply.raise_for_status()
        return DecisionResponse.model_validate(reply.json())

    def models(self) -> list[dict]:
        reply = self._http.get("/v1/models")
        reply.raise_for_status()
        return reply.json()["models"]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
