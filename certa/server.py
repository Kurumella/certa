"""HTTP surface for the decision engine.

Exposes the decision endpoint plus a model listing, with optional bearer-token auth and a
per-request id header. The engine is injected, so the app can be built against a fake
backend in tests without loading a checkpoint.
"""

from __future__ import annotations

import hmac
import os
import uuid
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .backend import DecisionBackend
from .batching import MicroBatcher
from .contract import CompileError, DecisionRequest, DecisionResponse

REQUEST_ID_HEADER = "x-certa-request-id"
_API_KEY_ENV = "CERTA_API_KEY"


def _bearer_guard(api_key: str | None) -> Callable:
    """Build a dependency that enforces a bearer token when one is configured."""

    def guard(authorization: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        expected = f"Bearer {api_key}"
        # Constant-time compare so a wrong key can't be recovered by timing the response.
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    return guard


def create_app(
    backend: DecisionBackend,
    *,
    api_key: str | None = None,
    batcher: MicroBatcher | None = None,
) -> FastAPI:
    """Build the FastAPI app around a backend. If ``batcher`` is given, concurrent requests are
    coalesced into shared forward passes; otherwise each request calls ``backend.decide``."""
    app = FastAPI(title="Certa", summary="Fast, calibrated, typed decisions.")
    require_auth = Depends(_bearer_guard(api_key))

    @app.middleware("http")
    async def tag_request(request: Request, call_next):
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = str(uuid.uuid4())
        return response

    @app.exception_handler(CompileError)
    async def _on_compile_error(_: Request, exc: CompileError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": backend.name}

    @app.get("/v1/models", dependencies=[require_auth])
    def list_models() -> dict:
        return {"models": [{"name": backend.name}]}

    @app.post("/v1/systemone", response_model=DecisionResponse, dependencies=[require_auth])
    async def decide(request: DecisionRequest) -> DecisionResponse:
        if batcher is not None:
            return await batcher.decide(request.state, request.questions)
        return backend.decide(request.state, request.questions)

    return app


def _engine_of(backend: DecisionBackend):
    """The underlying ModelEngine, whether the backend is local or a cache over a local one."""
    return getattr(backend, "engine", None) or getattr(getattr(backend, "_inner", None), "engine", None)


def app_from_env() -> FastAPI:
    """Construct the production app from environment configuration.

    ``CERTA_BACKEND`` (local|remote|vllm), ``CERTA_CHECKPOINT``, ``CERTA_API_KEY``,
    ``CERTA_CACHE`` (see :func:`certa.backend.from_env`), and ``CERTA_BATCH`` to enable
    cross-request micro-batching (local backends only).
    """
    from .backend import from_env

    backend = from_env()
    batcher = None
    if os.environ.get("CERTA_BATCH", "").lower() in {"1", "true", "yes"}:
        engine = _engine_of(backend)
        if engine is not None:
            batcher = MicroBatcher(engine, name=backend.name)
    return create_app(backend, api_key=os.environ.get(_API_KEY_ENV), batcher=batcher)
