"""Backend selection and caching (no model load)."""

from __future__ import annotations

import pytest

from certa.backend import CachedBackend, RemoteBackend, from_env
from certa.contract import DecisionResponse, TokenUsage, Verify, VerifyResult


class CountingBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        return DecisionResponse(
            model="fake",
            answers={q: VerifyResult(noul=0.5) for q in questions},
            usage=TokenUsage(),
        )


def test_cache_serves_repeats_and_misses_on_change():
    inner = CountingBackend()
    cached = CachedBackend(inner)
    q = {"u": Verify(type="noul", instructions="urgent?")}

    cached.decide("same state", q)
    cached.decide("same state", q)
    assert inner.calls == 1  # second identical request served from cache

    cached.decide("different state", q)
    assert inner.calls == 2  # different input → cache miss


def test_from_env_remote(monkeypatch):
    monkeypatch.setenv("CERTA_BACKEND", "remote")
    monkeypatch.setenv("CERTA_BASE_URL", "http://gateway:8000")
    monkeypatch.delenv("CERTA_CACHE", raising=False)
    assert isinstance(from_env(), RemoteBackend)


def test_from_env_cache_wraps(monkeypatch):
    monkeypatch.setenv("CERTA_BACKEND", "remote")
    monkeypatch.setenv("CERTA_BASE_URL", "http://gateway:8000")
    monkeypatch.setenv("CERTA_CACHE", "1")
    assert isinstance(from_env(), CachedBackend)


def test_from_env_vllm_is_optional_stub(monkeypatch):
    monkeypatch.setenv("CERTA_BACKEND", "vllm")
    with pytest.raises(NotImplementedError):
        from_env()


def test_from_env_rejects_unknown(monkeypatch):
    monkeypatch.setenv("CERTA_BACKEND", "bogus")
    with pytest.raises(ValueError):
        from_env()
