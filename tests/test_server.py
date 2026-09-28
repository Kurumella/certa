"""Server tests using a fake backend so no checkpoint is loaded."""

from __future__ import annotations

from typing import Mapping

from fastapi.testclient import TestClient

from certa.contract import (
    DecisionResponse,
    Question,
    TokenUsage,
    assemble_answer,
    compile_question,
)
from certa.server import REQUEST_ID_HEADER, create_app


class UniformBackend:
    """Answers every question with a uniform distribution over its options."""

    name = "certa-test"

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        answers = {}
        for qid, question in questions.items():
            plan = compile_question(state, question)  # may raise CompileError -> 422
            uniform = [1.0 / plan.width] * plan.width
            answers[qid] = assemble_answer(plan, uniform)
        return DecisionResponse(model=self.name, answers=answers, usage=TokenUsage())


def _payload():
    return {
        "state": "customer is furious about a double charge",
        "questions": {
            "team": {"type": "choice", "instructions": "route", "criteria": {"billing": "x", "tech": "y"}},
            "anger": {"type": "score", "instructions": "how angry", "criteria": ["calm", "mad"]},
            "urgent": {"type": "noul", "instructions": "urgent?"},
        },
    }


def test_decide_returns_typed_answers():
    client = TestClient(create_app(UniformBackend()))
    resp = client.post("/v1/systemone", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "certa-test"
    assert body["answers"]["team"]["type"] == "choice"
    assert body["answers"]["anger"]["type"] == "score"
    assert body["answers"]["urgent"]["type"] == "noul"
    assert REQUEST_ID_HEADER in resp.headers


def test_health_and_models():
    client = TestClient(create_app(UniformBackend()))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["models"][0]["name"] == "certa-test"


def test_validation_error_is_422():
    client = TestClient(create_app(UniformBackend()))
    bad = {"state": "x", "questions": {"q": {"type": "score", "instructions": "i", "criteria": ["only one"]}}}
    assert client.post("/v1/systemone", json=bad).status_code == 422


def test_too_many_options_is_422():
    client = TestClient(create_app(UniformBackend()))
    payload = {
        "state": "x",
        "questions": {
            "q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(30)}}
        },
    }
    assert client.post("/v1/systemone", json=payload).status_code == 422


def test_bearer_auth_enforced_when_key_set():
    client = TestClient(create_app(UniformBackend(), api_key="secret"))
    assert client.post("/v1/systemone", json=_payload()).status_code == 401
    ok = client.post(
        "/v1/systemone", json=_payload(), headers={"Authorization": "Bearer secret"}
    )
    assert ok.status_code == 200
