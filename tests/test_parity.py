"""Wire-compatibility check against the real typesafe-sdk.

Skipped automatically when the SDK isn't installed. When it is, this proves two things:
questions built with the SDK are accepted by our server, and our response validates
against the SDK's own response type.
"""

from __future__ import annotations

import pytest

pytest.importorskip("typesafe_sdk")

import typesafe_sdk as ts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from tests.test_server import UniformBackend  # noqa: E402
from certa.server import create_app  # noqa: E402


def test_sdk_questions_accepted_and_response_validates():
    questions = {
        "team": ts.Choice(instructions="route this", criteria={"billing": "x", "technical": "y"}),
        "anger": ts.Score(instructions="how angry", criteria=["calm", "annoyed", "furious"]),
        "urgent": ts.Noul(instructions="is it urgent?"),
    }
    payload = {
        "state": "charged twice, very upset",
        "questions": {qid: q.model_dump() for qid, q in questions.items()},
    }

    client = TestClient(create_app(UniformBackend()))
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200

    # The SDK's own response model must accept our body unchanged. Validate from raw
    # bytes, exactly as a real HTTP client does (JSON object keys arrive as strings and
    # are coerced to the declared int keys).
    parsed = ts.SystemOneResponse.model_validate_json(resp.content)
    assert parsed.answers["team"].choice in {"billing", "technical"}
    assert 0.0 <= parsed.answers["anger"].score <= 2.0
    assert 0.0 <= parsed.answers["urgent"].noul <= 1.0
