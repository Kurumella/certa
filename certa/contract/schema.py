"""Wire schemas for the decision protocol.

Requests describe *what to decide*; responses carry the typed outcome plus its
probability distribution. The three question kinds are distinguished by a ``type`` tag so
they can travel together in one payload.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator

from .constants import CHOICE_OPTION_LIMIT, SCORE_LEVEL_MAX, SCORE_LEVEL_MIN

# Free-form JSON is accepted for prose fields (a plain string, or nested structure).
Json = Any


# --- inbound: questions ------------------------------------------------------


class BooleanLabels(BaseModel):
    """Optional descriptions for the two poles of a yes/no question."""

    true: Json | None = None
    false: Json | None = None


class PickOne(BaseModel):
    """Choose exactly one option from an unordered set."""

    type: Literal["choice"]
    instructions: Json
    criteria: dict[str, Json | None]

    @field_validator("criteria")
    @classmethod
    def _within_option_limit(cls, value: dict) -> dict:
        count = len(value)
        if not 1 <= count <= CHOICE_OPTION_LIMIT:
            raise ValueError(f"expected 1..{CHOICE_OPTION_LIMIT} options, received {count}")
        return value


class Rate(BaseModel):
    """Place the state on an ordered scale; index 0 is the lowest level."""

    type: Literal["score"]
    instructions: Json
    criteria: list[Json]

    @field_validator("criteria")
    @classmethod
    def _within_level_range(cls, value: list) -> list:
        count = len(value)
        if not SCORE_LEVEL_MIN <= count <= SCORE_LEVEL_MAX:
            raise ValueError(
                f"expected {SCORE_LEVEL_MIN}..{SCORE_LEVEL_MAX} levels, received {count}"
            )
        return value


class Verify(BaseModel):
    """Judge whether a statement holds (yes/no)."""

    type: Literal["noul"]
    instructions: Json
    criteria: BooleanLabels | None = None


Question = Annotated[Union[PickOne, Rate, Verify], Field(discriminator="type")]


class DecisionRequest(BaseModel):
    state: Json
    questions: dict[str, Question]
    model: str | None = None


# --- outbound: answers -------------------------------------------------------


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class PickOneResult(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class RateResult(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, Json]
    probabilities: dict[int, float]


class VerifyResult(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


Result = Union[PickOneResult, RateResult, VerifyResult]


class DecisionResponse(BaseModel):
    model: str
    answers: dict[str, Result]
    usage: TokenUsage
