"""The decision contract shared by the runtime and the training pipeline.

Import everything protocol-related from here so there is exactly one definition of how a
question is encoded, scored, and answered.
"""

from __future__ import annotations

from .constants import (
    ANSWER_CUE,
    BASE_CHECKPOINT,
    CHOICE_OPTION_LIMIT,
    DECODE_OPTION_LIMIT,
    MODEL_ALIAS,
    OPTION_LABELS,
    SCORE_LEVEL_MAX,
    SCORE_LEVEL_MIN,
)
from .encoding import (
    BOOLEAN_KEYS,
    CompileError,
    DecodePlan,
    assemble_answer,
    compile_question,
)
from .schema import (
    BooleanLabels,
    DecisionRequest,
    DecisionResponse,
    PickOne,
    PickOneResult,
    Question,
    Rate,
    RateResult,
    Result,
    TokenUsage,
    Verify,
    VerifyResult,
)
from .scoring import expected_level, peak_confidence

__all__ = [
    # constants
    "ANSWER_CUE",
    "BASE_CHECKPOINT",
    "CHOICE_OPTION_LIMIT",
    "DECODE_OPTION_LIMIT",
    "MODEL_ALIAS",
    "OPTION_LABELS",
    "SCORE_LEVEL_MAX",
    "SCORE_LEVEL_MIN",
    # encoding
    "BOOLEAN_KEYS",
    "CompileError",
    "DecodePlan",
    "assemble_answer",
    "compile_question",
    # scoring
    "expected_level",
    "peak_confidence",
    # schema
    "BooleanLabels",
    "DecisionRequest",
    "DecisionResponse",
    "PickOne",
    "PickOneResult",
    "Question",
    "Rate",
    "RateResult",
    "Result",
    "TokenUsage",
    "Verify",
    "VerifyResult",
]
