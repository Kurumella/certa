"""Lowering questions to a single-forward-pass decode and lifting the result back.

A question becomes a :class:`DecodePlan`: a prompt whose continuation is scored over a
fixed set of option-label tokens, plus the keys that name each option. The runtime feeds
the prompt to a model, reads the label logits, and hands the resulting distribution back
here to be shaped into a typed answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from .constants import ANSWER_CUE, DECODE_OPTION_LIMIT, OPTION_LABELS
from .schema import (
    PickOne,
    PickOneResult,
    Question,
    Rate,
    RateResult,
    Result,
    Verify,
    VerifyResult,
)
from .scoring import expected_level, peak_confidence

# Keys used for the two poles of a yes/no decision; the first is the reported probability.
BOOLEAN_KEYS = ("true", "false")


class CompileError(ValueError):
    """A question cannot be lowered to a decode (e.g. too many options)."""


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class DecodePlan:
    """Everything needed to run and interpret one decision."""

    kind: str  # "choice" | "score" | "noul"
    prompt: str
    keys: tuple  # option-key strings, level ints, or BOOLEAN_KEYS
    legend: dict | None = None

    @property
    def width(self) -> int:
        return len(self.keys)


def _render(state, instructions, options: Sequence[str]) -> str:
    body = [
        _text(state).strip(),
        "",
        f"Question: {_text(instructions).strip()}",
        "",
    ]
    body += [f"{label}. {option}" for label, option in zip(OPTION_LABELS, options)]
    body += ["", ANSWER_CUE]
    return "\n".join(body)


def compile_question(state, question: Question) -> DecodePlan:
    """Lower one question into a :class:`DecodePlan`.

    Raises :class:`CompileError` when the option count exceeds the single-letter decode.
    """
    if isinstance(question, PickOne):
        keys = tuple(question.criteria)
        options = [d if d is not None else k for k, d in question.criteria.items()]
        kind, legend = "choice", None
    elif isinstance(question, Rate):
        keys = tuple(range(len(question.criteria)))
        options = list(question.criteria)
        kind, legend = "score", {i: question.criteria[i] for i in keys}
    elif isinstance(question, Verify):
        labels = question.criteria
        keys = BOOLEAN_KEYS
        options = [
            (labels.true if labels and labels.true is not None else "Yes"),
            (labels.false if labels and labels.false is not None else "No"),
        ]
        kind, legend = "noul", None
    else:  # pragma: no cover - the discriminated union covers all cases
        raise CompileError(f"unsupported question: {question!r}")

    if len(keys) > DECODE_OPTION_LIMIT:
        raise CompileError(
            f"{kind} carries {len(keys)} options but the decode addresses at most "
            f"{DECODE_OPTION_LIMIT}"
        )

    rendered = [o if isinstance(o, str) else _text(o) for o in options]
    return DecodePlan(kind=kind, prompt=_render(state, question.instructions, rendered), keys=keys, legend=legend)


def assemble_answer(plan: DecodePlan, distribution: Sequence[float]) -> Result:
    """Shape an option distribution into the typed result for ``plan``'s kind."""
    probs = [float(p) for p in distribution]
    if plan.kind == "choice":
        winner = plan.keys[probs.index(max(probs))]
        return PickOneResult(
            choice=str(winner),
            confidence=peak_confidence(probs),
            probabilities={str(k): p for k, p in zip(plan.keys, probs)},
        )
    if plan.kind == "score":
        return RateResult(
            score=expected_level(probs),
            confidence=peak_confidence(probs),
            legend=plan.legend or {},
            probabilities={int(k): p for k, p in zip(plan.keys, probs)},
        )
    return VerifyResult(noul=probs[0])
