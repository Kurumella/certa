"""Unit tests for the decision contract (no model required)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from certa.contract import (
    CompileError,
    PickOne,
    Rate,
    Verify,
    assemble_answer,
    compile_question,
    peak_confidence,
)


def test_confidence_endpoints():
    assert peak_confidence([1.0]) == 1.0  # single option
    assert peak_confidence([0.5, 0.5]) == pytest.approx(0.0)  # uniform
    assert peak_confidence([1.0, 0.0]) == pytest.approx(1.0)  # one-hot
    # (n * peak - 1) / (n - 1) for n=3, peak=0.7 -> (2.1 - 1) / 2 = 0.55
    assert peak_confidence([0.7, 0.2, 0.1]) == pytest.approx(0.55)


def test_choice_compiles_and_answers():
    q = PickOne(type="choice", instructions="pick a team", criteria={"billing": "money", "tech": None})
    plan = compile_question("a refund query", q)
    assert plan.kind == "choice"
    assert plan.keys == ("billing", "tech")
    assert "A. money" in plan.prompt and "B. tech" in plan.prompt  # null desc -> key
    assert plan.prompt.rstrip().endswith("The answer is")

    result = assemble_answer(plan, [0.8, 0.2])
    assert result.choice == "billing"
    assert result.probabilities == {"billing": pytest.approx(0.8), "tech": pytest.approx(0.2)}
    assert result.confidence == pytest.approx(0.6)


def test_score_expected_level_and_int_keys():
    q = Rate(type="score", instructions="how hot", criteria=["cold", "warm", "hot"])
    plan = compile_question("boiling", q)
    result = assemble_answer(plan, [0.0, 0.5, 0.5])
    assert result.score == pytest.approx(1.5)  # 0*0 + 1*0.5 + 2*0.5
    assert set(result.probabilities) == {0, 1, 2}
    assert result.legend == {0: "cold", 1: "warm", 2: "hot"}


def test_noul_reports_true_probability_without_confidence():
    q = Verify(type="noul", instructions="is it urgent")
    plan = compile_question("please respond asap", q)
    result = assemble_answer(plan, [0.9, 0.1])
    assert result.noul == pytest.approx(0.9)
    assert not hasattr(result, "confidence")


def test_choice_rejects_too_many_options_for_decode():
    criteria = {f"opt{i}": None for i in range(30)}
    q = PickOne(type="choice", instructions="many", criteria=criteria)
    with pytest.raises(CompileError):
        compile_question("state", q)


def test_schema_rejects_empty_choice_and_short_score():
    with pytest.raises(ValidationError):
        PickOne(type="choice", instructions="x", criteria={})
    with pytest.raises(ValidationError):
        Rate(type="score", instructions="x", criteria=["only one"])
