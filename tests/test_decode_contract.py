"""Pin the decode format (serving side).

The training repo (`certa-rlcd`) is independent and re-implements this encoding; this test
freezes the exact rendered prompt so the two cannot silently drift apart. The assertion below
is byte-identical to certa-rlcd/tests/test_decode_contract.py.
"""

from __future__ import annotations

from certa.contract import PickOne, compile_question

# Byte-identical to certa-rlcd/tests/test_decode_contract.py — do not change one without the other.
CANONICAL_PROMPT = "ctx\n\nQuestion: q?\n\nA. alpha\nB. beta\n\nThe answer is"


def test_render_matches_pinned_contract():
    plan = compile_question("ctx", PickOne(type="choice", instructions="q?", criteria={"0": "alpha", "1": "beta"}))
    assert plan.prompt == CANONICAL_PROMPT
