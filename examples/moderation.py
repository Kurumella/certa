"""Content moderation in one pass — a different domain, using all three primitives together.

    python examples/moderation.py
"""

from __future__ import annotations

from certa.client import choice, score, verify
from certa.engine import ModelEngine


def main() -> None:
    engine = ModelEngine("goutam/LFM2.5-1.2B-RLCD")
    text = "You people are idiots and this product is a total scam. I want my money back now."

    answers = engine.decide(
        text,
        {
            "category": choice(
                "What kind of message is this?",
                {
                    "complaint": "unhappy but legitimate feedback",
                    "abuse": "insults or harassment",
                    "spam": "promotional or junk content",
                    "praise": "positive feedback",
                },
            ),
            "severity": score("How severe is any policy violation?", ["none", "mild", "severe"]),
            "toxic": verify("Does the message contain insults or harassment?"),
            "actionable": verify("Is there a concrete request to act on?"),
        },
    ).answers

    print("category  :", answers["category"].choice, f"(conf {answers['category'].confidence:.2f})")
    print("severity  :", f"{answers['severity'].score:.2f} / 2")
    print("toxic     :", f"{answers['toxic'].noul:.2f}")
    print("actionable:", f"{answers['actionable'].noul:.2f}")


if __name__ == "__main__":
    main()
