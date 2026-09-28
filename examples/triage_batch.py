"""Batch-triage support tickets: route each one, and escalate the low-confidence ones.

This is the core Certa pattern — get a *calibrated* decision per item, then use the confidence
to decide when to act automatically vs. hand off to a human.

    python examples/triage_batch.py
"""

from __future__ import annotations

from certa.client import choice, score
from certa.engine import ModelEngine

TICKETS = [
    "I've been charged twice this month and nobody has replied in three days.",
    "How do I change the email address on my account?",
    "The app crashes every time I open the reports tab on iOS 17.",
    "Do you offer annual billing, and is there a discount?",
]

# Act automatically when the model is at least this confident; otherwise escalate.
ACT_THRESHOLD = 0.75


def main() -> None:
    engine = ModelEngine("goutam/LFM2.5-1.2B-RLCD")
    questions = {
        "team": choice(
            "Which team should handle this ticket?",
            {
                "billing": "charges and refunds",
                "technical": "bugs and outages",
                "sales": "plans and pricing",
                "account": "login and profile",
            },
        ),
        "priority": score("How urgent is it?", ["low", "normal", "high"]),
    }

    for ticket in TICKETS:
        answers = engine.decide(ticket, questions).answers
        team, priority = answers["team"], answers["priority"]
        decision = "auto-route" if team.confidence >= ACT_THRESHOLD else "ESCALATE (low confidence)"
        print(f"- {ticket[:58]!r}")
        print(f"    → {team.choice} (conf {team.confidence:.2f}), priority {priority.score:.2f}/2  ⇒ {decision}")


if __name__ == "__main__":
    main()
