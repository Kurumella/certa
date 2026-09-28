"""Run calibrated typed decisions locally (no server), using the model engine directly.

    python examples/decide_local.py [checkpoint]

`checkpoint` defaults to the released Certa model; pass a local path (e.g. a training
`runs/.../best` dir) to try your own.
"""

from __future__ import annotations

import json
import sys
import time

from certa.client import choice, score, verify
from certa.engine import ModelEngine


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "goutam/LFM2.5-1.2B-RLCD"
    engine = ModelEngine(checkpoint, name="certa-1.2b-rlcd")

    state = (
        "I was double-charged for my subscription this month and support has ignored me "
        "for three days. This is completely unacceptable."
    )
    questions = {
        "team": choice(
            "Which team should handle this ticket?",
            {"billing": "payments and charges", "technical": "bugs and outages", "sales": "plans and pricing"},
        ),
        "anger": score("How angry does the customer seem?", ["calm", "annoyed", "furious"]),
        "urgent": verify("Does the message convey urgency?"),
    }

    engine.decide(state, questions)  # warm up (first call compiles CUDA kernels)
    start = time.time()
    response = engine.decide(state, questions)
    print(json.dumps(response.model_dump(), indent=2))
    print(f"warm latency: {(time.time() - start) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
