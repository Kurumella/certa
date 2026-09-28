"""Pick a deployment backend at runtime — the calling code is identical either way.

Local (model in this process):
    CERTA_BACKEND=local CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD python examples/backends.py

Remote (a Certa gateway on another machine):
    CERTA_BACKEND=remote CERTA_BASE_URL=http://gateway:8000 CERTA_API_KEY=secret \
        python examples/backends.py
"""

from __future__ import annotations

from certa.backend import from_env
from certa.client import choice, verify


def main() -> None:
    backend = from_env()  # LocalBackend or RemoteBackend, chosen by CERTA_BACKEND
    print(f"backend: {type(backend).__name__} (model={backend.name})")

    resp = backend.decide(
        "I was double-charged and support has ignored me for three days.",
        {
            "team": choice("Route this ticket", {"billing": "charges", "technical": "bugs", "sales": "pricing"}),
            "urgent": verify("Does the message convey urgency?"),
        },
    )
    print("team  :", resp.answers["team"].choice, round(resp.answers["team"].confidence, 3))
    print("urgent:", round(resp.answers["urgent"].noul, 3))


if __name__ == "__main__":
    main()
