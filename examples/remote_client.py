"""Call a running Certa gateway over HTTP (the remote deployment path).

Start a gateway first, in another terminal:
    CERTA_CHECKPOINT=goutam/LFM2.5-1.2B-RLCD pip install 'certa[serve] @ git+https://github.com/Kurumella/certa.git'
    python -m certa --port 8000

Then:
    python examples/remote_client.py
"""

from __future__ import annotations

from certa.client import Client, choice, verify


def main() -> None:
    # add api_key="…" if the gateway sets CERTA_API_KEY
    with Client("http://localhost:8000") as client:
        resp = client.decide(
            "The checkout page returns a 500 error on Safari.",
            {
                "team": choice("Route this", {"billing": "charges", "technical": "bugs", "sales": "pricing"}),
                "urgent": verify("Is it urgent?"),
            },
        )
    print("team  :", resp.answers["team"].choice, round(resp.answers["team"].confidence, 3))
    print("urgent:", round(resp.answers["urgent"].noul, 3))


if __name__ == "__main__":
    main()
