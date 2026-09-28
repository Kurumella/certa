"""Command-line entry point: launch the decision server."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="certa", description="Serve the Certa decision API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="model checkpoint to load (overrides CERTA_CHECKPOINT).",
    )
    args = parser.parse_args()

    if args.checkpoint:
        os.environ["CERTA_CHECKPOINT"] = args.checkpoint

    import uvicorn

    from .server import app_from_env

    uvicorn.run(app_from_env(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
