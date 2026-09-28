"""Certa: fast, calibrated, typed decisions on open-weight models.

The public surface is intentionally small: the decision contract (schemas + encoding),
the model engine, and the server factory. Heavier pieces are imported lazily so that
``import certa`` stays cheap and free of optional dependencies.
"""

from __future__ import annotations

from .backend import (
    CachedBackend,
    DecisionBackend,
    LocalBackend,
    RemoteBackend,
    from_env,
)
from .contract import (
    DecisionRequest,
    DecisionResponse,
    PickOne,
    Rate,
    Verify,
    assemble_answer,
    compile_question,
    peak_confidence,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # backends (deployment)
    "DecisionBackend",
    "LocalBackend",
    "RemoteBackend",
    "CachedBackend",
    "from_env",
    # contract
    "DecisionRequest",
    "DecisionResponse",
    "PickOne",
    "Rate",
    "Verify",
    "assemble_answer",
    "compile_question",
    "peak_confidence",
]
