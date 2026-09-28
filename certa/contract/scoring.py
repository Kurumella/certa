"""Numeric helpers that turn a probability distribution into reported figures.

Pure functions with no framework dependencies, so the training package can reuse them
for evaluation.
"""

from __future__ import annotations

from typing import Sequence


def peak_confidence(distribution: Sequence[float]) -> float:
    """Map a distribution to a single confidence figure in ``[0, 1]``.

    The figure rescales the largest probability against a uniform baseline: it is ``0``
    when every option is equally likely and ``1`` when all mass sits on one option. A
    single-option distribution is treated as fully confident.

        conf = (n * p_max - 1) / (n - 1)
    """
    n = len(distribution)
    if n <= 1:
        return 1.0
    scaled = (n * max(distribution) - 1.0) / (n - 1.0)
    return min(1.0, max(0.0, scaled))


def expected_level(distribution: Sequence[float]) -> float:
    """Probability-weighted mean of the ordered level indices (0..n-1)."""
    return float(sum(index * p for index, p in enumerate(distribution)))
