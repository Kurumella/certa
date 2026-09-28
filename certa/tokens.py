"""Resolving option labels to vocabulary token ids.

Shared by the runtime engine and the training pipeline so that the tokens scored at train
time are exactly those scored at serve time. Needs only a tokenizer, no torch.
"""

from __future__ import annotations

from .contract import OPTION_LABELS


def resolve_option_tokens(tokenizer, labels=OPTION_LABELS) -> list[int]:
    """Token id for each option label as it appears after a space (mid-sentence).

    Each label is expected to encode to a single token; the last id is taken defensively
    in case the tokenizer emits a leading marker.
    """
    ids = []
    for label in labels:
        encoded = tokenizer.encode(f" {label}", add_special_tokens=False)
        ids.append(encoded[-1])
    return ids
