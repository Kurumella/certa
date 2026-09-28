"""Fixed values that pin the decision protocol.

Grouped separately so both the runtime and the training package can reference the exact
same limits and labels without importing anything heavier.
"""

from __future__ import annotations

# Model identity advertised on the wire, plus the base checkpoint the runtime loads.
MODEL_ALIAS = "certa-latest"
BASE_CHECKPOINT = "LiquidAI/LFM2.5-1.2B-Base"

# Single-token option labels. Each of these is a single token in the target vocabulary
# both on its own and when preceded by a space, which is what lets a decision resolve in
# one forward pass.
OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Text that immediately precedes the scored token.
ANSWER_CUE = "The answer is"

# Protocol limits mirrored from the target API.
CHOICE_OPTION_LIMIT = 255
SCORE_LEVEL_MIN = 2
SCORE_LEVEL_MAX = 10

# How many options the single-letter decode can address. Anything larger needs a
# different labelling strategy and is rejected at the edge.
DECODE_OPTION_LIMIT = len(OPTION_LABELS)
