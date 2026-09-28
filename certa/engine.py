"""Model-backed decision engine.

Wraps a causal language model and resolves a batch of questions in a single forward pass
by reading the logits of the option-label tokens directly. No autoregressive generation
happens, so latency is flat in the number of options.
"""

from __future__ import annotations

import json
import os
from typing import Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .contract import (
    DecodePlan,
    DecisionResponse,
    Question,
    TokenUsage,
    assemble_answer,
    compile_question,
)
from .contract.constants import BASE_CHECKPOINT, MODEL_ALIAS
from .tokens import resolve_option_tokens

# Sidecar file (written by the calibration step) holding a fitted decode temperature.
CALIBRATION_FILE = "certa_calibration.json"


def _select_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_temperature(checkpoint: str) -> float:
    """Read a fitted temperature from a local checkpoint directory, else 1.0."""
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, CALIBRATION_FILE)
        if os.path.isfile(path):
            with open(path) as fh:
                return float(json.load(fh).get("temperature", 1.0))
    return 1.0


class ModelEngine:
    """Runs decisions against a loaded checkpoint."""

    def __init__(
        self,
        checkpoint: str = BASE_CHECKPOINT,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        name: str = MODEL_ALIAS,
        temperature: float | None = None,
    ) -> None:
        self.name = name
        self.device = _select_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the final position aligned across a padded batch, so one
        # slice of the last-position logits covers every prompt.
        self.tokenizer.padding_side = "left"

        if dtype is None:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self._label_tokens = resolve_option_tokens(self.tokenizer)
        self.temperature = (
            temperature if temperature is not None else _load_temperature(checkpoint)
        )

    @torch.inference_mode()
    def distributions(self, plans: list[DecodePlan]) -> tuple[list[list[float]], int]:
        """Score a batch of plans in one forward pass.

        Returns the per-plan option distribution (aligned with each plan's keys) and the
        total number of prompt tokens consumed. This is the shared primitive used both by
        the API and by offline evaluation.
        """
        if not plans:
            return [], 0
        batch = self.tokenizer(
            [plan.prompt for plan in plans], return_tensors="pt", padding=True
        ).to(self.device)
        final_logits = self.model(**batch).logits[:, -1, :]

        result = []
        for row, plan in zip(final_logits, plans):
            option_ids = self._label_tokens[: plan.width]
            scaled = row[option_ids].float() / self.temperature
            result.append(torch.softmax(scaled, dim=-1).tolist())
        return result, int(batch["attention_mask"].sum().item())

    def decide(self, state, questions: Mapping[str, Question]) -> DecisionResponse:
        qids = list(questions)
        plans = [compile_question(state, questions[qid]) for qid in qids]
        distributions, input_tokens = self.distributions(plans)

        answers = {
            qid: assemble_answer(plan, dist)
            for qid, plan, dist in zip(qids, plans, distributions)
        }
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=len(plans))
        return DecisionResponse(model=self.name, answers=answers, usage=usage)
