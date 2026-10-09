"""First-stage proposal agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .physics_info import (
    BOUNDS_RULE,
    frozen_display_names,
    render_parameter_guide,
    translate_updates,
)
from .types import RoundContext
from .evaluation import BASE_EVALUATION_SYSTEM_PROMPT
from .image_reader import IMAGE_READER_SYSTEM_PROMPT
from .utils import (
    b64_image,
    call_with_validation,
    coerce_updates,
    new_validation_counters,
    parameter_whitelist,
    redact_history_block,
    validate_candidates,
)
from ..config import LLM_MODEL_DEFAULT
from ..parameters import ParameterSet


PROPOSAL_SYSTEM_PROMPT = (
    "You are a hydrologic calibration strategist."
    "\nGiven current metrics, history, and images, propose diverse parameter update strategies."
    "\nReturn STRICT JSON with a `candidates` list; each candidate needs an `id`, `goal` (short description) and `updates` mapping"
    " parameter names to either numbers or {\"op\": \"*|+|-|=\", \"value\": number} for multiplicative/additive adjustments."
    "\nMake every candidate explore a clearly different direction within the allowed parameter bounds; large steps are permitted."
)


def _slice(text: str, start: str, end: Optional[str] = None) -> str:
    """Verbatim segment of ``text`` from ``start`` up to (excluding) ``end``."""
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


# Diagnostic instructions of the perception stage, verbatim, minus the sentence
# that forbids proposing updates (which would contradict a single-call design).
_IMAGE_DIAGNOSIS_SEGMENT = _slice(
    IMAGE_READER_SYSTEM_PROMPT,
    "\nYou will receive a 2x2 montage image",
    "\nDo NOT propose parameter updates yet",
)
_NUMERIC_DIAGNOSIS_SEGMENT = (
    "\nYou will receive numerical diagnostic signatures (`image_summary`) computed from the same"
    " flow-duration curve and event hydrographs the montage would show."
    + _slice(IMAGE_READER_SYSTEM_PROMPT, "\nYour task is to describe the visible mismatches",
             "\nDo NOT propose parameter updates yet")
)
# Review instructions of the verification stage, verbatim: diversity, bounds and
# the warm-up down-weighting rule.
_REVIEW_DIVERSITY_SENTENCE = _slice(
    BASE_EVALUATION_SYSTEM_PROMPT,
    " Encourage markedly different strategies",
    "\nAll updates must stay within hydrologic bounds.",
)
_REVIEW_BOUNDS_AND_WARMUP = _slice(
    BASE_EVALUATION_SYSTEM_PROMPT,
    "\nAll updates must stay within hydrologic bounds.",
)


def build_direct_matched_system_prompt(perception: str = "image") -> str:
    """System prompt of the information-matched single-call baseline.

    Concatenates (1) the perception agent's diagnostic instructions, (2) the
    verification agent's review instructions — including the warm-up
    down-weighting rule — and (3) the proposal agent's output contract, so the
    only things removed relative to full INCA are the stage split, the
    independent verification call and the failure memory.
    """
    diagnosis = _IMAGE_DIAGNOSIS_SEGMENT if perception == "image" else (
        _NUMERIC_DIAGNOSIS_SEGMENT if perception == "numeric" else "")
    return (
        "You are a hydrologic calibration strategist who, in a single pass, diagnoses model behavior,"
        " reviews candidate updates, and proposes the final candidates."
        + ("\n[Diagnosis]" + diagnosis if diagnosis else "")
        + "\n[Review]"
        "\nYou also receive recent history data (`history_payload`: every earlier round, its candidates and"
        " their scores) and performance metrics."
        "\nAssess your own candidates, adjust them for diversity and safety."
        + _REVIEW_DIVERSITY_SENTENCE
        + _REVIEW_BOUNDS_AND_WARMUP
        + "\n[Proposal]"
        + _slice(PROPOSAL_SYSTEM_PROMPT, "\nGiven current metrics, history, and images")
    )


class ProposalAgent:
    def __init__(self,
                 model: str = LLM_MODEL_DEFAULT,
                 physics_information: bool = True,
                 display_name_map: Optional[Dict[str, str]] = None,
                 detail_output: bool = False,
                 provider: Optional[str] = None,
                 thinking_budget: Optional[int] = None,
                 direct_matched: bool = False,
                 perception: str = "image",
                 allow_wildcard: bool = True):
        self.allow_wildcard = allow_wildcard
        self.model = model
        self.provider = provider
        self.thinking_budget = thinking_budget
        self.direct_matched = direct_matched
        self.usage_log: List[Dict[str, Any]] = []
        self.validation_log: List[Dict[str, Any]] = []
        self.validation_counters: Dict[str, int] = new_validation_counters()
        self.physics_information = physics_information
        self.display_name_map = display_name_map or {}
        self.reverse_display_map = {v: k for k, v in self.display_name_map.items()}
        self.detail_output = detail_output
        frozen_names = frozen_display_names(self.display_name_map)
        frozen_text = ", ".join(frozen_names) if frozen_names else ""
        base_prompt = (
            build_direct_matched_system_prompt(perception) if direct_matched else PROPOSAL_SYSTEM_PROMPT
        )
        self.system_prompt = (
            base_prompt
            + (f"\nDo NOT modify {frozen_text}—those parameters remain fixed." if frozen_text else "")
            + "\n"
            + render_parameter_guide(physics_information, self.display_name_map)
        )

    def build_prompt(self,
                     context: RoundContext,
                     k: int,
                     history_payload: Optional[Dict[str, Any]] = None) -> str:
        payload: Dict[str, Any] = {
            "round": context.round_index,
            "current_params": context.display_params or context.params,
            "aggregate_metrics": context.aggregate_metrics,
            "full_metrics": context.full_metrics,
            "event_metrics": context.event_metrics,
            "history_summary": context.history_summary,
            "image_summary": context.image_summary,
            "failure_summary": context.failure_summary,
            "failed_attempts": context.failure_details,
        }
        if self.direct_matched:
            # Same object the verification agent receives in the staged workflow.
            payload["history_payload"] = history_payload if history_payload is not None else {}
        payload.update({
            "requested_candidates": k,
            "notes": context.description,
            "physics_information": context.physics_information,
            "parameter_guide": context.physics_prompt,
            "parameter_names": context.param_display_names,
        })
        if context.parameter_bounds or context.at_bound:
            payload["parameter_bounds"] = context.parameter_bounds
            payload["at_bound"] = context.at_bound
            payload["bounds_rule"] = BOUNDS_RULE
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def propose(self,
                context: RoundContext,
                k: int,
                return_log: bool = False,
                history_payload: Optional[Dict[str, Any]] = None,
                log_round: Optional[str] = None):
        user_prompt = self.build_prompt(context, k, history_payload=history_payload)
        if context.images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(img)}"}} for img in context.images)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        whitelist = parameter_whitelist(self.display_name_map, allow_wildcard=self.allow_wildcard)
        response, data, candidates, usage, attempts = call_with_validation(
            messages,
            model=self.model,
            provider=self.provider,
            thinking_budget=self.thinking_budget,
            stage="proposal",
            round_label=context.round_index,
            log_round=log_round,
            parse=lambda d: validate_candidates(d, "candidates", k, whitelist),
            usage_log=self.usage_log,
            validation_log=self.validation_log,
            counters=self.validation_counters,
        )
        raw = response.text
        if not return_log:
            return candidates

        log_payload = {
            "stage": "proposal",
            "round": context.round_index,
            "system_prompt": self.system_prompt,
            "user_prompt": redact_history_block(user_prompt),
            "input_files": [Path(img).name for img in context.images],
            "input_paths": [str(Path(img).resolve()) for img in context.images],
            "output_text": raw,
            "parsed_output": data,
            "usage": usage,
            "validation_attempts": attempts,
        }
        return candidates, log_payload

    def apply_candidates(self, base_params: ParameterSet, candidates: List[Dict[str, Any]]) -> List[ParameterSet]:
        param_sets: List[ParameterSet] = []
        for cand in candidates:
            updates = translate_updates(cand.get("updates", {}), self.reverse_display_map)
            new_params = coerce_updates(base_params, updates, stats=self.validation_counters)
            param_sets.append(new_params)
        return param_sets


__all__ = ["ProposalAgent", "PROPOSAL_SYSTEM_PROMPT", "build_direct_matched_system_prompt"]
