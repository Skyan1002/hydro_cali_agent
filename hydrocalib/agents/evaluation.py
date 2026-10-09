"""Second-stage evaluation agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .physics_info import BOUNDS_RULE, render_parameter_guide, translate_updates
from .types import RoundContext
from .utils import (
    b64_image,
    call_with_validation,
    coerce_updates,
    new_validation_counters,
    parameter_whitelist,
    redact_history_block,
    validate_candidates,
)
from ..config import LLM_MODEL_REASONING
from ..parameters import ParameterSet


BASE_EVALUATION_SYSTEM_PROMPT = (
    "You are a hydrologic calibration reviewer."
    "\nYou receive initial candidate parameter updates, recent history data, and performance metrics."
    "\nAssess the candidates, adjust them for diversity and safety, and output STRICT JSON with `refined_candidates`:"
    " list items containing `id`, `origin` (which proposal inspired it), `rationale`, and `updates` mapping parameters to numbers"
    " or operation dictionaries. Encourage markedly different strategies (timing focus, peak shaping, baseflow control, etc.)."
    "\nAll updates must stay within hydrologic bounds."
    "\nGive lower weight to event peaks from the earliest months because warm-up issues make them less reliable; focus more on"
    " later-season peaks when ranking candidates."
)


class EvaluationAgent:
    def __init__(self,
                 model: str = LLM_MODEL_REASONING,
                 physics_information: bool = True,
                 display_name_map: Optional[Dict[str, str]] = None,
                 detail_output: bool = False,
                 provider: Optional[str] = None,
                 thinking_budget: Optional[int] = None,
                 allow_wildcard: bool = True):
        self.allow_wildcard = allow_wildcard
        self.model = model
        self.provider = provider
        self.thinking_budget = thinking_budget
        self.usage_log: List[Dict[str, Any]] = []
        self.validation_log: List[Dict[str, Any]] = []
        self.validation_counters: Dict[str, int] = new_validation_counters()
        self.physics_information = physics_information
        self.display_name_map = display_name_map or {}
        self.reverse_display_map = {v: k for k, v in self.display_name_map.items()}
        self.detail_output = detail_output
        self.system_prompt = (
            BASE_EVALUATION_SYSTEM_PROMPT
            + "\n"
            + render_parameter_guide(physics_information, self.display_name_map)
        )

    def build_prompt(self,
                     context: RoundContext,
                     proposals: List[Dict[str, Any]],
                     history_payload: Dict[str, Any],
                     k: int) -> str:
        payload = {
            "round": context.round_index,
            "current_params": context.display_params or context.params,
            "aggregate_metrics": context.aggregate_metrics,
            "full_metrics": context.full_metrics,
            "event_metrics": context.event_metrics,
            "history_summary": context.history_summary,
            "image_summary": context.image_summary,
            "failure_summary": context.failure_summary,
            "failed_attempts": context.failure_details,
            "history_payload": history_payload,
            "initial_proposals": proposals,
            "requested_candidates": k,
            "physics_information": context.physics_information,
            "parameter_guide": context.physics_prompt,
            "parameter_names": context.param_display_names,
        }
        if context.parameter_bounds or context.at_bound:
            payload["parameter_bounds"] = context.parameter_bounds
            payload["at_bound"] = context.at_bound
            payload["bounds_rule"] = BOUNDS_RULE
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def refine(self,
               context: RoundContext,
               proposals: List[Dict[str, Any]],
               history_payload: Dict[str, Any],
               k: int,
               return_log: bool = False,
               log_round: Optional[str] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]]:
        user_prompt = self.build_prompt(context, proposals, history_payload, k)
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
        response, data, refined, usage, attempts = call_with_validation(
            messages,
            model=self.model,
            provider=self.provider,
            thinking_budget=self.thinking_budget,
            stage="evaluation",
            round_label=context.round_index,
            log_round=log_round,
            parse=lambda d: validate_candidates(d, "refined_candidates", k, whitelist),
            usage_log=self.usage_log,
            validation_log=self.validation_log,
            counters=self.validation_counters,
        )
        raw = response.text
        meta = {
            "rationale": data.get("rationale", ""),
            "risk": data.get("risk", ""),
            "focus": data.get("focus", ""),
        }
        log_payload = None
        if return_log:
            log_payload = {
                "stage": "evaluation",
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

        return refined, meta, log_payload

    def apply_candidates(self, base_params: ParameterSet, refined: List[Dict[str, Any]]) -> List[ParameterSet]:
        param_sets: List[ParameterSet] = []
        for cand in refined:
            updates = translate_updates(cand.get("updates", {}), self.reverse_display_map)
            new_params = coerce_updates(base_params, updates, stats=self.validation_counters)
            param_sets.append(new_params)
        return param_sets


__all__ = ["EvaluationAgent", "BASE_EVALUATION_SYSTEM_PROMPT"]
