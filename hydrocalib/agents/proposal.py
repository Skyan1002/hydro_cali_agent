"""First-stage proposal agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .physics_info import (
    frozen_display_names,
    render_parameter_guide,
    translate_updates,
)
from .types import RoundContext
from .utils import (
    b64_image,
    coerce_updates,
    extract_json_block,
    get_client,
    redact_history_block,
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


class ProposalAgent:
    def __init__(self,
                 model: str = LLM_MODEL_DEFAULT,
                 physics_information: bool = True,
                 display_name_map: Optional[Dict[str, str]] = None,
                 detail_output: bool = False):
        self.model = model
        self.client = get_client()
        self.physics_information = physics_information
        self.display_name_map = display_name_map or {}
        self.reverse_display_map = {v: k for k, v in self.display_name_map.items()}
        self.detail_output = detail_output
        frozen_names = frozen_display_names(self.display_name_map)
        frozen_text = ", ".join(frozen_names) if frozen_names else ""
        self.system_prompt = (
            PROPOSAL_SYSTEM_PROMPT
            + (f"\nDo NOT modify {frozen_text}—those parameters remain fixed." if frozen_text else "")
            + "\n"
            + render_parameter_guide(physics_information, self.display_name_map)
        )

    def build_prompt(self, context: RoundContext, k: int) -> str:
        payload = {
            "round": context.round_index,
            "current_params": context.display_params or context.params,
            "aggregate_metrics": context.aggregate_metrics,
            "full_metrics": context.full_metrics,
            "event_metrics": context.event_metrics,
            "history_summary": context.history_summary,
            "requested_candidates": k,
            "notes": context.description,
            "physics_information": context.physics_information,
            "parameter_guide": context.physics_prompt,
            "parameter_names": context.param_display_names,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def propose(self, context: RoundContext, k: int, return_log: bool = False):
        user_prompt = self.build_prompt(context, k)
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        raw = response.choices[0].message.content
        data = extract_json_block(raw)
        candidates = data.get("candidates", [])[:k]
        if not return_log:
            return candidates

        log_payload = {
            "stage": "proposal",
            "round": context.round_index,
            "system_prompt": self.system_prompt,
            "user_prompt": redact_history_block(user_prompt),
            "input_files": [Path(img).name for img in context.images],
            "output_text": raw,
            "parsed_output": data,
        }
        return candidates, log_payload

    def apply_candidates(self, base_params: ParameterSet, candidates: List[Dict[str, Any]]) -> List[ParameterSet]:
        param_sets: List[ParameterSet] = []
        for cand in candidates:
            updates = translate_updates(cand.get("updates", {}), self.reverse_display_map)
            new_params = coerce_updates(base_params, updates)
            param_sets.append(new_params)
        return param_sets


__all__ = ["ProposalAgent"]
