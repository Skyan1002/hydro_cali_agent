"""First-stage proposal agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .types import RoundContext
from .utils import b64_image, coerce_updates, extract_json_block, get_client
from ..config import LLM_MODEL_DEFAULT
from ..parameters import ParameterSet


EF5_PARAMETER_GUIDE = (
    "EF5 Parameter Overview for Calibration\n"
    "WM controls the total soil water storage capacity; higher WM increases infiltration and reduces runoff. "
    "B defines the shape of the variable infiltration curve; larger B yields more surface runoff for a given soil moisture. "
    "IM is the impervious area fraction—higher values reduce infiltration and increase runoff. "
    "KE scales potential evapotranspiration (PET); larger KE increases evaporation and decreases runoff. "
    "FC is the saturated hydraulic conductivity; higher FC allows faster infiltration, reducing surface flow. "
    "IWU sets the initial soil moisture; too high a value can exaggerate early runoff. "
    "TH determines the drainage threshold for channel initiation; a larger TH produces fewer, coarser channels. "
    "UNDER controls interflow velocity—higher values accelerate subsurface flow. "
    "LEAKI defines the leakage rate from the interflow layer; higher LEAKI speeds lateral drainage. "
    "ISU is the initial interflow storage; nonzero values may create unrealistic early peaks. "
    "ALPHA and BETA are routing parameters in the discharge equation Q = αA^β; increasing either slows wave propagation and broadens flood peaks. "
    "ALPHA0 applies the same relationship for non-channel cells. "
    "Together, these parameters govern infiltration, storage, and routing. During calibration, adjust WM, B, IM, and FC to shape runoff volume; "
    "tune KE for evapotranspiration balance; and modify ALPHA, BETA, UNDER, and LEAKI to match hydrograph timing and attenuation."
)


PROPOSAL_SYSTEM_PROMPT = (
    "You are a hydrologic calibration strategist."
    "\nGiven current metrics, history, and images, propose diverse parameter update strategies."
    "\nReturn STRICT JSON with a `candidates` list; each candidate needs an `id`, `goal` (short description) and `updates` mapping"
    " parameter names to either numbers or {\"op\": \"*|+|-|=\", \"value\": number} for multiplicative/additive adjustments."
    "\nMake every candidate explore a clearly different direction within the allowed parameter bounds; large steps are permitted."
    "\nDo NOT modify TH, IWU, or ISU—those parameters remain fixed."
    "\n" + EF5_PARAMETER_GUIDE
)


class ProposalAgent:
    def __init__(self,
                 model: str = LLM_MODEL_DEFAULT):
        self.model = model
        self.client = get_client()

    def build_prompt(self, context: RoundContext, k: int) -> str:
        payload = {
            "round": context.round_index,
            "current_params": context.params,
            "aggregate_metrics": context.aggregate_metrics,
            "full_metrics": context.full_metrics,
            "event_metrics": context.event_metrics,
            "history_summary": context.history_summary,
            "requested_candidates": k,
            "notes": context.description,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def propose(self, context: RoundContext, k: int) -> List[Dict[str, Any]]:
        user_prompt = self.build_prompt(context, k)
        if context.images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(img)}"}} for img in context.images)
            messages = [
                {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        else:
            messages = [
                {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        raw = response.choices[0].message.content
        data = extract_json_block(raw)
        candidates = data.get("candidates", [])[:k]
        return candidates

    def apply_candidates(self, base_params: ParameterSet, candidates: List[Dict[str, Any]]) -> List[ParameterSet]:
        param_sets: List[ParameterSet] = []
        for cand in candidates:
            updates = cand.get("updates", {})
            new_params = coerce_updates(base_params, updates)
            param_sets.append(new_params)
        return param_sets


__all__ = ["ProposalAgent"]
