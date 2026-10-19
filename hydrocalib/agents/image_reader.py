"""LLM image reader for flow diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import b64_image, call_with_validation, new_validation_counters
from ..config import LLM_MODEL_REASONING


IMAGE_READER_SYSTEM_PROMPT = (
    "You are a hydrologic analyst focused on diagnosing model behavior from plots."
    "\nYou will receive a 2x2 montage image with the following layout:"
    "\nTop-left: Flow Duration Curve (FDC) comparing simulated vs observed discharge."
    "\nTop-right: Event hydrograph #1 (observed vs simulated, linear/log panels)."
    "\nBottom-left: Event hydrograph #2 (observed vs simulated, linear/log panels)."
    "\nBottom-right: Event hydrograph #3 (observed vs simulated, linear/log panels)."
    "\nYour task is to describe the visible mismatches and biases that matter for parameter tuning."
    "\nFocus on: high-flow bias, low-flow bias/baseflow, timing/lag, peak magnitude, recession behavior,"
    " and variance across events."
    "\nDo NOT propose parameter updates yet—only provide diagnostic observations that would be useful"
    " for calibration decisions."
)


# Hybrid perception (arm H1, added 2026-09-27): the reader sees the same montage *and* the
# deterministic signatures computed from the same series, treats the numbers as exact and
# reports only what the plots add. The image-only prompt above is unchanged.
HYBRID_READER_SYSTEM_PROMPT = (
    IMAGE_READER_SYSTEM_PROMPT
    + "\nYou will also receive numerical diagnostic signatures computed exactly from the same simulated"
    " and observed series (water balance, flow-duration-curve segments, baseflow and recession, and per-event"
    " peak ratio, peak timing, volume and rising-limb metrics). Treat these numbers as exact."
    "\nUse the plots only to add information the numbers do not contain, for example the shape of rising limbs"
    " and recessions, secondary or missed peaks, where within an event the mismatch occurs, and how the errors"
    " differ between events. Do not restate the numbers."
    "\nIf a plot appears to contradict a number, say so explicitly and defer to the number."
)


def _non_empty_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty diagnostic summary")
    return text


class ImageSummaryAgent:
    def __init__(self,
                 model: str = LLM_MODEL_REASONING,
                 detail_output: bool = False,
                 provider: Optional[str] = None,
                 thinking_budget: Optional[int] = None,
                 mode: str = "image") -> None:
        if mode not in ("image", "hybrid"):
            raise ValueError("ImageSummaryAgent mode must be 'image' or 'hybrid'")
        self.mode = mode
        self.model = model
        self.provider = provider
        self.thinking_budget = thinking_budget
        self.detail_output = detail_output
        self.usage_log: list[Dict[str, Any]] = []
        self.validation_log: List[Dict[str, Any]] = []
        self.validation_counters: Dict[str, int] = new_validation_counters()
        self.system_prompt = HYBRID_READER_SYSTEM_PROMPT if mode == "hybrid" else IMAGE_READER_SYSTEM_PROMPT

    def summarize(self,
                  image_path: str,
                  *,
                  round_label: str,
                  return_log: bool = False,
                  log_round: Optional[str] = None,
                  signatures: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        if signatures:  # hybrid perception: the numbers travel with the image
            user_prompt = (
                f"Round: {round_label}\n"
                "Numerical diagnostic signatures of the simulation shown in the montage:\n"
                f"{signatures}\n\n"
                "Provide a concise summary (3-8 bullet points) of the additional diagnostic information"
                " visible in the montage."
            )
        else:
            user_prompt = (
                f"Round: {round_label}\n"
                "Provide a concise diagnostic summary (5-10 bullet points) based on the montage."
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(image_path)}"}},
                ],
            },
        ]
        response, _data, summary, usage, attempts = call_with_validation(
            messages,
            model=self.model,
            provider=self.provider,
            thinking_budget=self.thinking_budget,
            stage="image_reader",
            round_label=round_label,
            log_round=log_round,
            parse=_non_empty_text,
            expect_json=False,
            format_hint="Reply with the 5-10 bullet-point diagnostic summary as plain text.",
            usage_log=self.usage_log,
            validation_log=self.validation_log,
            counters=self.validation_counters,
        )
        log_payload = None
        if return_log:
            log_payload = {
                "stage": "image_reader",
                "round": round_label,
                "system_prompt": self.system_prompt,
                "user_prompt": user_prompt,
                "input_files": [Path(image_path).name],
                "input_paths": [str(Path(image_path).resolve())],
                "output_text": summary,
                "usage": usage,
                "validation_attempts": attempts,
            }
        return summary, log_payload


__all__ = ["ImageSummaryAgent", "IMAGE_READER_SYSTEM_PROMPT", "HYBRID_READER_SYSTEM_PROMPT"]
