"""LLM image reader for flow diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .utils import b64_image, get_client
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


class ImageSummaryAgent:
    def __init__(self, model: str = LLM_MODEL_REASONING, detail_output: bool = False) -> None:
        self.model = model
        self.client = get_client()
        self.detail_output = detail_output

    def summarize(self,
                  image_path: str,
                  *,
                  round_label: str,
                  return_log: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        user_prompt = (
            f"Round: {round_label}\n"
            "Provide a concise diagnostic summary (5-10 bullet points) based on the montage."
        )
        messages = [
            {"role": "system", "content": IMAGE_READER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image(image_path)}"}},
                ],
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        summary = response.choices[0].message.content or ""
        log_payload = None
        if return_log:
            log_payload = {
                "stage": "image_reader",
                "round": round_label,
                "system_prompt": IMAGE_READER_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "input_files": [Path(image_path).name],
                "input_paths": [str(Path(image_path).resolve())],
                "output_text": summary,
            }
        return summary, log_payload


__all__ = ["ImageSummaryAgent"]
