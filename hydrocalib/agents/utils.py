"""Shared utilities for calibration agents."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from ..config import FROZEN_PARAMETERS
from ..parameters import ParameterSet, apply_step_guard

load_dotenv()
_client = OpenAI()
_JSON_RE = re.compile(r"\{[\s\S]*\}", re.M)


def get_client() -> OpenAI:
    return _client


def extract_json_block(text: str) -> Dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def redact_history_block(prompt_text: str) -> str:
    """Remove bulky history payloads from a JSON prompt before logging."""

    try:
        payload = json.loads(prompt_text)
    except Exception:
        return prompt_text

    if "history_payload" in payload:
        payload["history_payload"] = "[omitted]"
    return json.dumps(payload, ensure_ascii=False, indent=2)


def coerce_updates(params: ParameterSet, updates: Dict[str, Any]) -> ParameterSet:
    new_vals = params.values.copy()
    for name, spec in updates.items():
        if name not in new_vals or name in FROZEN_PARAMETERS:
            continue
        old = new_vals[name]
        if isinstance(spec, dict) and "op" in spec and "value" in spec:
            op = spec["op"]
            val = float(spec["value"])
            if op == "*":
                candidate = old * val
            elif op == "+":
                candidate = old + val
            elif op == "-":
                candidate = old - val
            elif op == "=":
                candidate = val
            else:
                candidate = old
        else:
            candidate = float(spec)
        new_vals[name] = apply_step_guard(name, old, candidate)
    return ParameterSet(new_vals)


def b64_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


__all__ = [
    "get_client",
    "extract_json_block",
    "coerce_updates",
    "b64_image",
    "redact_history_block",
]
