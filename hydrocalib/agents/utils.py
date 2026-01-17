"""Shared utilities for calibration agents."""

from __future__ import annotations

import base64
import json
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from ..config import FROZEN_PARAMETERS
from ..parameters import ParameterSet, apply_step_guard

load_dotenv()
_JSON_RE = re.compile(r"\{[\s\S]*\}", re.M)


class UnifiedClient:
    """A wrapper client that routes to OpenAI or Google GenAI (via OpenAI adapter) based on model name."""
    def __init__(self):
        self._openai = None
        self._google_openai = None
    
    @property
    def openai_client(self):
        if self._openai is None:
            self._openai = OpenAI()
        return self._openai

    @property
    def google_openai_client(self):
        if self._google_openai is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                # Let it fail downstream if key is missing, or print warning?
                # OpenAI client might raise error if key is None on init or on call.
                pass
            self._google_openai = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        return self._google_openai

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        # Check if it's a Gemini model
        if "gemini" in model.lower():
            # Use Google's OpenAI-compatible endpoint
            return self.google_openai_client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            # Use standard OpenAI endpoint
            return self.openai_client.chat.completions.create(model=model, messages=messages, **kwargs)


_client = UnifiedClient()


def get_client() -> UnifiedClient:
    return _client


def extract_json_block(text: str) -> Dict[str, Any]:
    # Gemini sometimes puts ```json ... ``` wrapper
    # The existing regex \{[\s\S]*\} matches the JSON block inside
    match = _JSON_RE.search(text)
    if not match:
        # Fallback: try to find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except:
                pass
        raise ValueError("No JSON object found in LLM response")
    
    # Try parsing the match
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
         # If strict regex failed (due to markdown backticks potentially interfering if inside match? 
         # actually regex is greedy internal so it should capture everything between { })
         pass
         
    # Logic to strip ```json if they are caught inside?
    # Actually the current regex takes outermost braces if greedy? 
    # re.compile(r"\{[\s\S]*\}", re.M) is greedy.
    # It grabs from first { to last }.
    
    block = match.group(0)
    return json.loads(block)


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

def load_skill(skill_name: str) -> str:
    """Load the content of a skill from the skills directory."""
    # Assume skills are located in ../../skills relative to this file
    # This file is in hydrocalib/agents/utils.py -> ../../skills is project_root/skills
    base_dir = Path(__file__).parent.parent.parent
    skill_path = base_dir / "skills" / skill_name / "SKILL.md"
    
    if not skill_path.exists():
        print(f"[WARN] Skill {skill_name} not found at {skill_path}")
        return ""
        
    try:
        content = skill_path.read_text(encoding="utf-8")
        # Strip frontmatter if present (between --- and ---)
        if content.startswith("---"):
            try:
                _, _, body = content.split("---", 2)
                return body.strip()
            except ValueError:
                return content
        return content
    except Exception as e:
        print(f"[WARN] Failed to load skill {skill_name}: {e}")
        return ""


__all__ = [
    "get_client",
    "extract_json_block",
    "coerce_updates",
    "b64_image",
    "redact_history_block",
    "load_skill",
]
