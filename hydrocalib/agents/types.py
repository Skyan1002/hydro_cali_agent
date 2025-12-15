"""Shared data structures for agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RoundContext:
    round_index: int
    params: Dict[str, float]
    display_params: Dict[str, float]
    param_display_names: Dict[str, str]
    aggregate_metrics: Dict[str, float]
    full_metrics: Dict[str, float]
    event_metrics: List[Dict[str, Any]]
    history_summary: str
    description: str = ""
    images: List[str] = field(default_factory=list)
    physics_information: bool = True
    physics_prompt: str = ""
