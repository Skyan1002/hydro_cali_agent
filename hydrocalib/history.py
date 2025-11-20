"""History recording utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CandidateRecord:
    candidate_index: int
    params: Dict[str, float]
    metrics: Dict[str, float]
    full_metrics: Dict[str, float]
    event_metrics: List[Dict[str, Any]]


@dataclass
class RoundRecord:
    round_index: int
    proposals: List[Dict[str, Any]]
    refined_candidates: List[Dict[str, Any]]
    candidates: List[CandidateRecord]
    best_candidate_index: int
    rationale: str = ""
    risk: str = ""
    focus: str = ""
    best_candidates_by_metric: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class HistoryStore:
    path: Path
    rounds: List[RoundRecord] = field(default_factory=list)
    best_metrics: Optional[Dict[str, Any]] = None
    best_by_metric: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def save(self) -> None:
        payload = {
            "rounds": [
                {
                    "round_index": r.round_index,
                    "proposals": r.proposals,
                    "refined_candidates": r.refined_candidates,
                    "candidates": [
                        {
                            "candidate_index": c.candidate_index,
                            "params": c.params,
                            "metrics": c.metrics,
                            "full_metrics": c.full_metrics,
                            "event_metrics": c.event_metrics,
                        }
                        for c in r.candidates
                    ],
                    "best_candidate_index": r.best_candidate_index,
                    "rationale": r.rationale,
                    "risk": r.risk,
                    "focus": r.focus,
                    "best_candidates_by_metric": r.best_candidates_by_metric,
                }
                for r in self.rounds
            ],
            "best": self.best_metrics,
            "best_by_metric": self.best_by_metric,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def update_best(self,
                    aggregate_metrics: Dict[str, Any],
                    full_metrics: Dict[str, Any],
                    params: Dict[str, float],
                    round_index: int,
                    candidate_index: int) -> bool:
        current_best = self.best_metrics.get("aggregate_metrics", {}).get("NSE", float("-inf")) if self.best_metrics else float("-inf")
        candidate_score = aggregate_metrics.get("NSE", float("-inf"))
        if candidate_score > current_best:
            self.best_metrics = {
                "round_index": round_index,
                "candidate_index": candidate_index,
                "metrics": aggregate_metrics,
                "aggregate_metrics": aggregate_metrics,
                "full_metrics": full_metrics,
                "params": params,
            }
            return True
        return False

    def update_best_metric(self,
                           key: str,
                           value: float,
                           aggregate_metrics: Dict[str, Any],
                           full_metrics: Dict[str, Any],
                           params: Dict[str, float],
                           round_index: int,
                           candidate_index: int) -> bool:
        if not math.isfinite(value):
            return False
        current = self.best_by_metric.get(key, {})
        current_value = current.get("value", float("-inf"))
        if value > current_value:
            self.best_by_metric[key] = {
                "round_index": round_index,
                "candidate_index": candidate_index,
                "value": value,
                "aggregate_metrics": aggregate_metrics,
                "full_metrics": full_metrics,
                "params": params,
            }
            return True
        return False
