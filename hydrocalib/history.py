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
                    candidate_index: int,
                    objective_key: str = "NSE") -> bool:
        current_best = self.best_metrics.get("full_metrics", {}).get(objective_key, float("-inf")) if self.best_metrics else float("-inf")
        candidate_score = full_metrics.get(objective_key, float("-inf"))
        if candidate_score > current_best:
            self.best_metrics = {
                "round_index": round_index,
                "candidate_index": candidate_index,
                "metrics": aggregate_metrics,
                "aggregate_metrics": aggregate_metrics,
                "full_metrics": full_metrics,
                "params": params,
                "objective": objective_key,
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

    def load(self) -> bool:
        """Rebuild the store from calibration_history.json so a run can be resumed.

        Only the fields ``save`` writes are read back; ``timing``/``progress``/
        ``failures`` are the manager's and it restores them itself. Returns False
        when there is nothing usable on disk, so the caller can fall back to a
        fresh baseline instead of half-restoring.
        """
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return False
        rounds = []
        for r in payload.get("rounds") or []:
            cands = [CandidateRecord(candidate_index=int(c.get("candidate_index", 0)),
                                     params=dict(c.get("params") or {}),
                                     metrics=dict(c.get("metrics") or {}),
                                     full_metrics=dict(c.get("full_metrics") or {}),
                                     event_metrics=list(c.get("event_metrics") or []))
                     for c in (r.get("candidates") or [])]
            rounds.append(RoundRecord(round_index=int(r.get("round_index", len(rounds) + 1)),
                                      proposals=list(r.get("proposals") or []),
                                      refined_candidates=list(r.get("refined_candidates") or []),
                                      candidates=cands,
                                      best_candidate_index=int(r.get("best_candidate_index", 0)),
                                      rationale=r.get("rationale", ""),
                                      risk=r.get("risk", ""),
                                      focus=r.get("focus", ""),
                                      best_candidates_by_metric=dict(r.get("best_candidates_by_metric") or {})))
        best = payload.get("best")
        if not best or not best.get("params"):
            return False
        self.rounds = rounds
        self.best_metrics = best
        self.best_by_metric = dict(payload.get("best_by_metric") or {})
        return True
