"""Two-stage calibration manager orchestrating proposal/evaluation agents."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (DEFAULT_GAUGE_NUM, DEFAULT_PEAK_PICK_KWARGS, DEFAULT_SIM_FOLDER,
                       EVENTS_FOR_AGGREGATE, FROZEN_PARAMETERS, MAX_STEPS_DEFAULT, PARAM_BOUNDS)
from ..history import CandidateRecord, HistoryStore, RoundRecord
from ..signatures import compute_signature_summary
from ..metrics import (aggregate_event_metrics, compute_event_metrics,
                       read_metrics_for_period, read_metrics_from_csv)
from ..parameters import ParameterSet
from .physics_info import (
    bounds_hint as bounds_hint_of,
    build_display_name_map,
    build_zone_note,
    display_parameters,
    invert_display_map,
    render_parameter_guide,
    anon_guide_on as _anon_guide_on,
    guide_mode as _guide_mode,
)
from ..zoned import base_name as _base, expand_base, zone_of
from ..peak_events import pick_peak_events
from ..plotting import (
    plot_event_windows,
    plot_flow_duration_curve,
    plot_hydrograph_with_precipitation,
    plot_image_montage,
)
from ..simulation import SimulationResult, SimulationRunner, run_simulations_parallel
from .evaluation import EvaluationAgent
from .image_reader import ImageSummaryAgent
from .proposal import ProposalAgent
from .types import RoundContext
from .utils import VALIDATION_COUNTER_KEYS, CandidateRejected

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASINAREA_RE = re.compile(r"^\s*BASINAREA\s*=\s*([0-9.eE+-]+)", re.MULTILINE)
WORKFLOWS = ("full", "direct", "direct_matched")


class _Terminated(RuntimeError):
    """Raised inside the loop when SIGTERM arrives so ``finally`` blocks run."""


@dataclass
class CandidateOutcome:
    simulation: SimulationResult
    params: ParameterSet
    windows: List[Tuple]
    event_metrics: List[Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    full_metrics: Dict[str, float]
    hydrograph_path: Optional[str] = None
    fdc_path: Optional[str] = None
    montage_path: Optional[str] = None
    event_figures: List[str] = field(default_factory=list)


@dataclass
class TestConfig:
    enabled: bool
    warmup_begin: str
    warmup_end: str
    warmup_state: str
    time_begin: str
    time_end: str
    timestep: str
    eval_start: str
    eval_end: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + ``os.replace`` (never a torn file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class TwoStageCalibrationManager:
    def __init__(self,
                 args_obj,
                 simu_folder: str = DEFAULT_SIM_FOLDER,
                 gauge_num: str = DEFAULT_GAUGE_NUM,
                 n_candidates: int = 8,
                 n_peaks: int = EVENTS_FOR_AGGREGATE,
                 include_max_event_images: int = 3,
                 peak_pick_kwargs: Optional[Dict] = None,
                 history_path: Optional[str] = None,
                 memory_cutoff: Optional[int] = None,
                 max_workers: Optional[int] = None,
                 test_config: Optional[TestConfig] = None,
                 objective: str = "nse",
                 physics_information: bool = True,
                 image_input: bool = True,
                 image_type: str = "fdc",
                 failure_patient: int = 5,
                 detail_output: bool = False,
                 llm_model: Optional[str] = None,
                 llm_provider: Optional[str] = None,
                 workflow: str = "full",
                 perception: str = "image",
                 thinking_budget: Optional[int] = None,
                 keep_legacy_failure_requery: bool = False,
                 zone_ids: Optional[Sequence[str]] = None,
                 zone_areas: Optional[Dict[str, float]] = None,
                 multisite_gauges: Optional[Sequence[str]] = None,
                 gauge_areas_km2: Optional[Dict[str, float]] = None,
                 zone_wildcard: bool = True,
                 direct_keep_failures: bool = False,
                 resume: bool = False,
                 bounds_hint: bool = False,
                 zone_assembly: bool = False):
        self.args_obj = args_obj
        # per-zone assembly of an extra candidate each round (see _assemble_by_zone)
        self.zone_assembly = bool(zone_assembly)
        # tell the agent the parameter box and which values are on it (see
        # physics_info.bounds_hint); off by default so earlier arms are unchanged
        self.bounds_hint = bool(bounds_hint)
        # Multi-site: every gauge listed here has its own record; the objective
        # 'multisite_nse' is their mean NSE and the perception stage carries one
        # signature block per gauge. zone_wildcard=False removes the '<param>@*'
        # shorthand so the agent has to address every zone individually.
        self.multisite_gauges: List[str] = [str(g) for g in (multisite_gauges or [])]
        self.gauge_areas_km2: Dict[str, float] = dict(gauge_areas_km2 or {})
        self.zone_wildcard = bool(zone_wildcard)
        self.resume = bool(resume)
        self._resumed_segment = 0
        # Zoned (multi-gauge) run: the same 13 parameters exist once per zone, so the
        # starting point is the single-gauge one replicated into every zone and every
        # name carries a '@<zone>' suffix. Everything downstream keys off base_name().
        self.zone_ids: List[str] = list(zone_ids or [])
        self.zone_areas: Dict[str, float] = dict(zone_areas or {})
        self.current_params = ParameterSet.from_object(args_obj)
        if self.zone_ids:
            self.current_params = ParameterSet(expand_base(self.current_params.values, self.zone_ids))
        self.runner = SimulationRunner(simu_folder=simu_folder, gauge_num=gauge_num)
        self.display_name_map = build_display_name_map(
            physics_information, names=list(self.current_params.values) if self.zone_ids else None)
        self.reverse_display_map = invert_display_map(self.display_name_map)
        zone_note = (build_zone_note(self.zone_ids, self.zone_areas,
                                     allow_wildcard=self.zone_wildcard,
                                     observed=bool(self.multisite_gauges))
                     if self.zone_ids else "")
        self.physics_prompt = render_parameter_guide(physics_information, self.display_name_map, zone_note)
        if workflow not in WORKFLOWS:
            raise ValueError(f"workflow must be one of {WORKFLOWS}")
        if perception not in ("image", "numeric", "none"):
            raise ValueError("perception must be 'image', 'numeric' or 'none'")
        self.workflow = workflow
        self.perception = perception
        self.thinking_budget = thinking_budget
        self.keep_legacy_failure_requery = keep_legacy_failure_requery
        model_kw = {"model": llm_model} if llm_model else {}
        self.proposal_agent = ProposalAgent(
            physics_information=physics_information,
            display_name_map=self.display_name_map,
            detail_output=detail_output,
            provider=llm_provider,
            thinking_budget=thinking_budget,
            direct_matched=(workflow == "direct_matched"),
            perception=perception,
            allow_wildcard=self.zone_wildcard,
            **model_kw,
        )
        self.evaluation_agent = EvaluationAgent(
            physics_information=physics_information,
            display_name_map=self.display_name_map,
            detail_output=detail_output,
            provider=llm_provider,
            thinking_budget=thinking_budget,
            allow_wildcard=self.zone_wildcard,
            **model_kw,
        )
        self.image_reader = ImageSummaryAgent(
            detail_output=detail_output, provider=llm_provider,
            thinking_budget=thinking_budget, **model_kw
        )
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.n_candidates = n_candidates
        self.n_peaks = n_peaks
        self.include_max_event_images = include_max_event_images
        self.peak_pick_kwargs = peak_pick_kwargs or DEFAULT_PEAK_PICK_KWARGS
        hist_path = history_path or (Path(simu_folder) / "results" / "calibration_history.json")
        self.history = HistoryStore(Path(hist_path))
        if memory_cutoff is not None and memory_cutoff < 0:
            raise ValueError("memory_cutoff must be non-negative (>= 0)")
        self.memory_cutoff = memory_cutoff
        self.direct_keep_failures = bool(direct_keep_failures)
        if failure_patient < 1:
            raise ValueError("failure_patient must be at least 1")
        self.best_outcome: Optional[CandidateOutcome] = None
        self.round_index = 0
        self.round_label = "round000_000"
        self.stall = 0
        self.max_workers = max_workers
        self.test_config = test_config
        self.objective = objective
        self.physics_information = physics_information
        self.image_input = image_input
        self.image_type = image_type
        self.failure_patient = failure_patient
        self.detail_output = detail_output
        self.detail_dir = Path(simu_folder) / "results" / "detail_logs"
        self._detail_written: set = set()
        self._legacy_requery_active = False
        self.update_round = 1
        self.failure_round = 0
        self.best_objective_value = float("-inf")
        self.failure_history: List[Dict[str, Any]] = []
        # --- accounting (R2-8d / R2-9) ---
        self.basin_area_km2: Optional[float] = self._basin_area_from_control()
        self.timing_rounds: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []
        self._t_start = time.perf_counter()
        self.max_rounds_requested: Optional[int] = None
        self.stop_reason: Optional[str] = None
        self.completed = False

    # ------------------------------------------------------------------ setup
    def _basin_area_from_control(self) -> Optional[float]:
        """BASINAREA= from the base control.txt (needed for the runoff coefficient)."""
        try:
            m = BASINAREA_RE.search(self.runner._control_template)
            if not m:
                return None
            val = float(m.group(1))
            return val if np.isfinite(val) and val > 0 else None
        except Exception:
            return None

    def initialize_baseline(self) -> None:
        print("[Init] Running baseline simulation…")
        t0 = time.perf_counter()
        baseline_result = self.runner.run(self.current_params, round_label="round000_000", candidate_index=0)
        self.timing_rounds.append({
            "round_label": "round000_000", "update_round": 0, "failure_round": 0,
            "n_sims": 1, "ef5_seconds": round(time.perf_counter() - t0, 3),
            "n_test_sims": 0, "test_seconds": 0.0, "accepted": True, "kind": "baseline",
        })
        outcome = self._process_result(baseline_result)
        self._ensure_plots(outcome)
        self.best_outcome = outcome
        self.best_objective_value = self._objective_value(outcome.aggregate_metrics, outcome.full_metrics)
        self.history.update_best(
            aggregate_metrics=outcome.aggregate_metrics,
            full_metrics=outcome.full_metrics,
            params=outcome.params.values.copy(),
            round_index=0,
            candidate_index=0,
            objective_key=self.objective.upper(),
        )
        self._update_metric_bests([outcome], round_index=0)
        self.history.save()
        full = outcome.full_metrics
        print(
            "[Init] Baseline metrics "
            f"NSE={full.get('NSE', float('nan')):.3f} "
            f"CC={full.get('CC', float('nan')):.3f} "
            f"KGE={full.get('KGE', float('nan')):.3f}"
        )

    def _process_result(self, result: SimulationResult) -> CandidateOutcome:
        windows = pick_peak_events(result.csv_path, n=self.n_peaks, **self.peak_pick_kwargs)
        event_metrics = compute_event_metrics(result.csv_path, windows)
        aggregate = dict(aggregate_event_metrics(event_metrics, top_n=self.n_peaks))
        full_metrics = dict(read_metrics_from_csv(result.csv_path))
        if self.multisite_gauges:
            # one NSE per gauge for the prompt and the history, and their mean as the
            # objective under the key _objective_value looks up (self.objective.upper()).
            # Any gauge that fails to score makes the objective -inf: a missing series
            # must not raise the mean of the rest.
            per = self._gauge_metrics(Path(result.output_dir))
            aggregate.update({f"NSE@{g}": v for g, v in per.items()})
            vals = [per.get(g, float("nan")) for g in self.multisite_gauges]
            full_metrics["MULTISITE_NSE"] = (sum(vals) / len(vals)
                                             if vals and all(math.isfinite(v) for v in vals)
                                             else float("-inf"))
        return CandidateOutcome(result, result.params, windows, event_metrics, aggregate, full_metrics)

    def _gauge_metrics(self, out_dir: Path) -> Dict[str, float]:
        """NSE at every multi-site gauge from its own ts.<id>.crest.csv (EF5 lower-cases the id)."""
        vals: Dict[str, float] = {}
        for g in self.multisite_gauges:
            path = out_dir / f"ts.{g.lower()}.crest.csv"
            try:
                vals[g] = float(read_metrics_from_csv(str(path)).get("NSE", float("nan"))) \
                    if path.exists() else float("nan")
            except Exception:  # noqa: BLE001
                vals[g] = float("nan")
        return vals

    def _resume_from_disk(self) -> bool:
        """Continue a run from its own calibration_history.json.

        Restores the accepted rounds, the incumbent, the round counters, the forward-
        model timing and the failure memory, rotates the previous cost ledger aside
        (usage_ledger.seg<N>.json -- a resumed segment starts its own ledger, so
        the total is the sum over segments), then re-simulates the incumbent once to
        rebuild the outcome the next proposal is built from. Returns False if there
        is nothing usable, in which case the caller runs a fresh baseline.
        """
        if not self.history.load():
            print("[Resume] no usable calibration_history.json; starting fresh")
            return False
        best = self.history.best_metrics or {}
        try:
            payload = json.loads(Path(self.history.path).read_text())
        except (OSError, ValueError):
            payload = {}
        led = Path(self.history.path).parent / "usage_ledger.json"
        seg = 1
        while led.with_name(f"usage_ledger.seg{seg}.json").exists():
            seg += 1
        if led.exists():
            os.replace(led, led.with_name(f"usage_ledger.seg{seg}.json"))
        self._resumed_segment = seg
        self.timing_rounds = list((payload.get("timing") or {}).get("rounds") or [])
        self.failure_history = list(payload.get("failures") or [])
        prog = payload.get("progress") or {}
        self.update_round = int(prog.get("update_round") or (len(self.history.rounds) + 1))
        self.failure_round = int(prog.get("failure_round") or 0)
        self.current_params = ParameterSet(dict(best["params"]))
        if not self.zone_ids:
            self.current_params.to_object(self.args_obj)
        label = f"resume{seg:02d}_000"
        t0 = time.perf_counter()
        res = self.runner.run(self.current_params, round_label=label, candidate_index=0)
        outcome = self._process_result(res)
        self.best_outcome = outcome
        self.best_objective_value = self._objective_value(outcome.aggregate_metrics, outcome.full_metrics)
        self.timing_rounds.append({
            "round_label": label, "update_round": self.update_round, "failure_round": self.failure_round,
            "n_sims": 1, "ef5_seconds": round(time.perf_counter() - t0, 3),
            "n_test_sims": 0, "test_seconds": 0.0, "accepted": True, "kind": "resume",
        })
        print(f"[Resume] segment {seg}: {len(self.history.rounds)} accepted rounds on disk, "
              f"continuing at update_round={self.update_round} failure_round={self.failure_round}, "
              f"incumbent {self.objective}={self.best_objective_value:.3f}")
        return True

    def _run_test_suite(self,
                        params: Sequence[ParameterSet],
                        round_label: str,
                        timing: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, Any]]:
        if not self.test_config or not self.test_config.enabled:
            return {}

        print(f"[Round {round_label}] Starting test suite for {len(params)} candidates…")
        t0 = time.perf_counter()
        overrides = {
            "TIME_STATE": self.test_config.warmup_state,
            "WARMUP_TIME_BEGIN": self.test_config.warmup_begin,
            "WARMUP_TIME_END": self.test_config.warmup_end,
            "TIME_BEGIN": self.test_config.time_begin,
            "TIME_END": self.test_config.time_end,
            "TIMESTEP": self.test_config.timestep,
        }

        summaries: Dict[int, Dict[str, Any]] = {}
        for idx, param_set in enumerate(params):
            result = self.runner.run_with_overrides(
                param_set,
                round_label=round_label,
                candidate_index=idx,
                subfolder="test",
                states_dir=None,
                scalar_overrides=overrides,
            )
            metrics = read_metrics_for_period(
                result.csv_path,
                start=self.test_config.eval_start,
                end=self.test_config.eval_end,
            )
            summary = {
                "round_label": round_label,
                "candidate_index": idx,
                "params": param_set.values.copy(),
                "csv": result.csv_path,
                "metrics": metrics,
            }
            if self.multisite_gauges:
                # hold-out at every gauge, so the multi-site solution can be judged out
                # of sample where it was fitted, not only at the outlet
                per: Dict[str, float] = {}
                for g in self.multisite_gauges:
                    p = Path(result.output_dir) / f"ts.{g.lower()}.crest.csv"
                    try:
                        per[g] = float(read_metrics_for_period(
                            str(p), start=self.test_config.eval_start,
                            end=self.test_config.eval_end).get("NSE", float("nan"))) if p.exists() else float("nan")
                    except Exception:  # noqa: BLE001
                        per[g] = float("nan")
                finite = [v for v in per.values() if math.isfinite(v)]
                summary["metrics_by_gauge"] = per
                summary["metrics"]["MULTISITE_NSE"] = (sum(finite) / len(finite)
                                                       if len(finite) == len(self.multisite_gauges)
                                                       else float("nan"))
            summaries[idx] = summary
            summary_path = Path(result.output_dir) / "summary.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

            print(
                f"    [Round {round_label} Test {idx}] NSE={metrics.get('NSE', float('nan')):.3f} "
                f"CC={metrics.get('CC', float('nan')):.3f} KGE={metrics.get('KGE', float('nan')):.3f}"
            )

        if timing is not None:
            timing["n_test_sims"] = len(params)
            timing["test_seconds"] = round(time.perf_counter() - t0, 3)
        print(f"[Round {round_label}] Test suite finished.")
        return summaries

    # -------------------------------------------------------------- LLM stages
    def _request_candidates(self, context: RoundContext, round_label: str):
        print(f"[Round {round_label}] Requesting {self.n_candidates} proposals from proposal agent…")
        proposal_log = None
        if self.workflow == "direct_matched":
            # Information-matched single-call baseline (R2-3): the one call gets the
            # perception diagnostic instructions + the verification review
            # instructions + the proposal contract in its system prompt, the montage
            # (attached by _build_context) and the same history_payload the
            # verification agent receives — i.e. the same information budget as
            # full INCA. What is removed is only the stage split, the independent
            # verification call and the failure memory.
            proposals_result = self.proposal_agent.propose(
                context, self.n_candidates, return_log=self.detail_output,
                history_payload=self._history_payload(), log_round=round_label,
            )
        else:
            proposals_result = self.proposal_agent.propose(
                context, self.n_candidates, return_log=self.detail_output, log_round=round_label,
            )
        if self.detail_output:
            proposals, proposal_log = proposals_result
        else:
            proposals = proposals_result
        proposal_params = self.proposal_agent.apply_candidates(self.best_outcome.params, proposals)
        print(f"[Round {round_label}] Initial proposals and parameter sets:")
        for idx, (proposal, params) in enumerate(zip(proposals, proposal_params)):
            goal = proposal.get("goal") or proposal.get("id") or f"cand_{idx}"
            print(
                f"    [Proposal {idx}] goal={goal} updates={proposal.get('updates', {})} "
                f"→ params={params.values}"
            )

        if self.workflow == "direct":
            # Naive direct baseline: one LLM call per round with the proposal prompt
            # only. It lacks the perception diagnostic instructions, the review
            # instructions (incl. the warm-up rule) and the history_payload, in
            # addition to lacking the verification stage and failure memory — so it
            # removes prompt content as well as architecture. Kept as the lower rung;
            # use direct_matched for the architecture-only ablation.
            if proposal_log:
                self._log_detail("proposal", round_label, proposal_log)
            return proposals, proposals, proposal_params, {"rationale": "", "risk": "", "focus": ""}
        if self.workflow == "direct_matched":
            if proposal_log:
                self._log_detail("proposal", round_label, proposal_log)
            return proposals, proposals, proposal_params, {"rationale": "", "risk": "", "focus": ""}

        refined_candidates, eval_meta, eval_log = self.evaluation_agent.refine(
            context,
            proposals,
            self._history_payload(),
            self.n_candidates,
            return_log=self.detail_output,
            log_round=round_label,
        )
        refined_params = self.evaluation_agent.apply_candidates(self.best_outcome.params, refined_candidates)
        if not refined_params:
            refined_candidates = proposals
            refined_params = proposal_params
        print(f"[Round {round_label}] Evaluation agent refined parameter sets:")
        for idx, (candidate, params) in enumerate(zip(refined_candidates, refined_params)):
            goal = candidate.get("goal") or candidate.get("id") or f"cand_{idx}"
            print(
                f"    [Refined {idx}] goal={goal} updates={candidate.get('updates', {})} "
                f"→ params={params.values}"
            )
        if proposal_log:
            self._log_detail("proposal", round_label, proposal_log)
        if eval_log:
            self._log_detail("evaluation", round_label, eval_log)
        return proposals, refined_candidates, refined_params, eval_meta

    def _ensure_plots(self, outcome: CandidateOutcome) -> None:
        if self.perception != "image":
            # nothing downstream reads the figures in the numeric / none arms; they only
            # cost matplotlib time and disk on every accepted round
            return
        hydrograph_missing = not outcome.hydrograph_path or not Path(outcome.hydrograph_path).exists()
        if hydrograph_missing:
            outcome.hydrograph_path = plot_hydrograph_with_precipitation(outcome.simulation.csv_path, show=False)

        fdc_missing = not outcome.fdc_path or not Path(outcome.fdc_path).exists()
        if fdc_missing:
            outcome.fdc_path = plot_flow_duration_curve(outcome.simulation.csv_path, show=False)

        valid_figures = [fig for fig in outcome.event_figures if Path(fig).exists()]
        if len(valid_figures) < self.include_max_event_images:
            peaks_dir = Path(outcome.simulation.output_dir) / "peaks"
            valid_figures = plot_event_windows(
                outcome.simulation.csv_path,
                outcome.windows,
                out_dir=str(peaks_dir),
            )[:self.include_max_event_images]
        outcome.event_figures = valid_figures

    def _ensure_montage(self, outcome: CandidateOutcome) -> Optional[str]:
        image_paths: List[Optional[str]] = [outcome.fdc_path]
        image_paths.extend(outcome.event_figures[:3])
        while len(image_paths) < 4:
            image_paths.append(None)
        montage_dir = Path(outcome.simulation.output_dir)
        montage_path = montage_dir / "image_montage.png"
        outcome.montage_path = plot_image_montage(
            image_paths,
            str(montage_path),
            titles=["FDC", "Event 1", "Event 2", "Event 3"],
        )
        return outcome.montage_path

    def _publish_best(self,
                      outcome: CandidateOutcome,
                      subfolder: str,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        best_dir = Path(self.runner.simu_folder) / "results" / subfolder
        if best_dir.exists():
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        if outcome.hydrograph_path:
            shutil.copy2(outcome.hydrograph_path, best_dir / Path(outcome.hydrograph_path).name)
        if outcome.fdc_path:
            shutil.copy2(outcome.fdc_path, best_dir / Path(outcome.fdc_path).name)
        if outcome.montage_path:
            shutil.copy2(outcome.montage_path, best_dir / Path(outcome.montage_path).name)
        events_dir = best_dir / "events"
        events_dir.mkdir(exist_ok=True)
        for fig in outcome.event_figures:
            fig_path = Path(fig)
            if fig_path.exists():
                shutil.copy2(fig_path, events_dir / fig_path.name)
        summary: Dict[str, Any] = {
            "round_index": outcome.simulation.round_label,
            "candidate_index": outcome.simulation.candidate_index,
            "aggregate_metrics": outcome.aggregate_metrics,
            "full_metrics": outcome.full_metrics,
            "params": outcome.params.values.copy(),
        }
        if metadata:
            summary.update(metadata)
        (best_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    def _summarize_outcome(self,
                           outcome: CandidateOutcome,
                           *,
                           criterion: str,
                           value: float) -> Dict[str, Any]:
        return {
            "round_index": outcome.simulation.round_label,
            "candidate_index": outcome.simulation.candidate_index,
            "criterion": criterion,
            "criterion_value": value,
            "aggregate_metrics": outcome.aggregate_metrics,
            "full_metrics": outcome.full_metrics,
            "params": outcome.params.values.copy(),
            "hydrograph": outcome.hydrograph_path,
            "flow_duration_curve": outcome.fdc_path,
            "image_montage": outcome.montage_path,
            "event_figures": outcome.event_figures[: self.include_max_event_images],
        }

    def _history_limit(self, default: int) -> int:
        """Return the configured memory_cutoff when set, otherwise the provided default."""
        return self.memory_cutoff if self.memory_cutoff is not None else default

    def _apply_history_limit(self, items: List[Any], default: Optional[int] = None) -> List[Any]:
        """Trim a sequence of history entries using the configured cutoff or the provided default."""
        effective_default = default if default is not None else len(items)
        limit = self._history_limit(effective_default)
        if limit == 0:
            return []
        return items[-limit:]

    def _history_summary(self, last_k: int = 3) -> str:
        tail = self._apply_history_limit(self.history.rounds, last_k)
        if not tail:
            return "No prior rounds."
        parts = []
        for round_record in tail:
            best = next((c for c in round_record.candidates if c.candidate_index == round_record.best_candidate_index), None)
            if not best:
                continue
            metrics = best.metrics
            parts.append(
                f"r{round_record.round_index}: NSE={metrics.get('NSE', float('nan')):.3f} "
                f"CC={metrics.get('CC', float('nan')):.3f} "
                f"KGE={metrics.get('KGE', float('nan')):.3f}"
            )
        return " | ".join(parts) if parts else "No prior rounds."

    def _format_round_label(self, update_round: int, failure_round: int) -> str:
        return f"round{update_round:03d}_{failure_round:03d}"

    def _format_failure_summary(self, max_items: int = 5) -> tuple[str, List[Dict[str, Any]]]:
        if not self.failure_history:
            return "", []
        tail = self.failure_history[-max_items:]
        summary_lines = []
        for item in tail:
            summary_lines.append(
                f"{item['round_label']}: best objective={item['objective_value']:.3f} "
                f"(cand {item['candidate_index']})"
            )
        summary = "Recent failed attempts: " + " | ".join(summary_lines)

        # When image_input is off, filter out event-related metrics from failure details
        if not self.image_input:
            filtered_tail = []
            for item in tail:
                filtered_item = item.copy()
                if "aggregate_metrics" in filtered_item:
                    filtered_item["aggregate_metrics"] = self._filter_metrics_for_no_image(filtered_item["aggregate_metrics"])
                if "full_metrics" in filtered_item:
                    filtered_item["full_metrics"] = self._filter_metrics_for_no_image(filtered_item["full_metrics"])
                filtered_tail.append(filtered_item)
            return summary, filtered_tail

        return summary, tail

    def _build_image_summary(self, outcome: CandidateOutcome, log_label: Optional[str] = None) -> str:
        """Produce the perception summary handed to the proposal/evaluation agents.

        Three modes share one text channel, so the *information*, not the plumbing,
        is what differs between arms:

        * ``image``   — vision LLM reads the 2x2 montage (submitted behaviour)
        * ``numeric`` — deterministic hydrologic signatures from the same series
        * ``none``    — nothing (the w/o-image ablation)

        ``log_label`` only chooses the detail_logs directory (the request made
        for the *next* round is filed under that round); the prompt text still
        carries ``self.round_label`` exactly as before.
        """
        if self.perception == "none":
            return ""
        if self.perception == "numeric":
            # Signatures are a deterministic computation, not an LLM stage, so
            # they are still available to Direct without breaking its one-call budget.
            return self._build_numeric_summary(outcome, log_label=log_label)
        if self.workflow in ("direct", "direct_matched"):
            # The montage is attached directly to the single proposal call instead.
            return ""
        if not self.image_input or self.image_type in ("noimage", "noboth"):
            return ""
        self._ensure_plots(outcome)
        montage_path = self._ensure_montage(outcome)
        if not montage_path:
            return ""
        summary, reader_log = self.image_reader.summarize(
            montage_path,
            round_label=self.round_label,
            return_log=self.detail_output,
            log_round=log_label,
        )
        if reader_log:
            self._log_detail("image_reader", log_label or self.round_label, reader_log)
        return summary

    def _build_numeric_summary(self, outcome: CandidateOutcome, log_label: Optional[str] = None) -> str:
        """Numeric analogue of the montage: same series, same events, no image.

        Multi-site: one signature block per gauge, each headed by the gauge id, its
        zone's share of the basin and its NSE, so the agent can see *where* the
        error is -- the information the outlet-only setting never contained. The
        event windows are the outlet's (same storms at every gauge).
        """
        windows = [{"start": w[0], "end": w[1]} for w in outcome.windows[: self.n_peaks]]
        if not self.multisite_gauges:
            try:
                report, raw = compute_signature_summary(
                    outcome.simulation.csv_path, windows, area_km2=self.basin_area_km2
                )
            except Exception as exc:  # never let diagnostics kill a run
                print(f"[Warn] numeric signature computation failed: {type(exc).__name__}: {exc}")
                return ""
            inputs = [Path(outcome.simulation.csv_path)]
        else:
            out_dir = Path(outcome.simulation.output_dir)
            parts: List[str] = []
            raw: Dict[str, Any] = {}
            inputs = []
            for g in self.multisite_gauges:
                csv = out_dir / f"ts.{g.lower()}.crest.csv"
                share = self.zone_areas.get(g)
                area = self.gauge_areas_km2.get(g) or (self.basin_area_km2 if g == self.runner.gauge_num else None)
                nse = outcome.aggregate_metrics.get(f"NSE@{g}", float("nan"))
                head = (f"=== GAUGE {g} | zone share {share:.1f}% of basin | drainage {area:.0f} km2 | "
                        f"NSE={nse:.3f} ===" if share is not None and area else
                        f"=== GAUGE {g} | NSE={nse:.3f} ===")
                if not csv.exists():
                    parts.append(f"{head}\n(series missing)")
                    continue
                try:
                    rep, rw = compute_signature_summary(str(csv), windows, area_km2=area)
                except Exception as exc:  # noqa: BLE001
                    parts.append(f"{head}\n(signatures failed: {type(exc).__name__})")
                    continue
                parts.append(f"{head}\n{rep}")
                raw[g] = rw
                inputs.append(csv)
            report = "\n\n".join(parts)
        if self.detail_output:
            self._log_detail("signatures", log_label or self.round_label, {
                "stage": "signatures",
                "round": self.round_label,
                "system_prompt": "",
                "user_prompt": "",
                "input_files": [p.name for p in inputs],
                "input_paths": [str(p.resolve()) for p in inputs],
                "output_text": report,
                "parsed_output": raw,
            })
        return report

    # ------------------------------------------------------------ accounting
    def _all_agents(self):
        return (self.image_reader, self.proposal_agent, self.evaluation_agent)

    def provenance(self) -> Dict[str, Any]:
        """Fingerprint of code, prompts and settings that produced this run."""
        version = None
        try:
            vpath = PACKAGE_ROOT / "VERSION"
            if vpath.exists():
                version = vpath.read_text().strip()
        except Exception:
            version = None
        git_head = None
        try:
            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(PACKAGE_ROOT),
                capture_output=True, text=True, timeout=10,
            ).stdout.strip() or None
        except Exception:
            git_head = None
        return {
            "version": version,
            "git_head": git_head,
            "system_prompt_sha256": {
                "image_reader": _sha256(self.image_reader.system_prompt),
                "proposal": _sha256(self.proposal_agent.system_prompt),
                "evaluation": _sha256(self.evaluation_agent.system_prompt),
            },
            "param_bounds_sha256": _sha256(json.dumps(PARAM_BOUNDS, sort_keys=True)),
            "frozen_parameters_sha256": _sha256(json.dumps(sorted(FROZEN_PARAMETERS))),
            "param_bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
            "frozen_parameters": sorted(FROZEN_PARAMETERS),
            "zones": self.zone_ids,
            "zone_area_pct": self.zone_areas,
            "dim": len([n for n in self.current_params.values
                        if _base(n) not in FROZEN_PARAMETERS]),
            "objective": self.objective,
            "failure_patient": self.failure_patient,
            "max_rounds": self.max_rounds_requested,
            "n_candidates": self.n_candidates,
            "n_peaks": self.n_peaks,
            "memory_cutoff": self.memory_cutoff,
            "physics_information": self.physics_information,
            "image_input": self.image_input,
            "image_type": self.image_type,
            "llm_model_requested": self.llm_model,
            "llm_provider_requested": self.llm_provider,
            "thinking_budget": self.thinking_budget,
            "keep_legacy_failure_requery": self.keep_legacy_failure_requery,
            "direct_keep_failures": getattr(self, "direct_keep_failures", False),
            "anon_guide": _anon_guide_on(),
            "guide_mode": _guide_mode(),
            "basin_area_km2": self.basin_area_km2,
            "gauge": self.runner.gauge_num,
        }

    def usage_ledger(self) -> Dict[str, Any]:
        """Aggregate every LLM call made in this run — the R2-8d cost ledger."""
        calls: List[Dict[str, Any]] = []
        for agent in self._all_agents():
            calls.extend(getattr(agent, "usage_log", []))
        used_calls = [c for c in calls if c.get("used", True)]

        def _s(key: str, subset: List[Dict[str, Any]]) -> int:
            return int(sum(c.get(key) or 0 for c in subset))

        by_stage: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            st = c.get("stage", "?")
            d = by_stage.setdefault(st, {"n_calls": 0, "n_calls_used": 0, "prompt_tokens": 0,
                                         "completion_tokens": 0, "reasoning_tokens": 0,
                                         "latency_s": 0.0})
            d["n_calls"] += 1
            d["n_calls_used"] += 1 if c.get("used", True) else 0
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                d[k] += int(c.get(k) or 0)
            d["latency_s"] += float(c.get("latency_s") or 0.0)
        for d in by_stage.values():
            d["latency_s"] = round(d["latency_s"], 2)

        counters = {k: 0 for k in VALIDATION_COUNTER_KEYS}
        validation_events: List[Dict[str, Any]] = []
        for agent in self._all_agents():
            for k, v in getattr(agent, "validation_counters", {}).items():
                counters[k] = counters.get(k, 0) + int(v)
            validation_events.extend(getattr(agent, "validation_log", []))

        models = sorted({c.get("model_returned") for c in calls if c.get("model_returned")})
        timing_rounds = list(self.timing_rounds)
        return {
            "workflow": self.workflow,
            "perception": self.perception,
            "models_returned": models,
            "n_llm_calls": len(calls),
            "n_llm_calls_billed": len(calls),
            "n_llm_calls_used": len(used_calls),
            "prompt_tokens": _s("prompt_tokens", calls),
            "completion_tokens": _s("completion_tokens", calls),
            "reasoning_tokens": _s("reasoning_tokens", calls),
            "cached_prompt_tokens": _s("cached_prompt_tokens", calls),
            "total_tokens": _s("total_tokens", calls),
            "n_images_sent": _s("n_images", calls),
            "llm_latency_s": round(sum(float(c.get("latency_s") or 0.0) for c in calls), 2),
            "used_only": {
                "prompt_tokens": _s("prompt_tokens", used_calls),
                "completion_tokens": _s("completion_tokens", used_calls),
                "reasoning_tokens": _s("reasoning_tokens", used_calls),
                "total_tokens": _s("total_tokens", used_calls),
                "llm_latency_s": round(sum(float(c.get("latency_s") or 0.0) for c in used_calls), 2),
            },
            "validation": {**counters, "events": validation_events},
            "rejections": list(self.rejections),
            "timing": {
                "total_wall_s": round(time.perf_counter() - self._t_start, 3),
                "ef5_seconds_total": round(sum(float(t.get("ef5_seconds") or 0.0) for t in timing_rounds), 3),
                "test_seconds_total": round(sum(float(t.get("test_seconds") or 0.0) for t in timing_rounds), 3),
                "n_sims_total": int(sum(int(t.get("n_sims") or 0) for t in timing_rounds)),
                "n_test_sims_total": int(sum(int(t.get("n_test_sims") or 0) for t in timing_rounds)),
                "rounds": timing_rounds,
            },
            "progress": {
                "update_round": self.update_round,
                "failure_round": self.failure_round,
                "accepted_rounds": len(self.history.rounds),
                "n_failed_rounds": len(self.failure_history),
                "completed": self.completed,
                "stop_reason": self.stop_reason,
            },
            "provenance": self.provenance(),
            "by_stage": by_stage,
            "calls": calls,
        }

    def write_usage_ledger(self) -> Optional[str]:
        try:
            path = Path(self.history.path).parent / "usage_ledger.json"
            _atomic_write_text(path, json.dumps(self.usage_ledger(), indent=2, ensure_ascii=False))
            return str(path)
        except Exception as exc:
            print(f"[Warn] could not write usage ledger: {type(exc).__name__}: {exc}")
            return None

    def _write_history_timing(self) -> None:
        """Attach forward-model wall-clock to calibration_history.json (atomic rewrite).

        The ``timing`` block is stripped again in ``_history_payload`` so the
        payload handed to the LLM is unchanged.
        """
        try:
            path = Path(self.history.path)
            if not path.exists():
                return
            payload = json.loads(path.read_text())
            payload["timing"] = {
                "total_wall_s": round(time.perf_counter() - self._t_start, 3),
                "rounds": list(self.timing_rounds),
            }
            # what a resumed process needs and cannot rebuild from the rounds alone
            payload["progress"] = {
                "update_round": self.update_round,
                "failure_round": self.failure_round,
                "resumed_segment": self._resumed_segment,
            }
            payload["failures"] = list(self.failure_history)
            _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[Warn] could not write timing into history: {type(exc).__name__}: {exc}")

    def _checkpoint(self) -> None:
        """Persist ledger + timing; called after every round, failure and rejection."""
        self._write_history_timing()
        self.write_usage_ledger()

    def _record_failure(self,
                        best_outcome: CandidateOutcome,
                        objective_value: float,
                        proposals: List[Dict[str, Any]],
                        refined_candidates: List[Dict[str, Any]]) -> None:
        candidate_index = best_outcome.simulation.candidate_index
        refined_candidate = (
            refined_candidates[candidate_index]
            if 0 <= candidate_index < len(refined_candidates)
            else {}
        )
        record = {
            "round_label": self.round_label,
            "candidate_index": candidate_index,
            "objective": self.objective,
            "objective_value": objective_value,
            "aggregate_metrics": best_outcome.aggregate_metrics,
            "full_metrics": best_outcome.full_metrics,
            "best_candidate": refined_candidate,
            "proposals": proposals,
        }
        self.failure_history.append(record)

    def _handle_rejected_request(self, exc: CandidateRejected) -> bool:
        """Book a validation rejection as a failure round. Returns True to stop."""
        self.rejections.append({
            "stage": exc.stage,
            "round_label": exc.round_label,
            "n_attempts": len(exc.attempts),
            "final_error": exc.attempts[-1].get("error") if exc.attempts else None,
            "update_round": self.update_round,
            "failure_round_before": self.failure_round,
        })
        self._log_detail(f"{exc.stage}_rejected", exc.round_label, {
            "stage": exc.stage,
            "round": exc.round_label,
            "system_prompt": "",
            "user_prompt": "",
            "input_files": [],
            "input_paths": [],
            "output_text": "",
            "attempts": exc.attempts,
        })
        self.failure_round += 1
        print(
            f"[Round {exc.round_label}] {exc.stage} reply rejected after {len(exc.attempts)} attempt(s): "
            f"{exc.attempts[-1].get('error') if exc.attempts else '?'}. Counting as a failure "
            f"({self.failure_round}/{self.failure_patient})."
        )
        self._checkpoint()
        if self.failure_round >= self.failure_patient:
            print(
                f"[Round {exc.round_label}] Failure limit reached "
                f"({self.failure_round}/{self.failure_patient}). Stopping calibration."
            )
            self.stop_reason = "failure_patient"
            return True
        return False

    def _filter_metrics_for_no_image(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Remove event-related metrics (peak_ratio, lag_hours, etc.) when image_input is off."""
        excluded_keys = {"peak_ratio", "lag_hours", "lag_hours_sim_minus_obs", "sim_peak", "obs_peak"}
        return {k: v for k, v in metrics.items() if k not in excluded_keys}

    def _build_context(self, *, image_summary: str) -> RoundContext:
        assert self.best_outcome is not None
        description = "Top candidate metrics averaged across selected events."
        images: List[str] = []
        if self.workflow in ("direct", "direct_matched") and self.perception == "image" and self.image_input:
            # Single-call workflows have no perception stage, so the montage goes
            # straight to the one proposal call (in direct_matched this is what keeps
            # the information budget matched to full INCA).
            self._ensure_plots(self.best_outcome)
            montage = self._ensure_montage(self.best_outcome)
            if montage:
                images = [montage]
        display_params = display_parameters(self.best_outcome.params.values, self.display_name_map)
        prompt_param_names = (
            self.display_name_map
            if self.physics_information
            else {v: v for v in self.display_name_map.values()}
        )
        failure_summary, failure_details = self._format_failure_summary()
        if self.workflow in ("direct", "direct_matched") and not self.direct_keep_failures:
            # No failure memory in the single-call baselines — that is part of the
            # architecture being ablated, not part of the information budget.
            # With direct_keep_failures the single call sees the same failure summary
            # as full INCA, so the only thing removed is the stage split itself.
            failure_summary, failure_details = "", []

        # When image_input is off, exclude all event-related diagnostics
        if not self.image_input:
            description = "Event diagnostics withheld for this run."
            event_metrics = []
            aggregate_metrics = self._filter_metrics_for_no_image(self.best_outcome.aggregate_metrics)
            full_metrics = self._filter_metrics_for_no_image(self.best_outcome.full_metrics)
        elif self.image_type in ("noimage", "noboth"):
            description = "Event diagnostics withheld for this run."
            event_metrics = []
            aggregate_metrics = self._filter_metrics_for_no_image(self.best_outcome.aggregate_metrics)
            full_metrics = self._filter_metrics_for_no_image(self.best_outcome.full_metrics)
        else:
            event_metrics = self.best_outcome.event_metrics[: self.n_peaks]
            aggregate_metrics = self.best_outcome.aggregate_metrics
            full_metrics = self.best_outcome.full_metrics

        bounds, at_bound = ({}, {})
        if self.bounds_hint:
            bounds, at_bound = bounds_hint_of(self.best_outcome.params.values, self.display_name_map)
        return RoundContext(
            round_index=self.round_label,
            params=self.best_outcome.params.values.copy(),
            display_params=display_params,
            param_display_names=prompt_param_names,
            aggregate_metrics=aggregate_metrics,
            full_metrics=full_metrics,
            event_metrics=event_metrics,
            history_summary=self._history_summary(),
            description=description,
            images=images,
            image_summary=image_summary,
            failure_summary=failure_summary,
            failure_details=failure_details,
            physics_information=self.physics_information,
            physics_prompt=self.physics_prompt,
            parameter_bounds=bounds,
            at_bound=at_bound,
        )

    def _history_payload(self) -> Dict[str, Any]:
        if not self.history.path.exists():
            return {}
        payload = json.loads(self.history.path.read_text())
        # Run-level timing / resume bookkeeping, never part of the prompt.
        for key in ("timing", "progress", "failures"):
            payload.pop(key, None)
        rounds = payload.get("rounds")
        if isinstance(rounds, list):
            payload["rounds"] = self._apply_history_limit(rounds)

        # When image_input is off, filter out event-related metrics from history
        if not self.image_input:
            payload = self._filter_history_payload(payload)

        return payload

    def _filter_history_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Remove event-related metrics from history payload when image_input is off."""
        filtered = payload.copy()

        # Filter rounds
        if "rounds" in filtered and isinstance(filtered["rounds"], list):
            filtered_rounds = []
            for round_data in filtered["rounds"]:
                filtered_round = round_data.copy()
                # Remove event_metrics from each candidate
                if "candidates" in filtered_round:
                    filtered_candidates = []
                    for cand in filtered_round["candidates"]:
                        filtered_cand = cand.copy()
                        filtered_cand["event_metrics"] = []
                        if "metrics" in filtered_cand:
                            filtered_cand["metrics"] = self._filter_metrics_for_no_image(filtered_cand["metrics"])
                        if "full_metrics" in filtered_cand:
                            filtered_cand["full_metrics"] = self._filter_metrics_for_no_image(filtered_cand["full_metrics"])
                        filtered_candidates.append(filtered_cand)
                    filtered_round["candidates"] = filtered_candidates
                # Filter best_candidates_by_metric
                if "best_candidates_by_metric" in filtered_round:
                    for key, best_data in filtered_round["best_candidates_by_metric"].items():
                        if "aggregate_metrics" in best_data:
                            best_data["aggregate_metrics"] = self._filter_metrics_for_no_image(best_data["aggregate_metrics"])
                        if "full_metrics" in best_data:
                            best_data["full_metrics"] = self._filter_metrics_for_no_image(best_data["full_metrics"])
                filtered_rounds.append(filtered_round)
            filtered["rounds"] = filtered_rounds

        # Filter best metrics
        if "best" in filtered and filtered["best"]:
            if "metrics" in filtered["best"]:
                filtered["best"]["metrics"] = self._filter_metrics_for_no_image(filtered["best"]["metrics"])
            if "aggregate_metrics" in filtered["best"]:
                filtered["best"]["aggregate_metrics"] = self._filter_metrics_for_no_image(filtered["best"]["aggregate_metrics"])
            if "full_metrics" in filtered["best"]:
                filtered["best"]["full_metrics"] = self._filter_metrics_for_no_image(filtered["best"]["full_metrics"])

        # Filter best_by_metric
        if "best_by_metric" in filtered:
            for key, best_data in filtered["best_by_metric"].items():
                if "aggregate_metrics" in best_data:
                    best_data["aggregate_metrics"] = self._filter_metrics_for_no_image(best_data["aggregate_metrics"])
                if "full_metrics" in best_data:
                    best_data["full_metrics"] = self._filter_metrics_for_no_image(best_data["full_metrics"])

        return filtered

    def _log_detail(self, stage: str, round_label: str, payload: Dict[str, Any]) -> None:
        """Write ``detail_logs/<round_label>/<stage>.json``.

        The first write of a (round, stage) pair in this process always lands on
        ``<stage>.json`` — that file therefore describes the candidates actually
        simulated under that label. A second write of the same pair (legacy
        failure re-query, prefetch collisions) goes to ``<stage>_2.json``, ``_3``…
        instead of overwriting.
        """
        if not self.detail_output:
            return
        if self._legacy_requery_active and isinstance(payload.get("usage"), dict):
            # Same dict object as the ledger entry: the file and the ledger agree.
            payload["usage"]["used"] = False
            payload["usage"]["legacy_failure_requery"] = True
        out_dir = self.detail_dir / round_label
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{stage}.json"
        seq = 1
        while str(target) in self._detail_written:
            seq += 1
            target = out_dir / f"{stage}_{seq}.json"
        self._detail_written.add(str(target))
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[Detail] Saved {stage} log for round {round_label} to {target}.")
        prompt_text = payload.get("user_prompt")
        if prompt_text:
            print(f"[Detail] {stage.capitalize()} prompt (round {round_label}):\n{prompt_text}")

    def _candidate_score(self, aggregate: Dict[str, float], full: Dict[str, float]) -> float:
        nse = aggregate.get("NSE", float("nan"))
        if not np.isfinite(nse):
            return float("-inf")
        score = 0.5 * nse
        full_nse = full.get("NSE", float("nan"))
        if np.isfinite(full_nse):
            score += 0.3 * full_nse
        cc = aggregate.get("CC", float("nan"))
        if np.isfinite(cc):
            score += 0.1 * cc
        kge = aggregate.get("KGE", float("nan"))
        if np.isfinite(kge):
            score += 0.1 * kge
        lag = aggregate.get("lag_hours", float("nan"))
        if np.isfinite(lag):
            score -= 0.05 * abs(lag)
        peak_ratio = aggregate.get("peak_ratio", float("nan"))
        if np.isfinite(peak_ratio) and peak_ratio > 0:
            score -= 0.1 * abs(math.log(peak_ratio))
        return score

    def _objective_value(self, aggregate: Dict[str, float], full: Dict[str, float]) -> float:
        metric_key = self.objective.upper()
        value = full.get(metric_key, float("nan"))

        return value if np.isfinite(value) else float("-inf")

    def _collect_round_bests(self, outcomes: Sequence[CandidateOutcome]) -> Dict[str, Dict[str, Any]]:
        metric_extractors = {
            "objective": lambda outcome: self._objective_value(outcome.aggregate_metrics, outcome.full_metrics),
            "full_nse": lambda outcome: outcome.full_metrics.get("NSE", float("nan")),
            "full_cc": lambda outcome: outcome.full_metrics.get("CC", float("nan")),
            "full_kge": lambda outcome: outcome.full_metrics.get("KGE", float("nan")),
        }
        round_bests: Dict[str, Dict[str, Any]] = {}
        for key, extractor in metric_extractors.items():
            best_value = float("-inf")
            best_outcome: Optional[CandidateOutcome] = None
            for outcome in outcomes:
                value = extractor(outcome)
                if not np.isfinite(value):
                    continue
                if best_outcome is None or value > best_value:
                    best_value = value
                    best_outcome = outcome
            if best_outcome is None:
                continue
            self._ensure_plots(best_outcome)
            round_bests[key] = self._summarize_outcome(
                best_outcome,
                criterion=key,
                value=best_value,
            )
        return round_bests

    def _assemble_by_zone(self,
                          outcomes: List[CandidateOutcome],
                          timing: Optional[Dict[str, Any]] = None) -> Optional[CandidateOutcome]:
        """Build one extra candidate by taking each zone's block from whoever fitted that
        zone's own gauge best, then actually simulate it.

        Winner-take-all throws away most of what a round produced. In the 100-parameter
        multi-site run, seven of eight candidates improved the Lamington gauge every round
        and all seven were discarded because the candidate with the best *mean* happened
        to be the one that left that zone alone -- a zone worth 1/10 of the objective
        cannot outvote the two zones holding 55% of the basin. Assembling per zone keeps
        those fixes.

        The incumbent is one of the sources, so a zone nobody improved keeps its current
        block. Zones are nested, so a gauge's score also reflects the zones above it and
        the assembled vector is not guaranteed to be better -- which is why it is
        evaluated (one extra forward run) and then competes on equal terms rather than
        being accepted on assumption. Returns None when the assembly reproduces a
        candidate already simulated, so nothing is spent.
        """
        if not (self.zone_assembly and self.multisite_gauges and self.zone_ids):
            return None
        sources: List[Tuple[str, CandidateOutcome]] = [("incumbent", self.best_outcome)]
        sources += [(str(o.simulation.candidate_index), o) for o in outcomes]

        picks: Dict[str, str] = {}
        values: Dict[str, float] = {}
        for zone in self.zone_ids:
            key = f"NSE@{zone}"
            best_tag, best_val = None, float("-inf")
            for tag, o in sources:
                v = (o.aggregate_metrics or {}).get(key, float("nan"))
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v > best_val:
                    best_tag, best_val = tag, v
            if best_tag is None:
                return None
            picks[zone] = best_tag
            values[zone] = best_val
        if len(set(picks.values())) <= 1:
            return None  # one source already owns every zone: nothing to assemble

        by_tag = dict(sources)
        merged = dict(self.best_outcome.params.values)
        for name in merged:
            zone = zone_of(name)
            if zone in picks:
                src = by_tag[picks[zone]].params.values
                if name in src:
                    merged[name] = src[name]
        params = ParameterSet(merged)
        label = ", ".join(f"{z[-5:]}<-{picks[z]}({values[z]:+.2f})" for z in self.zone_ids)
        print(f"[Round {self.round_label}] Zone assembly: {label}")

        idx = max((o.simulation.candidate_index for o in outcomes), default=-1) + 1
        t0 = time.perf_counter()
        try:
            res = self.runner.run(params, round_label=self.round_label, candidate_index=idx)
        except Exception as exc:  # noqa: BLE001 - a failed assembly must not kill the round
            print(f"[Warn] zone assembly simulation failed: {type(exc).__name__}: {exc}")
            return None
        if timing is not None:
            timing["n_sims"] = int(timing.get("n_sims") or 0) + 1
            timing["ef5_seconds"] = round(float(timing.get("ef5_seconds") or 0.0)
                                          + (time.perf_counter() - t0), 3)
            timing["zone_assembly"] = picks
        outcome = self._process_result(res)
        print(f"[Round {self.round_label}]     [Cand {idx}] assembled: "
              f"objective({self.objective})="
              f"{self._objective_value(outcome.aggregate_metrics, outcome.full_metrics):.3f}")
        return outcome

    def _select_best(self, outcomes: List[CandidateOutcome]) -> int:
        best_idx = -1
        best_score = -math.inf
        for idx, outcome in enumerate(outcomes):
            score = self._objective_value(outcome.aggregate_metrics, outcome.full_metrics)
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx if best_idx != -1 else 0

    # ------------------------------------------------------------------- run
    def run(self, max_rounds: int = MAX_STEPS_DEFAULT) -> None:
        """Run the calibration loop, always emitting the cost ledger on exit.

        SIGTERM (SLURM wall-clock) is converted into an exception so the
        ``finally`` block — ledger + timing — still runs before the process dies.
        """
        self.max_rounds_requested = max_rounds
        self._t_start = time.perf_counter()

        def _on_sigterm(signum, frame):  # pragma: no cover - signal path
            raise _Terminated(f"received signal {signum}")

        prev_handler = None
        try:
            prev_handler = signal.signal(signal.SIGTERM, _on_sigterm)
        except (ValueError, OSError):  # not in the main thread / unsupported
            prev_handler = None

        terminated: Optional[BaseException] = None
        try:
            self._run_loop(max_rounds)
            self.completed = True
            if self.stop_reason is None:
                self.stop_reason = "max_rounds" if self.update_round > max_rounds else "loop_exit"
        except _Terminated as exc:
            terminated = exc
            self.stop_reason = "sigterm"
            print(f"[Warn] {exc}; flushing ledger before exit.")
        except BaseException as exc:
            if self.stop_reason is None:
                self.stop_reason = f"exception:{type(exc).__name__}"
            raise
        finally:
            self._write_history_timing()
            path = self.write_usage_ledger()
            if prev_handler is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_handler)
                except (ValueError, OSError):
                    pass
            if path:
                led = self.usage_ledger()
                val = led["validation"]
                print(
                    f"[Cost] {led['n_llm_calls_billed']} LLM calls billed / {led['n_llm_calls_used']} used | "
                    f"prompt={led['prompt_tokens']} completion={led['completion_tokens']} "
                    f"(reasoning={led['reasoning_tokens']}) | "
                    f"LLM wall-clock {led['llm_latency_s']}s | EF5 {led['timing']['ef5_seconds_total']}s "
                    f"({led['timing']['n_sims_total']} sims) | total wall {led['timing']['total_wall_s']}s | "
                    f"validation retries={val['n_retry']} rejected={val['n_rejected']} | "
                    f"models={led['models_returned']} | ledger -> {path}"
                )
        if terminated is not None:
            raise SystemExit(143)

    def _run_loop(self, max_rounds: int = MAX_STEPS_DEFAULT) -> None:
        if self.detail_output:
            print(f"[Debug] Writing detailed agent I/O to {self.detail_dir}.")
        if self.best_outcome is None:
            if not (self.resume and self._resume_from_disk()):
                self.initialize_baseline()

        next_proposals = None
        next_refined = None
        next_params = None
        next_eval_meta: Dict[str, Any] = {}

        while self.update_round <= max_rounds:
            self.round_label = self._format_round_label(self.update_round, self.failure_round)
            self.round_index = self.update_round

            if next_params is None:
                try:
                    image_summary = self._build_image_summary(self.best_outcome)
                    context = self._build_context(image_summary=image_summary)
                    proposals, refined_candidates, refined_params, eval_meta = self._request_candidates(
                        context, self.round_label
                    )
                except CandidateRejected as exc:
                    if self._handle_rejected_request(exc):
                        break
                    continue
            else:
                proposals = next_proposals or []
                refined_candidates = next_refined or []
                refined_params = next_params
                eval_meta = next_eval_meta
                next_proposals = None
                next_refined = None
                next_params = None
                next_eval_meta = {}

            print(
                f"[Round {self.round_label}] Launching {len(refined_params)} simulations "
                f"(max_workers={self.max_workers or 'auto'})…"
            )
            timing = {
                "round_label": self.round_label, "update_round": self.update_round,
                "failure_round": self.failure_round, "n_sims": len(refined_params),
                "ef5_seconds": None, "n_test_sims": 0, "test_seconds": 0.0,
                "accepted": False, "kind": "candidates",
            }
            self.timing_rounds.append(timing)
            t0 = time.perf_counter()
            results = run_simulations_parallel(
                self.runner,
                refined_params,
                self.round_label,
                self.max_workers,
            )
            timing["ef5_seconds"] = round(time.perf_counter() - t0, 3)
            outcomes = [self._process_result(res) for res in results]
            composite = self._assemble_by_zone(outcomes, timing)
            if composite is not None:
                outcomes.append(composite)

            print(f"[Round {self.round_label}] Candidate performance summary:")
            for outcome in outcomes:
                agg = outcome.aggregate_metrics
                full = outcome.full_metrics
                objective_value = self._objective_value(agg, full)
                print(
                    f"    [Cand {outcome.simulation.candidate_index}] "
                    f"objective({self.objective})={objective_value:.3f} | "
                    f"full NSE={full.get('NSE', float('nan')):.3f} "
                    f"CC={full.get('CC', float('nan')):.3f} "
                    f"KGE={full.get('KGE', float('nan')):.3f}"
                )

            best_idx = self._select_best(outcomes)
            best_outcome = outcomes[best_idx]
            self._ensure_plots(best_outcome)

            agg = best_outcome.aggregate_metrics
            full = best_outcome.full_metrics
            objective_value = self._objective_value(agg, full)
            print(
                f"[Round {self.round_label}] Best candidate {best_outcome.simulation.candidate_index}: "
                f"objective({self.objective})={objective_value:.3f} | "
                f"full NSE={full.get('NSE', float('nan')):.3f} "
                f"full CC={full.get('CC', float('nan')):.3f} "
                f"full KGE={full.get('KGE', float('nan')):.3f}"
            )

            if objective_value <= self.best_objective_value:
                self._record_failure(best_outcome, objective_value, proposals, refined_candidates)
                self.failure_round += 1
                print(
                    f"[Round {self.round_label}] No improvement over best objective "
                    f"({self.best_objective_value:.3f}). Retrying proposals "
                    f"(failure {self.failure_round}/{self.failure_patient})."
                )
                if self.failure_round >= self.failure_patient:
                    print(
                        f"[Round {self.round_label}] Failure limit reached "
                        f"({self.failure_round}/{self.failure_patient}). Stopping calibration."
                    )
                    self.stop_reason = "failure_patient"
                    self._checkpoint()
                    break
                if self.keep_legacy_failure_requery:
                    # Submitted-version behaviour, reproduced on request only: this
                    # request is a dead store (the loop top re-requests because
                    # next_params is still None), so its calls are flagged
                    # used=False and its detail logs get a _2 suffix.
                    marks = [len(a.usage_log) for a in self._all_agents()]
                    self._legacy_requery_active = True
                    try:
                        image_summary = self._build_image_summary(self.best_outcome)
                        context = self._build_context(image_summary=image_summary)
                        self._request_candidates(context, self.round_label)
                    except CandidateRejected as exc:
                        print(f"[Round {self.round_label}] (legacy re-query) {exc}")
                    finally:
                        self._legacy_requery_active = False
                    for agent, mark in zip(self._all_agents(), marks):
                        for entry in agent.usage_log[mark:]:
                            entry["used"] = False
                            entry["legacy_failure_requery"] = True
                self._checkpoint()
                # The next iteration re-requests with the advanced round label
                # (round{update}_{failure+1}); exactly one request per simulated round.
                continue

            self.best_outcome = best_outcome
            self.best_objective_value = objective_value
            self.current_params = best_outcome.params.copy()
            if not self.zone_ids:
                # zoned names ('wm@V1') have no argparse attribute to mirror into
                self.current_params.to_object(self.args_obj)
            self.failure_round = 0
            self.update_round += 1
            timing["accepted"] = True

            candidate_records = [
                CandidateRecord(
                    candidate_index=outcome.simulation.candidate_index,
                    params=outcome.params.values.copy(),
                    metrics=outcome.aggregate_metrics,
                    full_metrics=outcome.full_metrics,
                    event_metrics=outcome.event_metrics,
                )
                for outcome in outcomes
            ]
            round_metric_bests = self._collect_round_bests(outcomes)
            round_record = RoundRecord(
                round_index=self.round_index,
                proposals=proposals,
                refined_candidates=refined_candidates,
                candidates=candidate_records,
                best_candidate_index=best_outcome.simulation.candidate_index,
                rationale=eval_meta.get("rationale", ""),
                risk=eval_meta.get("risk", ""),
                focus=eval_meta.get("focus", ""),
                best_candidates_by_metric=round_metric_bests,
            )
            self.history.rounds.append(round_record)
            improved = self.history.update_best(
                aggregate_metrics=best_outcome.aggregate_metrics,
                full_metrics=best_outcome.full_metrics,
                params=best_outcome.params.values.copy(),
                round_index=self.round_index,
                candidate_index=best_outcome.simulation.candidate_index,
                objective_key=self.objective.upper(),
            )
            self._update_metric_bests(outcomes, round_index=self.round_index)
            self.history.save()
            if improved:
                print(
                    f"[Round {self.round_label}] New global best found "
                    f"(candidate {best_outcome.simulation.candidate_index})."
                )

            stop_after_prefetch = False
            if self.test_config and self.test_config.enabled:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {"test": executor.submit(self._run_test_suite, refined_params, self.round_label, timing)}
                    if self.update_round <= max_rounds:
                        next_label = self._format_round_label(self.update_round, 0)
                        # Prompt text still carries self.round_label (unchanged);
                        # only the detail_logs directory is the next round's.
                        image_summary = self._build_image_summary(self.best_outcome, log_label=next_label)
                        next_context = self._build_context(image_summary=image_summary)
                        futures["proposal"] = executor.submit(
                            self._request_candidates, next_context, next_label
                        )

                    for key, future in futures.items():
                        if key == "test":
                            test_summaries = future.result()
                            print(
                                f"[Round {self.round_label}] Test summaries written for {len(test_summaries)} candidates."
                            )
                        elif key == "proposal":
                            try:
                                (next_proposals, next_refined, next_params, next_eval_meta) = future.result()
                            except CandidateRejected as exc:
                                next_proposals = next_refined = next_params = None
                                next_eval_meta = {}
                                if self._handle_rejected_request(exc):
                                    stop_after_prefetch = True
            self._checkpoint()
            if stop_after_prefetch:
                break

    def _update_metric_bests(self,
                             outcomes: Sequence[CandidateOutcome],
                             round_index: int) -> None:
        for outcome in outcomes:
            agg = outcome.aggregate_metrics
            full = outcome.full_metrics
            for metric in ("NSE", "CC", "KGE"):
                value = full.get(metric, float("nan"))
                if not np.isfinite(value):
                    continue
                key = f"full_{metric.lower()}"
                if self.history.update_best_metric(
                    key=key,
                    value=value,
                    aggregate_metrics=agg,
                    full_metrics=full,
                    params=outcome.params.values.copy(),
                    round_index=round_index,
                    candidate_index=outcome.simulation.candidate_index,
                ):
                    self._ensure_plots(outcome)
                    self._publish_best(
                        outcome,
                        subfolder=f"best_{key}",
                        metadata={"criterion": key, "criterion_value": value},
                    )

            objective_value = self._objective_value(agg, full)
            if self.history.update_best_metric(
                key="objective",
                value=objective_value,
                aggregate_metrics=agg,
                full_metrics=full,
                params=outcome.params.values.copy(),
                round_index=round_index,
                candidate_index=outcome.simulation.candidate_index,
            ):
                self._ensure_plots(outcome)
                self._publish_best(
                    outcome,
                    subfolder=f"best_objective_{self.objective}",
                    metadata={"criterion": self.objective, "criterion_value": objective_value},
                )


__all__ = ["TwoStageCalibrationManager", "CandidateOutcome", "TestConfig"]
