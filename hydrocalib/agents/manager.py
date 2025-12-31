"""Two-stage calibration manager orchestrating proposal/evaluation agents."""

from __future__ import annotations

import json
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import (DEFAULT_GAUGE_NUM, DEFAULT_PEAK_PICK_KWARGS, DEFAULT_SIM_FOLDER,
                       EVENTS_FOR_AGGREGATE, IMPROVE_PATIENCE, MAX_STEPS_DEFAULT)
from ..history import CandidateRecord, HistoryStore, RoundRecord
from ..metrics import (aggregate_event_metrics, compute_event_metrics,
                       read_metrics_for_period, read_metrics_from_csv)
from ..parameters import ParameterSet
from .physics_info import (
    build_display_name_map,
    display_parameters,
    invert_display_map,
    render_parameter_guide,
)
from ..peak_events import pick_peak_events
from ..plotting import plot_event_windows, plot_hydrograph_with_precipitation
from ..simulation import SimulationResult, SimulationRunner, run_simulations_parallel
from .evaluation import EvaluationAgent
from .proposal import ProposalAgent
from .types import RoundContext


@dataclass
class CandidateOutcome:
    simulation: SimulationResult
    params: ParameterSet
    windows: List[Tuple]
    event_metrics: List[Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    full_metrics: Dict[str, float]
    hydrograph_path: Optional[str] = None
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
                 objective: str = "nse_event",
                 physics_information: bool = True,
                 image_input: bool = True,
                 detail_output: bool = False):
        self.args_obj = args_obj
        self.current_params = ParameterSet.from_object(args_obj)
        self.runner = SimulationRunner(simu_folder=simu_folder, gauge_num=gauge_num)
        self.display_name_map = build_display_name_map(physics_information)
        self.reverse_display_map = invert_display_map(self.display_name_map)
        self.physics_prompt = render_parameter_guide(physics_information, self.display_name_map)
        self.proposal_agent = ProposalAgent(
            physics_information=physics_information,
            display_name_map=self.display_name_map,
            detail_output=detail_output,
        )
        self.evaluation_agent = EvaluationAgent(
            physics_information=physics_information,
            display_name_map=self.display_name_map,
            detail_output=detail_output,
        )
        self.n_candidates = n_candidates
        self.n_peaks = n_peaks
        self.include_max_event_images = include_max_event_images
        self.peak_pick_kwargs = peak_pick_kwargs or DEFAULT_PEAK_PICK_KWARGS
        hist_path = history_path or (Path(simu_folder) / "results" / "calibration_history.json")
        self.history = HistoryStore(Path(hist_path))
        if memory_cutoff is not None and memory_cutoff < 0:
            raise ValueError("memory_cutoff must be non-negative (>= 0)")
        self.memory_cutoff = memory_cutoff
        self.best_outcome: Optional[CandidateOutcome] = None
        self.round_index = 0
        self.stall = 0
        self.max_workers = max_workers
        self.test_config = test_config
        self.objective = objective
        self.physics_information = physics_information
        self.image_input = image_input
        self.detail_output = detail_output
        self.detail_dir = Path(simu_folder) / "results" / "detail_logs"

    def initialize_baseline(self) -> None:
        print("[Init] Running baseline simulation…")
        baseline_result = self.runner.run(self.current_params, round_index=0, candidate_index=0)
        outcome = self._process_result(baseline_result)
        self._ensure_plots(outcome)
        self.best_outcome = outcome
        improved = self.history.update_best(
            aggregate_metrics=outcome.aggregate_metrics,
            full_metrics=outcome.full_metrics,
            params=outcome.params.values.copy(),
            round_index=0,
            candidate_index=0,
        )
        self._update_metric_bests([outcome], round_index=0)
        self.history.save()
        agg = outcome.aggregate_metrics
        full = outcome.full_metrics
        print(
            "[Init] Baseline metrics "
            f"event NSE={agg.get('NSE', float('nan')):.3f} "
            f"event CC={agg.get('CC', float('nan')):.3f} "
            f"event KGE={agg.get('KGE', float('nan')):.3f} | "
            f"full NSE={full.get('NSE', float('nan')):.3f} "
            f"full CC={full.get('CC', float('nan')):.3f} "
            f"full KGE={full.get('KGE', float('nan')):.3f}"
        )

    def _process_result(self, result: SimulationResult) -> CandidateOutcome:
        windows = pick_peak_events(result.csv_path, n=self.n_peaks, **self.peak_pick_kwargs)
        event_metrics = compute_event_metrics(result.csv_path, windows)
        aggregate = aggregate_event_metrics(event_metrics, top_n=self.n_peaks)
        full_metrics = read_metrics_from_csv(result.csv_path)
        return CandidateOutcome(result, result.params, windows, event_metrics, aggregate, full_metrics)

    def _run_test_suite(self, params: Sequence[ParameterSet], round_index: int) -> Dict[int, Dict[str, Any]]:
        if not self.test_config or not self.test_config.enabled:
            return {}

        print(f"[Round {round_index}] Starting test suite for {len(params)} candidates…")
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
                round_index=round_index,
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
                "round_index": round_index,
                "candidate_index": idx,
                "params": param_set.values.copy(),
                "csv": result.csv_path,
                "metrics": metrics,
            }
            summaries[idx] = summary
            summary_path = Path(result.output_dir) / "summary.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

            print(
                f"    [Round {round_index} Test {idx}] NSE={metrics.get('NSE', float('nan')):.3f} "
                f"CC={metrics.get('CC', float('nan')):.3f} KGE={metrics.get('KGE', float('nan')):.3f}"
            )

        print(f"[Round {round_index}] Test suite finished.")
        return summaries

    def _request_candidates(self, context: RoundContext, round_index: int):
        print(f"[Round {round_index}] Requesting {self.n_candidates} proposals from proposal agent…")
        proposal_log = None
        proposals_result = self.proposal_agent.propose(
            context, self.n_candidates, return_log=self.detail_output
        )
        if self.detail_output:
            proposals, proposal_log = proposals_result
        else:
            proposals = proposals_result
        proposal_params = self.proposal_agent.apply_candidates(self.best_outcome.params, proposals)
        print(f"[Round {round_index}] Initial proposals and parameter sets:")
        for idx, (proposal, params) in enumerate(zip(proposals, proposal_params)):
            goal = proposal.get("goal") or proposal.get("id") or f"cand_{idx}"
            print(
                f"    [Proposal {idx}] goal={goal} updates={proposal.get('updates', {})} "
                f"→ params={params.values}"
            )

        refined_candidates, eval_meta, eval_log = self.evaluation_agent.refine(
            context,
            proposals,
            self._history_payload(),
            self.n_candidates,
            return_log=self.detail_output,
        )
        refined_params = self.evaluation_agent.apply_candidates(self.best_outcome.params, refined_candidates)
        if not refined_params:
            refined_candidates = proposals
            refined_params = proposal_params
        print(f"[Round {round_index}] Evaluation agent refined parameter sets:")
        for idx, (candidate, params) in enumerate(zip(refined_candidates, refined_params)):
            goal = candidate.get("goal") or candidate.get("id") or f"cand_{idx}"
            print(
                f"    [Refined {idx}] goal={goal} updates={candidate.get('updates', {})} "
                f"→ params={params.values}"
            )
        if proposal_log:
            self._log_detail("proposal", round_index, proposal_log)
        if eval_log:
            self._log_detail("evaluation", round_index, eval_log)
        return proposals, refined_candidates, refined_params, eval_meta

    def _ensure_plots(self, outcome: CandidateOutcome) -> None:
        hydrograph_missing = not outcome.hydrograph_path or not Path(outcome.hydrograph_path).exists()
        if hydrograph_missing:
            outcome.hydrograph_path = plot_hydrograph_with_precipitation(outcome.simulation.csv_path, show=False)

        valid_figures = [fig for fig in outcome.event_figures if Path(fig).exists()]
        if len(valid_figures) < self.include_max_event_images:
            peaks_dir = Path(outcome.simulation.output_dir) / "peaks"
            valid_figures = plot_event_windows(
                outcome.simulation.csv_path,
                outcome.windows,
                out_dir=str(peaks_dir),
            )[:self.include_max_event_images]
        outcome.event_figures = valid_figures

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
        events_dir = best_dir / "events"
        events_dir.mkdir(exist_ok=True)
        for fig in outcome.event_figures:
            fig_path = Path(fig)
            if fig_path.exists():
                shutil.copy2(fig_path, events_dir / fig_path.name)
        summary: Dict[str, Any] = {
            "round_index": outcome.simulation.round_index,
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
            "round_index": outcome.simulation.round_index,
            "candidate_index": outcome.simulation.candidate_index,
            "criterion": criterion,
            "criterion_value": value,
            "aggregate_metrics": outcome.aggregate_metrics,
            "full_metrics": outcome.full_metrics,
            "params": outcome.params.values.copy(),
            "hydrograph": outcome.hydrograph_path,
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

    def _build_context(self) -> RoundContext:
        assert self.best_outcome is not None
        description = "Top candidate metrics averaged across selected events."
        images = []
        if self.image_input:
            if self.best_outcome.hydrograph_path:
                images.append(self.best_outcome.hydrograph_path)
            images.extend(self.best_outcome.event_figures[: self.include_max_event_images])
        display_params = display_parameters(self.best_outcome.params.values, self.display_name_map)
        prompt_param_names = (
            self.display_name_map
            if self.physics_information
            else {v: v for v in self.display_name_map.values()}
        )
        return RoundContext(
            round_index=self.round_index,
            params=self.best_outcome.params.values.copy(),
            display_params=display_params,
            param_display_names=prompt_param_names,
            aggregate_metrics=self.best_outcome.aggregate_metrics,
            full_metrics=self.best_outcome.full_metrics,
            event_metrics=self.best_outcome.event_metrics[: self.n_peaks],
            history_summary=self._history_summary(),
            description=description,
            images=images,
            physics_information=self.physics_information,
            physics_prompt=self.physics_prompt,
        )

    def _history_payload(self) -> Dict[str, Any]:
        if not self.history.path.exists():
            return {}
        payload = json.loads(self.history.path.read_text())
        rounds = payload.get("rounds")
        if isinstance(rounds, list):
            payload["rounds"] = self._apply_history_limit(rounds)
        return payload

    def _log_detail(self, stage: str, round_index: int, payload: Dict[str, Any]) -> None:
        if not self.detail_output:
            return
        out_dir = self.detail_dir / f"round_{round_index:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{stage}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[Detail] Saved {stage} log for round {round_index} to {target}.")
        prompt_text = payload.get("user_prompt")
        if prompt_text:
            print(f"[Detail] {stage.capitalize()} prompt (round {round_index}):\n{prompt_text}")

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
        if self.objective == "score":
            return self._candidate_score(aggregate, full)

        if self.objective == "nse_full":
            value = full.get("NSE", float("nan"))
        else:
            value = aggregate.get("NSE", float("nan"))

        return value if np.isfinite(value) else float("-inf")

    def _collect_round_bests(self, outcomes: Sequence[CandidateOutcome]) -> Dict[str, Dict[str, Any]]:
        metric_extractors = {
            "score": lambda outcome: self._candidate_score(outcome.aggregate_metrics, outcome.full_metrics),
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

    def _select_best(self, outcomes: List[CandidateOutcome]) -> int:
        best_idx = -1
        best_score = -math.inf
        for idx, outcome in enumerate(outcomes):
            score = self._objective_value(outcome.aggregate_metrics, outcome.full_metrics)
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx if best_idx != -1 else 0

    def run(self, max_rounds: int = MAX_STEPS_DEFAULT) -> None:
        if self.detail_output:
            print(f"[Debug] Writing detailed agent I/O to {self.detail_dir}.")
        if self.best_outcome is None:
            self.initialize_baseline()
        # Prime the first batch of candidates
        initial_context = self._build_context()
        proposals, refined_candidates, refined_params, eval_meta = self._request_candidates(initial_context, 1)

        for r in range(1, max_rounds + 1):
            self.round_index = r

            print(
                f"[Round {r}] Launching {len(refined_params)} simulations (max_workers={self.max_workers or 'auto'})…"
            )
            results = run_simulations_parallel(
                self.runner,
                refined_params,
                r,
                self.max_workers,
            )
            outcomes = [self._process_result(res) for res in results]

            print(f"[Round {r}] Candidate performance summary:")
            for outcome in outcomes:
                agg = outcome.aggregate_metrics
                full = outcome.full_metrics
                score = self._candidate_score(agg, full)
                objective_value = self._objective_value(agg, full)
                print(
                    f"    [Cand {outcome.simulation.candidate_index}] "
                    f"objective({self.objective})={objective_value:.3f} | score={score:.3f} | "
                    f"event NSE={agg.get('NSE', float('nan')):.3f} "
                    f"CC={agg.get('CC', float('nan')):.3f} KGE={agg.get('KGE', float('nan')):.3f} "
                    f"lag={agg.get('lag_hours', float('nan')):.2f}h | "
                    f"full NSE={full.get('NSE', float('nan')):.3f} CC={full.get('CC', float('nan')):.3f} "
                    f"KGE={full.get('KGE', float('nan')):.3f}"
                )

            best_idx = self._select_best(outcomes)
            best_outcome = outcomes[best_idx]
            self._ensure_plots(best_outcome)
            self.best_outcome = best_outcome
            self.current_params = best_outcome.params.copy()
            self.current_params.to_object(self.args_obj)

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
                round_index=r,
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
                round_index=r,
                candidate_index=best_outcome.simulation.candidate_index,
            )
            self._update_metric_bests(outcomes, round_index=r)
            self.history.save()
            if improved:
                print(f"[Round {r}] New global best found (candidate {best_outcome.simulation.candidate_index}).")

            agg = best_outcome.aggregate_metrics
            full = best_outcome.full_metrics
            objective_value = self._objective_value(agg, full)
            print(
                f"[Round {r}] Best candidate {best_outcome.simulation.candidate_index}: "
                f"objective({self.objective})={objective_value:.3f} | "
                f"event NSE={agg.get('NSE', float('nan')):.3f} "
                f"event CC={agg.get('CC', float('nan')):.3f} "
                f"event KGE={agg.get('KGE', float('nan')):.3f} | "
                f"full NSE={full.get('NSE', float('nan')):.3f} "
                f"full CC={full.get('CC', float('nan')):.3f} "
                f"full KGE={full.get('KGE', float('nan')):.3f}"
            )

            # Kick off test and next-round proposal requests in parallel and wait
            next_proposals = None
            next_refined = None
            next_params = None
            next_eval_meta = {}

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                if self.test_config and self.test_config.enabled:
                    futures["test"] = executor.submit(self._run_test_suite, refined_params, r)
                if r < max_rounds:
                    next_context = self._build_context()
                    futures["proposal"] = executor.submit(
                        self._request_candidates, next_context, r + 1
                    )

                test_summaries = {}
                for key, future in futures.items():
                    if key == "test":
                        test_summaries = future.result()
                        print(f"[Round {r}] Test summaries written for {len(test_summaries)} candidates.")
                    elif key == "proposal":
                        (next_proposals, next_refined, next_params, next_eval_meta) = future.result()

            if r >= IMPROVE_PATIENCE:
                break

            if next_params is not None:
                proposals = next_proposals
                refined_candidates = next_refined
                refined_params = next_params
                eval_meta = next_eval_meta
            else:
                break

    def _update_metric_bests(self,
                             outcomes: Sequence[CandidateOutcome],
                             round_index: int) -> None:
        for outcome in outcomes:
            agg = outcome.aggregate_metrics
            full = outcome.full_metrics
            score = self._candidate_score(agg, full)
            if self.history.update_best_metric(
                key="score",
                value=score,
                aggregate_metrics=agg,
                full_metrics=full,
                params=outcome.params.values.copy(),
                round_index=round_index,
                candidate_index=outcome.simulation.candidate_index,
            ):
                self._ensure_plots(outcome)
                self._publish_best(
                    outcome,
                    subfolder="best",
                    metadata={"criterion": "score", "criterion_value": score},
                )

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


__all__ = ["TwoStageCalibrationManager", "CandidateOutcome"]
