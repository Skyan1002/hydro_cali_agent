"""SCE-UA style calibration benchmark.

This module mirrors the agent-oriented calibration stack but runs a
standalone Shuffled Complex Evolution (SCE-UA) search. It uses the
same scoring logic as the LLM-driven manager and produces iterative
summaries so we can benchmark runtime, iteration counts, and skill.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import DEFAULT_GAUGE_NUM, DEFAULT_PEAK_PICK_KWARGS, FROZEN_PARAMETERS, PARAM_BOUNDS
from ..metrics import aggregate_event_metrics, compute_event_metrics, read_metrics_from_csv
from ..parameters import ParameterSet, safe_clip
from ..peak_events import pick_peak_events
from ..simulation import SimulationResult, SimulationRunner


@dataclass
class CandidateEvaluation:
    params: ParameterSet
    outcome: "CandidateOutcome"
    score: float
    runtime_seconds: float


@dataclass
class IterationReport:
    iteration: int
    best_score: float
    best_metrics: Dict[str, float]
    best_full_metrics: Dict[str, float]
    best_params: Dict[str, float]
    elapsed_seconds: float
    evaluations: int


@dataclass
class CandidateOutcome:
    simulation: SimulationResult
    windows: List
    event_metrics: List[Dict[str, float]]
    aggregate_metrics: Dict[str, float]
    full_metrics: Dict[str, float]


@dataclass
class SceUaConfig:
    n_complexes: int = 3
    complex_size: int = 6
    max_iterations: int = 15
    complex_steps: int = 2
    shuffle_period: int = 1
    seed: Optional[int] = None
    n_peaks: int = 3
    peak_pick_kwargs: Dict = field(default_factory=lambda: DEFAULT_PEAK_PICK_KWARGS.copy())
    benchmark_dir: Optional[Path] = None
    max_workers: Optional[int] = None

    def population_size(self) -> int:
        return self.n_complexes * self.complex_size


class SceUaCalibrator:
    """Minimal SCE-UA implementation aligned with the agent stack."""

    def __init__(self,
                 runner: SimulationRunner,
                 initial_params: ParameterSet,
                 config: SceUaConfig,
                 gauge_num: str = DEFAULT_GAUGE_NUM):
        self.runner = runner
        self.base_params = initial_params
        self.config = config
        self.gauge_num = gauge_num
        self.random = random.Random(config.seed)
        self.free_names = [n for n in PARAM_BOUNDS if n not in FROZEN_PARAMETERS]
        self.eval_counter = 0
        self.best_evaluation: Optional[CandidateEvaluation] = None
        self.progress: List[IterationReport] = []

    # --------------------------- SCE-UA core ---------------------------
    def calibrate(self) -> CandidateEvaluation:
        population = self._initialize_population()
        pop_scores = [ev.score for ev in population]
        self.best_evaluation = population[int(np.argmax(pop_scores))]

        for iteration in range(self.config.max_iterations):
            iter_start = time.time()
            sorted_indices = sorted(range(len(population)), key=lambda i: population[i].score, reverse=True)
            complexes = [sorted_indices[i:: self.config.n_complexes] for i in range(self.config.n_complexes)]

            for complex_idx, complex_indices in enumerate(complexes):
                population = self._evolve_complex(population, complex_indices, iteration, complex_idx)

            # Re-evaluate best after reshuffle
            best_idx = int(np.argmax([ev.score for ev in population]))
            self.best_evaluation = population[best_idx]
            iter_elapsed = time.time() - iter_start
            self._record_iteration(iteration, iter_elapsed)

        return self.best_evaluation

    # --------------------------- Helpers ---------------------------
    def _initialize_population(self) -> List[CandidateEvaluation]:
        population: List[CandidateEvaluation] = []
        # Keep the provided starting point
        population.append(self._evaluate(self.base_params, round_index=0))

        for i in range(1, self.config.population_size()):
            params = self._random_params()
            population.append(self._evaluate(params, round_index=0))
        return population

    def _random_params(self) -> ParameterSet:
        vals = self.base_params.values.copy()
        for name in self.free_names:
            low, high = PARAM_BOUNDS[name]
            vals[name] = self.random.uniform(low, high)
        return ParameterSet(vals)

    def _params_to_vector(self, params: ParameterSet) -> np.ndarray:
        return np.array([params.values.get(name, 0.0) for name in self.free_names], dtype=float)

    def _vector_to_params(self, vector: Sequence[float]) -> ParameterSet:
        vals = self.base_params.values.copy()
        for name, val in zip(self.free_names, vector):
            vals[name] = safe_clip(name, float(val))
        return ParameterSet(vals)

    def _evolve_complex(self,
                        population: List[CandidateEvaluation],
                        indices: Sequence[int],
                        iteration: int,
                        complex_idx: int) -> List[CandidateEvaluation]:
        if len(indices) < 2:
            return population
        subset = sorted(indices, key=lambda i: population[i].score, reverse=True)
        best_vecs = [self._params_to_vector(population[i].params) for i in subset[:-1]]
        worst_vec = self._params_to_vector(population[subset[-1]].params)
        centroid = np.mean(best_vecs, axis=0)

        for step in range(self.config.complex_steps):
            reflection = centroid + (centroid - worst_vec)
            new_params = self._vector_to_params(reflection)
            new_eval = self._evaluate(new_params, round_index=iteration + 1)

            if new_eval.score > population[subset[-1]].score:
                population[subset[-1]] = new_eval
            else:
                jitter = self.random.normalvariate(0, 0.1)
                mutated = centroid + jitter * (self.random.random() - 0.5)
                population[subset[-1]] = self._evaluate(self._vector_to_params(mutated), round_index=iteration + 1)

            # Update ordering for subsequent steps
            subset = sorted(indices, key=lambda i: population[i].score, reverse=True)
            best_vecs = [self._params_to_vector(population[i].params) for i in subset[:-1]]
            worst_vec = self._params_to_vector(population[subset[-1]].params)
            centroid = np.mean(best_vecs, axis=0)
        return population

    def _evaluate(self, params: ParameterSet, round_index: int) -> CandidateEvaluation:
        start = time.time()
        candidate_index = self.eval_counter
        self.eval_counter += 1
        sim_result = self.runner.run(params, round_index=round_index, candidate_index=candidate_index)
        outcome = self._process_result(sim_result)
        score = self._candidate_score(outcome.aggregate_metrics, outcome.full_metrics)
        runtime = time.time() - start
        return CandidateEvaluation(params=params.copy(), outcome=outcome, score=score, runtime_seconds=runtime)

    def _process_result(self, result: SimulationResult) -> CandidateOutcome:
        windows = pick_peak_events(result.csv_path, n=self.config.n_peaks, **self.config.peak_pick_kwargs)
        event_metrics = compute_event_metrics(result.csv_path, windows)
        aggregate = aggregate_event_metrics(event_metrics, top_n=self.config.n_peaks)
        full_metrics = read_metrics_from_csv(result.csv_path)
        return CandidateOutcome(result, windows, event_metrics, aggregate, full_metrics)

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

    def _record_iteration(self, iteration: int, elapsed: float) -> None:
        assert self.best_evaluation is not None
        report = IterationReport(
            iteration=iteration,
            best_score=self.best_evaluation.score,
            best_metrics=self.best_evaluation.outcome.aggregate_metrics,
            best_full_metrics=self.best_evaluation.outcome.full_metrics,
            best_params=self.best_evaluation.params.values.copy(),
            elapsed_seconds=elapsed,
            evaluations=self.eval_counter,
        )
        self.progress.append(report)
        if self.config.benchmark_dir:
            self._persist_iteration(report)

    def _persist_iteration(self, report: IterationReport) -> None:
        out_dir = Path(self.config.benchmark_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        history_path = out_dir / "sce_ua_progress.jsonl"
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(report)) + "\n")

    def finalize(self) -> None:
        if not self.config.benchmark_dir or not self.best_evaluation:
            return
        out_dir = Path(self.config.benchmark_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best_path = out_dir / "sce_ua_best.json"
        payload = {
            "score": self.best_evaluation.score,
            "aggregate_metrics": self.best_evaluation.outcome.aggregate_metrics,
            "full_metrics": self.best_evaluation.outcome.full_metrics,
            "params": self.best_evaluation.params.values,
            "evaluations": self.eval_counter,
        }
        best_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


__all__ = [
    "SceUaConfig",
    "SceUaCalibrator",
    "IterationReport",
    "CandidateEvaluation",
]
