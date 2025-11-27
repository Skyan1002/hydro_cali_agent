"""SCE-UA benchmark runner aligned with the existing calibration framework."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    DEFAULT_GAUGE_NUM,
    DEFAULT_PEAK_PICK_KWARGS,
    FROZEN_PARAMETERS,
    PARAM_BOUNDS,
)
from .metrics import aggregate_event_metrics, compute_event_metrics, read_metrics_from_csv
from .parameters import ParameterSet, safe_clip
from .peak_events import pick_peak_events
from .simulation import SimulationRunner


@dataclass
class SceEvaluation:
    """Book-keeping for a single simulation call."""

    iteration: int
    candidate_index: int
    params: Dict[str, float]
    aggregate_metrics: Dict[str, float]
    full_metrics: Dict[str, float]
    score: float
    elapsed_seconds: float
    output_dir: str


@dataclass
class SceHistory:
    """Persisted benchmark metadata and evolution traces."""

    config: Dict[str, float]
    evaluations: List[SceEvaluation] = field(default_factory=list)
    best_index: Optional[int] = None

    def add(self, evaluation: SceEvaluation) -> None:
        self.evaluations.append(evaluation)
        if self.best_index is None:
            self.best_index = 0
        else:
            best = self.evaluations[self.best_index]
            if evaluation.score > best.score:
                self.best_index = len(self.evaluations) - 1

    def best(self) -> Optional[SceEvaluation]:
        if self.best_index is None:
            return None
        return self.evaluations[self.best_index]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "best_index": self.best_index,
            "evaluations": [
                {
                    "iteration": e.iteration,
                    "candidate_index": e.candidate_index,
                    "params": e.params,
                    "aggregate_metrics": e.aggregate_metrics,
                    "full_metrics": e.full_metrics,
                    "score": e.score,
                    "elapsed_seconds": e.elapsed_seconds,
                    "output_dir": e.output_dir,
                }
                for e in self.evaluations
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class SceUaOptimizer:
    """Shuffled Complex Evolution (SCE-UA) optimizer for EF5 calibration."""

    def __init__(
        self,
        runner: SimulationRunner,
        base_params: ParameterSet,
        gauge_num: str = DEFAULT_GAUGE_NUM,
        n_peaks: int = 3,
        peak_pick_kwargs: Optional[Dict] = None,
        criterion: str = "NSE",
        complexes: int = 5,
        complex_size: int = 8,
        subset_size: Optional[int] = None,
        evolutions_per_complex: int = 1,
        reflection_coef: float = 1.3,
        contraction_coef: float = 0.6,
        rng: Optional[np.random.Generator] = None,
    ):
        self.runner = runner
        self.base_params = base_params
        self.gauge_num = gauge_num
        self.n_peaks = n_peaks
        self.peak_pick_kwargs = peak_pick_kwargs or DEFAULT_PEAK_PICK_KWARGS
        self.criterion = criterion
        self.complexes = max(1, complexes)
        self.complex_size = max(3, complex_size)
        self.subset_size = subset_size
        self.evolutions_per_complex = max(1, evolutions_per_complex)
        self.reflection_coef = reflection_coef
        self.contraction_coef = contraction_coef
        self.rng = rng or np.random.default_rng()

        self.var_names = [name for name in PARAM_BOUNDS if name not in FROZEN_PARAMETERS]
        self.lower = np.array([PARAM_BOUNDS[n][0] for n in self.var_names], dtype=float)
        self.upper = np.array([PARAM_BOUNDS[n][1] for n in self.var_names], dtype=float)

    def _score(self, aggregate_metrics: Dict[str, float], full_metrics: Dict[str, float]) -> float:
        """Higher score is better."""
        metric = aggregate_metrics.get(self.criterion)
        if metric is None or not math.isfinite(metric):
            metric = full_metrics.get(self.criterion, float("nan"))
        return float(metric) if math.isfinite(metric) else float("nan")

    def _build_params(self, vector: np.ndarray) -> ParameterSet:
        values = self.base_params.values.copy()
        for name, val in zip(self.var_names, vector):
            values[name] = safe_clip(name, float(val))
        return ParameterSet(values)

    def _evaluate(self, params: ParameterSet, iteration: int, candidate_index: int) -> SceEvaluation:
        start = time.time()
        simulation = self.runner.run(params, round_index=iteration, candidate_index=candidate_index)
        windows = pick_peak_events(simulation.csv_path, n=self.n_peaks, **self.peak_pick_kwargs)
        event_metrics = compute_event_metrics(simulation.csv_path, windows)
        aggregate = aggregate_event_metrics(event_metrics, top_n=self.n_peaks)
        full_metrics = read_metrics_from_csv(simulation.csv_path)
        score = self._score(aggregate, full_metrics)
        elapsed = time.time() - start
        return SceEvaluation(
            iteration=iteration,
            candidate_index=candidate_index,
            params=params.values.copy(),
            aggregate_metrics=aggregate,
            full_metrics=full_metrics,
            score=score,
            elapsed_seconds=elapsed,
            output_dir=simulation.output_dir,
        )

    def _initial_population(self, population_size: int) -> np.ndarray:
        pop = self.rng.uniform(self.lower, self.upper, size=(population_size, len(self.var_names)))
        seed = np.array(
            [
                self.base_params.values.get(name, (lo + hi) / 2.0)
                for name, lo, hi in zip(self.var_names, self.lower, self.upper)
            ]
        )
        pop[0, :] = seed
        return pop

    def run(
        self,
        max_iterations: int = 30,
        max_evaluations: int = 200,
        time_limit_seconds: float = 3600.0,
    ) -> SceHistory:
        population_size = max(self.complexes * self.complex_size, len(self.var_names) + 1)
        subset = self.subset_size or min(len(self.var_names) + 1, self.complex_size)
        population = self._initial_population(population_size)
        scores = np.full(population_size, np.nan, dtype=float)

        history = SceHistory(
            config={
                "criterion": self.criterion,
                "complexes": self.complexes,
                "complex_size": self.complex_size,
                "subset_size": subset,
                "evolutions_per_complex": self.evolutions_per_complex,
                "reflection_coef": self.reflection_coef,
                "contraction_coef": self.contraction_coef,
                "max_iterations": max_iterations,
                "max_evaluations": max_evaluations,
                "time_limit_seconds": time_limit_seconds,
            }
        )

        eval_count = 0
        candidate_index = 0
        start_time = time.time()

        # Evaluate initial population
        for idx in range(population_size):
            params = self._build_params(population[idx])
            evaluation = self._evaluate(params, iteration=0, candidate_index=candidate_index)
            history.add(evaluation)
            scores[idx] = evaluation.score
            eval_count += 1
            candidate_index += 1
            if eval_count >= max_evaluations:
                return history

        iteration = 1
        while iteration <= max_iterations and eval_count < max_evaluations:
            if (time.time() - start_time) >= time_limit_seconds:
                break

            order = np.argsort(-scores)
            population = population[order]
            scores = scores[order]

            # Shuffle into complexes
            complexes = [list(range(i, population_size, self.complexes)) for i in range(self.complexes)]

            for complex_id, complex_indices in enumerate(complexes):
                for _ in range(self.evolutions_per_complex):
                    subset_indices = self._sample_subset(complex_indices, subset)
                    new_point, replaced_idx, best_vector = self._evolve_subset(
                        population, scores, subset_indices
                    )
                    previous_score = scores[replaced_idx]
                    population[replaced_idx] = new_point
                    params = self._build_params(new_point)
                    evaluation = self._evaluate(params, iteration=iteration, candidate_index=candidate_index)
                    history.add(evaluation)
                    current_score = evaluation.score
                    scores[replaced_idx] = current_score
                    eval_count += 1
                    candidate_index += 1

                    needs_contraction = (
                        not math.isfinite(current_score)
                        or not math.isfinite(previous_score)
                        or current_score <= previous_score
                    )
                    if needs_contraction and eval_count < max_evaluations:
                        contracted_point = self._contract_point(best_vector, new_point)
                        population[replaced_idx] = contracted_point
                        contracted_params = self._build_params(contracted_point)
                        contracted_eval = self._evaluate(
                            contracted_params, iteration=iteration, candidate_index=candidate_index
                        )
                        history.add(contracted_eval)
                        scores[replaced_idx] = contracted_eval.score
                        eval_count += 1
                        candidate_index += 1

                    if eval_count >= max_evaluations or (time.time() - start_time) >= time_limit_seconds:
                        break
                if eval_count >= max_evaluations or (time.time() - start_time) >= time_limit_seconds:
                    break
            iteration += 1
        return history

    def _sample_subset(self, complex_indices: Sequence[int], subset_size: int) -> List[int]:
        probs = np.linspace(len(complex_indices), 1, num=len(complex_indices), dtype=float)
        probs /= probs.sum()
        subset_size = min(subset_size, len(complex_indices))
        return list(self.rng.choice(complex_indices, size=subset_size, replace=False, p=probs))

    def _evolve_subset(
        self,
        population: np.ndarray,
        scores: np.ndarray,
        subset_indices: Sequence[int],
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        subset = population[list(subset_indices)]
        subset_scores = scores[list(subset_indices)]
        order = np.argsort(-subset_scores)
        worst_local = order[-1]
        worst_global_idx = subset_indices[worst_local]
        best_vectors = subset[order[:-1]]
        centroid = np.mean(best_vectors, axis=0)
        worst = subset[worst_local]

        # Reflection
        candidate = centroid + self.reflection_coef * (centroid - worst)
        candidate = np.clip(candidate, self.lower, self.upper)
        if not np.all(np.isfinite(candidate)):
            candidate = self.rng.uniform(self.lower, self.upper)

        return candidate, worst_global_idx, best_vectors[0]

    def _contract_point(self, best: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        contracted = best + self.contraction_coef * (candidate - best)
        contracted = np.clip(contracted, self.lower, self.upper)
        if not np.all(np.isfinite(contracted)):
            contracted = self.rng.uniform(self.lower, self.upper)
        return contracted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SCE-UA benchmark calibration using the EF5 runner",
        fromfile_prefix_chars="@",
    )

    parser.add_argument("--site_num", required=True, help="USGS gauge id, e.g., 08069000")
    parser.add_argument("--basic_data_path", required=True,
                        help="Folder containing dem_usa.tif, fdir_usa.tif, facc_usa.tif")
    parser.add_argument("--default_param_dir", required=True,
                        help="Folder containing crest_params/ and kw_params/ subfolders")
    parser.add_argument("--cali_set_dir", default="./cali_set",
                        help="Base folder to hold <site>_<tag>/control.txt")
    parser.add_argument("--cali_tag", default="2018", help="Suffix tag for the calibration folder name")

    parser.add_argument("--precip_path", required=True, help="Folder of precipitation rasters")
    parser.add_argument("--precip_name", required=True, help="File name pattern, e.g. GaugeCorr_QPE_....tif")
    parser.add_argument("--pet_path", required=True, help="Folder of PET rasters")
    parser.add_argument("--pet_name", required=True, help="File name pattern, e.g. etYYYYMMDD.tif")

    parser.add_argument("--gauge_outdir", required=True, help="Folder to save USGS hourly CSV")
    parser.add_argument("--results_outdir", required=True, help="CREST outputs folder")

    parser.add_argument("--time_begin", required=True, help="YYYYMMDDhhmm, e.g., 201801010000")
    parser.add_argument("--time_end", required=True, help="YYYYMMDDhhmm, e.g., 201812312300")
    parser.add_argument("--time_step", default="1h", help="CREST time step, e.g., 1h")

    parser.add_argument("--model", default="CREST")
    parser.add_argument("--routing", default="KW")

    parser.add_argument("--wm", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--im", type=float, default=1.0)
    parser.add_argument("--ke", type=float, default=1.0)
    parser.add_argument("--fc", type=float, default=1.0)
    parser.add_argument("--iwu", type=float, default=25.0)
    parser.add_argument("--under", type=float, default=1.0)
    parser.add_argument("--leaki", type=float, default=1.0)
    parser.add_argument("--th", type=float, default=10.0)
    parser.add_argument("--isu", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha0", type=float, default=0.0)

    parser.add_argument("--n_peaks", type=int, default=3)

    # Compatibility flags from other calibration drivers; accepted and ignored so
    # "@cali_args.txt" files can be reused without modification.
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="(compat) number of candidates per round from other drivers")
    parser.add_argument("--max_rounds", type=int, default=20,
                        help="(compat) maximum rounds from other drivers")
    parser.add_argument("--sce-criterion", dest="sce_criterion", default="NSE", choices=["NSE", "KGE", "CC"],
                        help="Metric used to rank candidates")
    parser.add_argument("--sce-complexes", dest="sce_complexes", type=int, default=5)
    parser.add_argument("--sce-complex-size", dest="sce_complex_size", type=int, default=8)
    parser.add_argument("--sce-subset-size", dest="sce_subset_size", type=int, default=None)
    parser.add_argument("--sce-evolutions", dest="sce_evolutions", type=int, default=1,
                        help="Evolution steps per complex per iteration")
    parser.add_argument("--sce-max-iter", dest="sce_max_iter", type=int, default=30)
    parser.add_argument("--sce-max-evals", dest="sce_max_evals", type=int, default=200)
    parser.add_argument("--sce-time-limit", dest="sce_time_limit", type=float, default=3600.0,
                        help="Wall-clock limit in seconds")
    parser.add_argument("--sce-seed", dest="sce_seed", type=int, default=42)
    parser.add_argument("--sce-output-tag", dest="sce_output_tag", default="sce_benchmark",
                        help="Subfolder name under results to store benchmark artifacts")

    parser.add_argument("--simu_folder", default=None,
                        help="Optional explicit simulation folder; defaults to <cali_set_dir>/<site>_<tag>")
    parser.add_argument("--gauge_num", default=DEFAULT_GAUGE_NUM,
                        help="Gauge number used for locating EF5 output CSVs")

    return parser


def save_history(history: SceHistory, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(history.to_json())


def run_cli(args: argparse.Namespace) -> Path:
    simu_folder = args.simu_folder or Path(args.cali_set_dir) / f"{args.site_num}_{args.cali_tag}"
    runner = SimulationRunner(simu_folder=str(simu_folder), gauge_num=args.gauge_num)
    base_params = ParameterSet.from_object(args)
    rng = np.random.default_rng(args.sce_seed)
    optimizer = SceUaOptimizer(
        runner=runner,
        base_params=base_params,
        gauge_num=args.gauge_num,
        n_peaks=args.n_peaks,
        criterion=args.sce_criterion,
        complexes=args.sce_complexes,
        complex_size=args.sce_complex_size,
        subset_size=args.sce_subset_size,
        evolutions_per_complex=args.sce_evolutions,
        rng=rng,
    )

    history = optimizer.run(
        max_iterations=args.sce_max_iter,
        max_evaluations=args.sce_max_evals,
        time_limit_seconds=args.sce_time_limit,
    )

    best = history.best()
    results_dir = Path(simu_folder) / "results" / args.sce_output_tag
    summary_path = results_dir / "sce_summary.json"
    history_path = results_dir / "sce_history.json"
    if best:
        summary = {
            "best_iteration": best.iteration,
            "best_candidate_index": best.candidate_index,
            "criterion": args.sce_criterion,
            "score": best.score,
            "params": best.params,
            "aggregate_metrics": best.aggregate_metrics,
            "full_metrics": best.full_metrics,
            "elapsed_seconds": best.elapsed_seconds,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    save_history(history, history_path)
    return history_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognized arguments: {unknown}")
    history_path = run_cli(args)
    print(f"SCE-UA benchmark complete. History saved to {history_path}")


if __name__ == "__main__":
    main()
