"""SCE-UA benchmark runner aligned with the existing calibration framework."""
from __future__ import annotations

import argparse
import json
import math
import time
import shutil
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
        print(
            f"[iter {iteration} cand {candidate_index}] starting simulation with params: {params.values}",
            flush=True,
        )
        simulation = self.runner.run(params, round_index=iteration, candidate_index=candidate_index)
        windows = pick_peak_events(simulation.csv_path, n=self.n_peaks, **self.peak_pick_kwargs)
        event_metrics = compute_event_metrics(simulation.csv_path, windows)
        aggregate = aggregate_event_metrics(event_metrics, top_n=self.n_peaks)
        full_metrics = read_metrics_from_csv(simulation.csv_path)
        score = self._score(aggregate, full_metrics)
        elapsed = time.time() - start
        print(
            f"[iter {iteration} cand {candidate_index}] completed in {elapsed:.1f}s "
            f"score={score} output={simulation.output_dir}",
            flush=True,
        )
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

        print(
            "Starting SCE-UA run with configuration:",
            json.dumps(history.config, indent=2, ensure_ascii=False),
            flush=True,
        )
        print(
            f"Population size={population_size}, complexes={self.complexes}, "
            f"complex size={self.complex_size}, subset size={subset}",
            flush=True,
        )

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

            print(
                f"=== Iteration {iteration} (evaluations so far: {eval_count}/{max_evaluations}) ===",
                flush=True,
            )
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

    parser.add_argument("--time_begin", default="201801010000",
                        help="YYYYMMDDhhmm, e.g., 201801010000")
    parser.add_argument("--time_end", default="201801051230",
                        help="YYYYMMDDhhmm, e.g., 201812312300")
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


def _prepare_simu_folder(args: argparse.Namespace) -> Path:
    """Prepare an isolated SCE simulation folder.

    By default we mirror the existing <site>_<tag> folder but append a `_sce`
    suffix to avoid mixing artifacts with other calibration runs. If the base
    folder exists, copy its contents so control.txt and data files are
    available; otherwise, create the destination and let downstream checks
    surface any missing inputs.
    """

    if args.simu_folder:
        return Path(args.simu_folder)

    base = Path(args.cali_set_dir) / f"{args.site_num}_{args.cali_tag}"
    sce_folder = base.with_name(f"{base.name}_sce")

    if sce_folder.exists():
        print(f"Using existing SCE simulation folder: {sce_folder}", flush=True)
        return sce_folder

    if base.exists():
        print(f"Creating SCE simulation folder {sce_folder} by copying {base}", flush=True)
        shutil.copytree(base, sce_folder, dirs_exist_ok=True)
    else:
        print(
            f"Base folder {base} not found; creating empty SCE folder {sce_folder}. "
            "Ensure control.txt and inputs exist before running.",
            flush=True,
        )
        sce_folder.mkdir(parents=True, exist_ok=True)
    return sce_folder


def _ensure_control_file(sce_folder: Path, args: argparse.Namespace) -> None:
    """Guarantee control.txt exists in the SCE folder.

    Priority:
    1) Reuse an existing control.txt under the SCE folder.
    2) Copy control.txt from the base <site>_<tag> folder if present.
    3) Attempt to synthesize control.txt using the same helpers as
       ``hydro_cali_main.py`` so SCE benchmarks can run standalone.
    """

    control_path = sce_folder / "control.txt"
    if control_path.exists():
        return

    base_folder = Path(args.cali_set_dir) / f"{args.site_num}_{args.cali_tag}"
    base_control = base_folder / "control.txt"
    if base_control.exists():
        sce_folder.mkdir(parents=True, exist_ok=True)
        print(f"Copying control.txt from {base_control} to {control_path}", flush=True)
        shutil.copy2(base_control, control_path)
        return

    try:
        from hydro_cali_main import (
            DEFAULT_TEMPLATE,
            build_control_text,
            build_obs_csv_path,
            ensure_abs_path,
            get_usgs_site_info,
        )
    except Exception as exc:  # pragma: no cover - defensive import
        raise FileNotFoundError(
            f"control.txt missing in {sce_folder} and {base_folder}; "
            "could not import helpers to synthesize one."
        ) from exc

    print(
        f"control.txt not found; creating one in {control_path} using calibration arguments",
        flush=True,
    )

    # Resolve absolute paths to mirror hydro_cali_main behavior.
    basic_data_path = ensure_abs_path(args.basic_data_path)
    default_param_dir = ensure_abs_path(args.default_param_dir)
    precip_path = ensure_abs_path(args.precip_path)
    pet_path = ensure_abs_path(args.pet_path)
    gauge_outdir = ensure_abs_path(args.gauge_outdir)
    results_outdir = ensure_abs_path(args.results_outdir)

    crest_dir = Path(default_param_dir) / "crest_params"
    kw_dir = Path(default_param_dir) / "kw_params"

    dem_path = Path(basic_data_path) / "dem_usa.tif"
    ddm_path = Path(basic_data_path) / "fdir_usa.tif"
    fam_path = Path(basic_data_path) / "facc_usa.tif"

    site_info = get_usgs_site_info(args.site_num)

    control_text = build_control_text(
        template=DEFAULT_TEMPLATE,
        site_no=args.site_num,
        lon=site_info.longitude,
        lat=site_info.latitude,
        basin_km2=site_info.drainage_area_km2,
        DEM_PATH=str(dem_path),
        DDM_PATH=str(ddm_path),
        FAM_PATH=str(fam_path),
        PRECIP_PATH=precip_path,
        PRECIP_NAME=args.precip_name,
        PET_PATH=pet_path,
        PET_NAME=args.pet_name,
        OBS_PATH=build_obs_csv_path(gauge_outdir, args.site_num),
        WM_GRID=str(crest_dir / "wm_usa.tif"),
        IM_GRID=str(crest_dir / "im_usa.tif"),
        FC_GRID=str(crest_dir / "ksat_usa.tif"),
        B_GRID=str(crest_dir / "b_usa.tif"),
        WM=args.wm,
        B=args.b,
        IM=args.im,
        KE=args.ke,
        FC=args.fc,
        IWU=args.iwu,
        LEAKI_GRID=str(kw_dir / "leaki_usa.tif"),
        ALPHA_GRID=str(kw_dir / "alpha_usa.tif"),
        BETA_GRID=str(kw_dir / "beta_usa.tif"),
        ALPHA0_GRID=str(kw_dir / "alpha0_usa.tif"),
        UNDER=args.under,
        LEAKI=args.leaki,
        TH=args.th,
        ISU=args.isu,
        ALPHA=args.alpha,
        BETA=args.beta,
        ALPHA0=args.alpha0,
        MODEL=args.model,
        ROUTING=args.routing,
        RESULTS_OUTDIR=results_outdir,
        TIME_STEP=args.time_step,
        TIME_BEGIN=args.time_begin,
        TIME_END=args.time_end,
    )

    sce_folder.mkdir(parents=True, exist_ok=True)
    control_path.write_text(control_text)
    print(f"Wrote synthesized control.txt to {control_path}", flush=True)


def _apply_time_window_to_control(sce_folder: Path, args: argparse.Namespace) -> None:
    """Rewrite TIME_BEGIN/TIME_END in control.txt to match CLI arguments."""

    control_path = sce_folder / "control.txt"
    if not control_path.exists():
        raise FileNotFoundError(f"control.txt not found in {sce_folder}; cannot set time window")

    original = control_path.read_text().splitlines()
    updated = []
    replaced_begin = False
    replaced_end = False

    for line in original:
        if line.startswith("TIME_BEGIN="):
            updated.append(f"TIME_BEGIN={args.time_begin}")
            replaced_begin = True
        elif line.startswith("TIME_END="):
            updated.append(f"TIME_END={args.time_end}")
            replaced_end = True
        else:
            updated.append(line)

    if not replaced_begin:
        updated.append(f"TIME_BEGIN={args.time_begin}")
    if not replaced_end:
        updated.append(f"TIME_END={args.time_end}")

    if updated != original:
        control_path.write_text("\n".join(updated) + "\n")
        print(
            f"Updated control.txt time window -> {args.time_begin} to {args.time_end} at {control_path}",
            flush=True,
        )


def run_cli(args: argparse.Namespace) -> Path:
    simu_folder = _prepare_simu_folder(args)
    _ensure_control_file(simu_folder, args)
    _apply_time_window_to_control(simu_folder, args)
    runner = SimulationRunner(simu_folder=str(simu_folder), gauge_num=args.gauge_num)
    base_params = ParameterSet.from_object(args)
    rng = np.random.default_rng(args.sce_seed)

    print("=== Resolved benchmark inputs ===", flush=True)
    print(f"site_num: {args.site_num}", flush=True)
    print(f"simu_folder: {simu_folder}", flush=True)
    print(f"gauge_num: {args.gauge_num}", flush=True)
    print(f"time window: {args.time_begin} -> {args.time_end} (step {args.time_step})", flush=True)
    print("model/routing:", args.model, args.routing, flush=True)
    print("paths:", flush=True)
    print(f"  basic_data_path: {args.basic_data_path}", flush=True)
    print(f"  default_param_dir: {args.default_param_dir}", flush=True)
    print(f"  precip_path/name: {args.precip_path} / {args.precip_name}", flush=True)
    print(f"  pet_path/name: {args.pet_path} / {args.pet_name}", flush=True)
    print(f"  gauge_outdir: {args.gauge_outdir}", flush=True)
    print(f"  results_outdir: {args.results_outdir}", flush=True)
    print("parameters:", json.dumps(base_params.values, indent=2, ensure_ascii=False), flush=True)
    print(
        "SCE config:",
        json.dumps(
            {
                "criterion": args.sce_criterion,
                "complexes": args.sce_complexes,
                "complex_size": args.sce_complex_size,
                "subset_size": args.sce_subset_size,
                "evolutions": args.sce_evolutions,
                "max_iter": args.sce_max_iter,
                "max_evals": args.sce_max_evals,
                "time_limit": args.sce_time_limit,
                "seed": args.sce_seed,
                "n_peaks": args.n_peaks,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(
        f"Working folder: {simu_folder}\n"
        f"Results will be stored under: {simu_folder / 'results' / args.sce_output_tag}",
        flush=True,
    )
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
