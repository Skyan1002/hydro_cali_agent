#!/usr/bin/env python3
"""Run an SCE-UA calibration benchmark aligned with the agent framework."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from hydrocalib.agents.sceua import SceUaCalibrator, SceUaConfig
from hydrocalib.config import DEFAULT_GAUGE_NUM
from hydrocalib.parameters import ParameterSet
from hydrocalib.simulation import SimulationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCE-UA benchmark runner using the same EF5 plumbing as the agent",
        fromfile_prefix_chars="@",
    )

    # Core calibration inputs (kept aligned with hydro_cali_main for @cali_args.txt)
    parser.add_argument("--site_num", required=True, help="USGS gauge id, e.g., 08069000")
    parser.add_argument("--basic_data_path", required=True, help="DEM/FDR/FACC folder (not used directly here)")
    parser.add_argument("--default_param_dir", required=True, help="Parameter grids folder (not used directly here)")
    parser.add_argument("--cali_set_dir", default="./cali_set", help="Base folder to hold <site>_<tag>/control.txt")
    parser.add_argument("--cali_tag", default="2018", help="Suffix tag for the calibration folder name")
    parser.add_argument("--precip_path", required=True, help="Folder of precipitation rasters")
    parser.add_argument("--precip_name", required=True, help="File name pattern for precipitation")
    parser.add_argument("--pet_path", required=True, help="Folder of PET rasters")
    parser.add_argument("--pet_name", required=True, help="File name pattern for PET")
    parser.add_argument("--gauge_outdir", required=True, help="Where USGS CSV is stored (for consistency)")
    parser.add_argument("--results_outdir", required=True, help="CREST outputs folder recorded in control.txt")
    parser.add_argument("--time_begin", required=True, help="Simulation start (YYYYMMDDhhmm)")
    parser.add_argument("--time_end", required=True, help="Simulation end (YYYYMMDDhhmm)")
    parser.add_argument("--time_step", default="1h", help="CREST timestep, e.g., 1h")
    parser.add_argument("--model", default="CREST")
    parser.add_argument("--routing", default="KW")

    # Parameter seeds
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

    # Benchmark controls
    parser.add_argument("--n_peaks", type=int, default=3, help="Number of peaks considered for event metrics")
    parser.add_argument("--sce_n_complexes", type=int, default=3, help="Number of complexes in the SCE-UA population")
    parser.add_argument("--sce_complex_size", type=int, default=6, help="Number of members inside each complex")
    parser.add_argument("--sce_complex_steps", type=int, default=2, help="Steps to evolve each complex per iteration")
    parser.add_argument("--sce_max_iterations", type=int, default=15, help="Total iterations to perform")
    parser.add_argument("--sce_seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--benchmark_dir",
        default=None,
        help="Optional directory to store progress/best summaries (default: <simu_folder>/results/sce_ua)",
    )
    parser.add_argument(
        "--simu_folder",
        default=None,
        help="If provided, overrides the constructed <cali_set_dir>/<site>_<tag> folder",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Reserved for future parallel EF5 runs; currently single-core for determinism",
    )

    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()
    simu_folder = Path(args.simu_folder) if args.simu_folder else Path(args.cali_set_dir) / f"{args.site_num}_{args.cali_tag}"
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else simu_folder / "results" / "sce_ua"

    initial_params = ParameterSet.from_object(args)
    config = SceUaConfig(
        n_complexes=args.sce_n_complexes,
        complex_size=args.sce_complex_size,
        max_iterations=args.sce_max_iterations,
        complex_steps=args.sce_complex_steps,
        seed=args.sce_seed,
        n_peaks=args.n_peaks,
        benchmark_dir=benchmark_dir,
        max_workers=args.max_workers,
    )

    runner = SimulationRunner(simu_folder=str(simu_folder), gauge_num=args.site_num or DEFAULT_GAUGE_NUM)
    calibrator = SceUaCalibrator(runner=runner, initial_params=initial_params, config=config, gauge_num=args.site_num)

    total_start = time.time()
    print(f"[SCE-UA] Starting benchmark in {simu_folder} with population {config.population_size()} …")
    best_eval = calibrator.calibrate()
    calibrator.finalize()
    total_elapsed = time.time() - total_start

    summary = {
        "site_num": args.site_num,
        "simu_folder": str(simu_folder),
        "benchmark_dir": str(benchmark_dir),
        "total_runtime_seconds": total_elapsed,
        "evaluations": calibrator.eval_counter,
        "best_score": best_eval.score,
        "best_aggregate_metrics": best_eval.outcome.aggregate_metrics,
        "best_full_metrics": best_eval.outcome.full_metrics,
        "best_params": best_eval.params.values,
        "config": {
            "n_complexes": config.n_complexes,
            "complex_size": config.complex_size,
            "max_iterations": config.max_iterations,
            "complex_steps": config.complex_steps,
            "seed": config.seed,
            "n_peaks": config.n_peaks,
        },
    }

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / "sce_ua_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[SCE-UA] Finished in {total_elapsed:.2f}s after {calibrator.eval_counter} evaluations.")
    print(f"[SCE-UA] Best NSE={best_eval.outcome.aggregate_metrics.get('NSE', float('nan')):.3f} | score={best_eval.score:.3f}")
    print(f"[SCE-UA] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
