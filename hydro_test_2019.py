#!/usr/bin/env python3
"""Run 2019 test simulations using metric-specific best parameters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

from hydrocalib.ef5_runner import run_ef5
from hydrocalib.metrics import read_metrics_for_period
from hydrocalib.simulation import CONTROL_PATTERN


BEST_SUBFOLDERS: Dict[str, str] = {
    "score": "best",
    "full_nse": "best_full_nse",
    "full_cc": "best_full_cc",
    "full_kge": "best_full_kge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EF5 tests for 2019 using stored best parameter sets.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--site_num", required=True, help="USGS gauge id, e.g., 08069000")
    parser.add_argument("--cali_set_dir", default="./cali_set", help="Root cali_set directory")
    parser.add_argument("--cali_tag", default="2018", help="Calibration tag that follows the site number")
    parser.add_argument("--results_tag", default="test_2019", help="Subfolder name under results for test outputs")
    parser.add_argument("--time_begin", default="201801010000", help="EF5 TIME_BEGIN for the test run")
    parser.add_argument("--time_end", default="201912312300", help="EF5 TIME_END for the test run")
    parser.add_argument("--time_step", default="1h", help="EF5 TIMESTEP override")
    parser.add_argument("--eval_start", default="2019-01-01 00:00", help="Start of evaluation window")
    parser.add_argument("--eval_end", default="2019-12-31 23:00", help="End of evaluation window")
    parser.add_argument("--ef5_executable", default="./EF5/bin/ef5", help="Path to EF5 executable")
    return parser.parse_args()


def load_template(site_dir: Path) -> str:
    control_path = site_dir / "control.txt"
    if not control_path.exists():
        raise FileNotFoundError(f"Base control.txt not found at {control_path}")
    return control_path.read_text()


def load_best_params(site_dir: Path, subfolder: str) -> Dict[str, float]:
    summary_path = site_dir / "results" / subfolder / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json for {subfolder}: {summary_path}")
    summary = json.loads(summary_path.read_text())
    params = summary.get("params") or {}
    if not params:
        raise ValueError(f"No params found in {summary_path}")
    return {k: float(v) for k, v in params.items()}


def _replace_scalar(content: str, name: str, value: str) -> str:
    pattern = re.compile(rf"{name}=\s*[^\s]+")
    if pattern.search(content):
        return pattern.sub(f"{name}={value}", content)
    return content + f"\n{name}={value}\n"


def render_control(template: str,
                   params: Dict[str, float],
                   output_dir: Path,
                   time_begin: str,
                   time_end: str,
                   time_step: str) -> str:
    content = template
    for key, value in params.items():
        pattern = re.compile(rf"{key}=\s*[0-9.eE+-]+")
        content = pattern.sub(f"{key}={value}", content)

    for name, val in {
        "TIME_BEGIN": time_begin,
        "TIME_END": time_end,
        "TIMESTEP": time_step,
    }.items():
        content = _replace_scalar(content, name, val)

    if CONTROL_PATTERN.search(content):
        content = CONTROL_PATTERN.sub(rf"\1{output_dir}/", content)
    else:
        content += f"\nOUTPUT={output_dir}/\n"

    return content


def write_control(content: str, site_dir: Path, results_tag: str, subfolder: str) -> Path:
    out_dir = site_dir / "controls" / results_tag / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    control_path = out_dir / "control.txt"
    control_path.write_text(content)
    return control_path


def locate_csv(output_dir: Path, gauge_num: str) -> Path:
    expected = output_dir / f"ts.{gauge_num}.crest.csv"
    if expected.exists():
        return expected
    csv_files = list(output_dir.glob("ts.*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No EF5 CSV found under {output_dir}")
    return csv_files[0]


def run_test_for_folder(site_dir: Path,
                        gauge_num: str,
                        template: str,
                        subfolder: str,
                        params: Dict[str, float],
                        time_begin: str,
                        time_end: str,
                        time_step: str,
                        results_tag: str,
                        ef5_executable: str,
                        eval_bounds: Tuple[str, str]) -> Dict[str, object]:
    output_dir = site_dir / "results" / results_tag / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    control_text = render_control(template, params, output_dir, time_begin, time_end, time_step)
    control_path = write_control(control_text, site_dir, results_tag, subfolder)

    log_path = output_dir / "logs" / "ef5.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_ef5(str(control_path), ef5_executable=ef5_executable, cwd=str(site_dir), log_path=str(log_path))

    csv_path = locate_csv(output_dir, gauge_num)
    metrics_2019 = read_metrics_for_period(csv_path, start=eval_bounds[0], end=eval_bounds[1])

    return {
        "control_path": str(control_path),
        "output_csv": str(csv_path),
        "params": params,
        "metrics_2019": metrics_2019,
    }


def main() -> None:
    args = parse_args()
    site_dir = Path(args.cali_set_dir) / f"{args.site_num}_{args.cali_tag}"
    template = load_template(site_dir)

    results_root = site_dir / "results" / args.results_tag
    results_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {}
    for metric_key, subfolder in BEST_SUBFOLDERS.items():
        print(f"[INFO] Running test for best {metric_key} parameters (folder={subfolder})…")
        params = load_best_params(site_dir, subfolder)
        summary[subfolder] = run_test_for_folder(
            site_dir=site_dir,
            gauge_num=args.site_num,
            template=template,
            subfolder=subfolder,
            params=params,
            time_begin=args.time_begin,
            time_end=args.time_end,
            time_step=args.time_step,
            results_tag=args.results_tag,
            ef5_executable=args.ef5_executable,
            eval_bounds=(args.eval_start, args.eval_end),
        )

    summary_path = results_root / "test_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Test results saved to {summary_path}")


if __name__ == "__main__":
    main()
