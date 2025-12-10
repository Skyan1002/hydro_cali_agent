#!/usr/bin/env python3
"""Run 2019 test simulations using metric-specific best parameters."""

from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

from hydro_cali_main import ensure_abs_path, run_usgs_downloader

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
    parser.add_argument("--gauge_outdir", default="./data_cali/gauge_18_19", help="Directory to store downloaded USGS CSV")
    parser.add_argument("--python_exec", default=sys.executable, help="Python executable for the USGS downloader")
    parser.add_argument("--usgs_script_path", default="./usgs_gauge_download.py", help="Path to the USGS download script")
    parser.add_argument("--skip_gauge_download", action="store_true", help="Skip downloading USGS observations")
    parser.add_argument("--skip_download", action="store_true", dest="skip_gauge_download", help="Deprecated alias for --skip_gauge_download")
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
                   time_step: str,
                   obs_path: Optional[str] = None) -> str:
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

    if obs_path:
        content = _replace_scalar(content, "OBS", obs_path)

    output_str = str(output_dir)
    if CONTROL_PATTERN.search(content):
        content = CONTROL_PATTERN.sub(rf"\1{output_str}/", content)
    else:
        content += f"\nOUTPUT={output_str}/\n"

    return content


def write_control(content: str, site_dir: Path, results_tag: str, subfolder: str) -> Path:
    out_dir = (site_dir / "controls" / results_tag / subfolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    control_path = (out_dir / "control.txt").resolve()
    control_path.write_text(content)
    return control_path


def find_obs_path(template: str) -> Optional[Path]:
    match = re.search(r"^OBS\s*=\s*(.+)$", template, flags=re.MULTILINE)
    if match:
        return Path(match.group(1).strip())
    return None


def resolve_obs_csv_path(template: str, site_dir: Path, gauge_outdir: Path, site_num: str) -> Path:
    from_template = find_obs_path(template)
    if from_template:
        obs_path = from_template.expanduser()
        if not obs_path.is_absolute():
            obs_path = (site_dir / obs_path).resolve()
        return obs_path
    return (gauge_outdir / f"USGS_{site_num}_1h_UTC.csv").resolve()


def ensure_gauge_data(site_num: str,
                      obs_csv_path: Path,
                      time_begin: str,
                      time_end: str,
                      time_step: str,
                      python_exec: str,
                      usgs_script_path: str,
                      skip_gauge_download: bool) -> None:
    if skip_gauge_download:
        print("[INFO] Skipping gauge download for test run.")
        return

    out_dir = obs_csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading USGS data to {out_dir} …")
    run_usgs_downloader(
        python_exec=python_exec,
        script_path=usgs_script_path,
        site_no=site_num,
        time_begin=time_begin,
        time_end=time_end,
        test_time_end=time_end,
        time_step=time_step,
        outdir=str(out_dir),
    )

    downloaded = out_dir / f"USGS_{site_num}_1h_UTC.csv"
    if downloaded.exists() and downloaded.resolve() != obs_csv_path.resolve():
        shutil.copy2(downloaded, obs_csv_path)


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
                        obs_csv_path: str,
                        ef5_executable: str,
                        eval_bounds: Tuple[str, str]) -> Dict[str, object]:
    output_dir = (site_dir / "results" / results_tag / subfolder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()

    control_text = render_control(template, params, output_dir, time_begin, time_end, time_step, obs_path=obs_csv_path)
    control_path = write_control(control_text, site_dir, results_tag, subfolder)

    log_path = (output_dir / "logs" / "ef5.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    run_ef5(str(control_path), ef5_executable=ef5_executable, cwd=str(site_dir), log_path=str(log_path.resolve()))

    csv_path = locate_csv(output_dir, gauge_num).resolve()
    metrics_2019 = read_metrics_for_period(csv_path, start=eval_bounds[0], end=eval_bounds[1])

    return {
        "control_path": str(control_path.resolve()),
        "output_csv": str(csv_path),
        "params": params,
        "metrics_2019": metrics_2019,
    }


def main() -> None:
    args = parse_args()
    site_dir = (Path(args.cali_set_dir).expanduser().resolve() / f"{args.site_num}_{args.cali_tag}")
    template = load_template(site_dir)

    gauge_outdir = Path(ensure_abs_path(args.gauge_outdir))
    obs_csv_path = resolve_obs_csv_path(template, site_dir, gauge_outdir, args.site_num)
    ensure_gauge_data(
        site_num=args.site_num,
        obs_csv_path=obs_csv_path,
        time_begin=args.time_begin,
        time_end=args.time_end,
        time_step=args.time_step,
        python_exec=args.python_exec,
        usgs_script_path=ensure_abs_path(args.usgs_script_path),
        skip_gauge_download=args.skip_gauge_download,
    )

    ef5_executable = ensure_abs_path(args.ef5_executable)

    results_root = (site_dir / "results" / args.results_tag).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=len(BEST_SUBFOLDERS)) as executor:
        for metric_key, subfolder in BEST_SUBFOLDERS.items():
            print(f"[INFO] Running test for best {metric_key} parameters (folder={subfolder})…")
            params = load_best_params(site_dir, subfolder)
            future = executor.submit(
                run_test_for_folder,
                site_dir,
                args.site_num,
                template,
                subfolder,
                params,
                args.time_begin,
                args.time_end,
                args.time_step,
                args.results_tag,
                str(obs_csv_path),
                ef5_executable,
                (args.eval_start, args.eval_end),
            )
            futures[future] = subfolder

        for future in as_completed(futures):
            subfolder = futures[future]
            summary[subfolder] = future.result()

    summary_path = (results_root / "test_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Test results saved to {summary_path}")


if __name__ == "__main__":
    main()
