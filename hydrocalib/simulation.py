"""Simulation runner utilities."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .config import DEFAULT_GAUGE_NUM, DEFAULT_SIM_FOLDER
from .ef5_runner import run_ef5
from .parameters import ParameterSet


CONTROL_PATTERN = re.compile(r"^(OUTPUT\s*=\s*).*$", re.MULTILINE)
STATES_PATTERN = re.compile(r"^(STATES\s*=\s*).*$", re.MULTILINE)


@dataclass
class SimulationResult:
    round_index: int
    candidate_index: int
    params: ParameterSet
    output_dir: str
    csv_path: str


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class SimulationRunner:
    """Prepare control files and execute EF5 simulations."""

    def __init__(self,
                 simu_folder: str = DEFAULT_SIM_FOLDER,
                 ef5_executable: str = "./EF5/bin/ef5",
                 gauge_num: str = DEFAULT_GAUGE_NUM):
        self.simu_folder = Path(simu_folder).resolve()
        self.ef5_executable = self._resolve_executable(ef5_executable)
        self.gauge_num = gauge_num
        self._control_template = self._load_template()

    def _resolve_executable(self, exe: str) -> str:
        env_override = os.getenv("EF5_EXECUTABLE")
        candidates = []
        if env_override:
            candidates.append(Path(env_override))
        exe_path = Path(exe)
        if exe_path.is_absolute():
            candidates.append(exe_path)
        else:
            candidates.extend([
                PACKAGE_ROOT / exe_path,
                self.simu_folder / exe_path,
                Path.cwd() / exe_path,
            ])
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        return str(exe)

    def _load_template(self) -> str:
        control_path = self.simu_folder / "control.txt"
        if not control_path.exists():
            raise FileNotFoundError(f"Base control.txt not found at {control_path}")
        return control_path.read_text()

    def _render_control(self, params: ParameterSet, output_dir: str) -> str:
        content = self._control_template
        for key, value in params.items():
            pattern = re.compile(rf"{key}=\s*[0-9.eE+-]+")
            content = pattern.sub(f"{key}={value}", content)
        if CONTROL_PATTERN.search(content):
            content = CONTROL_PATTERN.sub(rf"\1{output_dir}/", content)
        else:
            content += f"\nOUTPUT={output_dir}/\n"

        if STATES_PATTERN.search(content):
            content = STATES_PATTERN.sub(rf"\1{output_dir}/", content)
        else:
            content += f"\nSTATES={output_dir}/\n"
        return content

    def _write_control_file(self, content: str, round_index: int, candidate_index: int) -> str:
        out_dir = self.simu_folder / "controls" / f"cali_{round_index:03d}" / f"cand_{candidate_index:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        control_path = out_dir / "control.txt"
        control_path.write_text(content)
        return str(control_path.resolve())

    def _output_dir(self, round_index: int, candidate_index: int) -> str:
        out_dir = self.simu_folder / "results" / f"cali_{round_index:03d}" / f"cand_{candidate_index:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir.resolve())

    def run(self, params: ParameterSet, round_index: int, candidate_index: int) -> SimulationResult:
        output_dir = self._output_dir(round_index, candidate_index)
        control_content = self._render_control(params, output_dir)
        control_file = self._write_control_file(control_content, round_index, candidate_index)
        log_path = Path(output_dir) / "logs" / "ef5.log"
        run_ef5(control_file, self.ef5_executable, cwd=str(self.simu_folder), log_path=str(log_path))
        csv_path = self._locate_csv(output_dir)
        if csv_path is None:
            raise FileNotFoundError(f"Simulation output CSV not found in {output_dir}")
        return SimulationResult(round_index, candidate_index, params.copy(), output_dir, csv_path)

    def _locate_csv(self, output_dir: str) -> Optional[str]:
        expected = Path(output_dir) / f"ts.{self.gauge_num}.crest.csv"
        if expected.exists():
            return str(expected)
        for fn in Path(output_dir).glob("ts.*.csv"):
            return str(fn)
        return None


def run_simulations_parallel(runner: SimulationRunner,
                             params_list: Sequence[ParameterSet],
                             round_index: int,
                             max_workers: Optional[int] = None) -> List[SimulationResult]:
    results: List[SimulationResult] = []
    worker_count = max_workers or min(len(params_list), os.cpu_count() or len(params_list))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(runner.run, params, round_index, idx): idx
            for idx, params in enumerate(params_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            print(f"    [Sim] Candidate {idx} completed EF5 run.")
            results.append(result)
    results.sort(key=lambda r: r.candidate_index)
    return results


__all__ = ["SimulationRunner", "SimulationResult", "run_simulations_parallel"]
