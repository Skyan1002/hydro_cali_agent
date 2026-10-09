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
from .zoned import is_zoned, render_zoned


CONTROL_PATTERN = re.compile(r"^(OUTPUT\s*=\s*).*$", re.MULTILINE)
STATES_PATTERN = re.compile(r"^(STATES\s*=\s*).*$", re.MULTILINE)
SCALAR_PATTERN = lambda name: re.compile(rf"^{name}\s*=\s*.*$", re.MULTILINE)
TASK_BLOCK_PATTERN = lambda task: re.compile(rf"(?s)(\[Task {re.escape(task)}\].*?)(?=\n\[Task|\Z)")


@dataclass
class SimulationResult:
    round_label: str
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

    def _apply_scalar(self, content: str, name: str, value: Optional[str]) -> str:
        if value is None:
            return content
        pattern = SCALAR_PATTERN(name)
        if pattern.search(content):
            return pattern.sub(f"{name}={value}", content)
        return content + f"\n{name}={value}\n"

    def _apply_task_scalar(self, content: str, task: str, field: str, value: Optional[str]) -> str:
        if value is None:
            return content
        block_pattern = TASK_BLOCK_PATTERN(task)
        match = block_pattern.search(content)
        if not match:
            return content

        block = match.group(1)
        updated = re.sub(rf"^{field}\s*=.*$", f"{field}={value}", block, flags=re.MULTILINE)
        if block == updated:
            updated = block + f"\n{field}={value}"

        return content[:match.start(1)] + updated + content[match.end(1):]

    def _render_control(self,
                        params: ParameterSet,
                        output_dir: str,
                        *,
                        states_dir: Optional[str] = None,
                        scalar_overrides: Optional[Dict[str, str]] = None) -> str:
        content = self._control_template
        if is_zoned(params.values.keys()):
            # 'wm@V1' style keys: edit only the gauge=V1 block of the right parameter
            # set. The legacy global substitution below would rewrite every block.
            content = render_zoned(content, params.values)
        else:
            for key, value in params.items():
                pattern = re.compile(rf"{key}=\s*[0-9.eE+-]+")
                content = pattern.sub(f"{key}={value}", content)
        if CONTROL_PATTERN.search(content):
            content = CONTROL_PATTERN.sub(rf"\1{output_dir}/", content)
        else:
            content += f"\nOUTPUT={output_dir}/\n"

        states_target = states_dir or output_dir
        if STATES_PATTERN.search(content):
            content = STATES_PATTERN.sub(rf"\1{states_target}/", content)
        else:
            content += f"\nSTATES={states_target}/\n"

        if scalar_overrides:
            overrides = dict(scalar_overrides)

            # Apply warmup-task-specific overrides without touching the main SIMU task.
            warmup_begin = overrides.pop("WARMUP_TIME_BEGIN", None)
            warmup_end = overrides.pop("WARMUP_TIME_END", None)
            warmup_state = overrides.pop("WARMUP_TIME_STATE", overrides.pop("TIME_STATE", None))
            content = self._apply_task_scalar(content, "warmup", "TIME_BEGIN", warmup_begin)
            content = self._apply_task_scalar(content, "warmup", "TIME_END", warmup_end)
            content = self._apply_task_scalar(content, "warmup", "TIME_STATE", warmup_state)

            # Apply main simulation window overrides to the Simu task only.
            main_begin = overrides.pop("TIME_BEGIN", None)
            main_end = overrides.pop("TIME_END", None)
            content = self._apply_task_scalar(content, "Simu", "TIME_BEGIN", main_begin)
            content = self._apply_task_scalar(content, "Simu", "TIME_END", main_end)

            for name, value in overrides.items():
                content = self._apply_scalar(content, name, value)
        return content

    def _write_control_file(self,
                            content: str,
                            round_label: str,
                            candidate_index: int,
                            *,
                            subfolder: Optional[str] = None) -> str:
        out_dir = self.simu_folder / "controls" / round_label / f"cand_{candidate_index:02d}"
        if subfolder:
            out_dir = out_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        control_path = out_dir / "control.txt"
        control_path.write_text(content)
        return str(control_path.resolve())

    def _output_dir(self, round_label: str, candidate_index: int, *, subfolder: Optional[str] = None) -> str:
        out_dir = self.simu_folder / "results" / round_label / f"cand_{candidate_index:02d}"
        if subfolder:
            out_dir = out_dir / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir.resolve())

    # --- node-local scratch for EF5 outputs ---------------------------------
    # /blue is a group-shared Lustre quota; each EF5 run writes ~2 MB of state
    # rasters and logs that nothing downstream reads, versus ~0.6 MB of time
    # series that everything reads. When INCA_LOCAL_SCRATCH is set (SLURM
    # scripts point it at the node's local disk), EF5 writes there and only the
    # csv/results.json are harvested back to the shared results directory.
    HARVEST = ("ts.*.csv", "results.json")

    def _scratch_dir(self, output_dir: str) -> str:
        root = os.getenv("INCA_LOCAL_SCRATCH")
        if not root:
            return output_dir
        rel = Path(output_dir).resolve().relative_to(self.simu_folder)
        local = Path(root) / self.simu_folder.name / rel
        local.mkdir(parents=True, exist_ok=True)
        return str(local)

    def _harvest(self, ef5_dir: str, output_dir: str) -> None:
        if Path(ef5_dir).resolve() == Path(output_dir).resolve():
            return
        import shutil
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for pattern in self.HARVEST:
            for src in Path(ef5_dir).glob(pattern):
                shutil.copy2(src, Path(output_dir) / src.name)

    def run(self, params: ParameterSet, round_label: str, candidate_index: int) -> SimulationResult:
        output_dir = self._output_dir(round_label, candidate_index)
        ef5_dir = self._scratch_dir(output_dir)
        control_content = self._render_control(params, ef5_dir)
        control_file = self._write_control_file(control_content, round_label, candidate_index)
        log_path = Path(ef5_dir) / "logs" / "ef5.log"
        run_ef5(control_file, self.ef5_executable, cwd=str(self.simu_folder), log_path=str(log_path))
        self._harvest(ef5_dir, output_dir)
        csv_path = self._locate_csv(output_dir)
        if csv_path is None:
            raise FileNotFoundError(f"Simulation output CSV not found in {output_dir}")
        return SimulationResult(round_label, candidate_index, params.copy(), output_dir, csv_path)

    def run_with_overrides(self,
                           params: ParameterSet,
                           round_label: str,
                           candidate_index: int,
                           *,
                           subfolder: Optional[str] = None,
                           states_dir: Optional[str] = None,
                           scalar_overrides: Optional[Dict[str, str]] = None) -> SimulationResult:
        output_dir = self._output_dir(round_label, candidate_index, subfolder=subfolder)
        ef5_dir = self._scratch_dir(output_dir)
        control_content = self._render_control(
            params,
            ef5_dir,
            states_dir=states_dir,
            scalar_overrides=scalar_overrides,
        )
        control_file = self._write_control_file(
            control_content,
            round_label,
            candidate_index,
            subfolder=subfolder,
        )
        log_path = Path(ef5_dir) / "logs" / "ef5.log"
        run_ef5(control_file, self.ef5_executable, cwd=str(self.simu_folder), log_path=str(log_path))
        self._harvest(ef5_dir, output_dir)
        csv_path = self._locate_csv(output_dir)
        if csv_path is None:
            raise FileNotFoundError(f"Simulation output CSV not found in {output_dir}")
        return SimulationResult(round_label, candidate_index, params.copy(), output_dir, csv_path)

    def discard(self, round_label: str, candidate_index: int, *, subfolder: Optional[str] = None) -> None:
        """Delete one evaluation's outputs, control file and node-local scratch.

        A 2000-evaluation run keeps every evaluation on disk until the job ends, which is
        pointless -- only the incumbent's series is ever read again -- and it is what makes
        a run vulnerable to somebody else filling the node's /tmp halfway through. Paths are
        rebuilt from the same helpers that created them and every removal is confined to
        ``simu_folder`` (or the configured scratch root), so a wrong label deletes nothing.
        """
        import shutil

        rel = Path(round_label) / f"cand_{candidate_index:02d}"
        if subfolder:
            rel = rel / subfolder
        roots = [self.simu_folder / "results" / rel, self.simu_folder / "controls" / rel]
        scratch_root = os.getenv("INCA_LOCAL_SCRATCH")
        if scratch_root:
            roots.append(Path(scratch_root) / self.simu_folder.name / "results" / rel)
        for path in roots:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not any(str(resolved).startswith(str(base)) for base in
                       (self.simu_folder.resolve(), Path(scratch_root).resolve() if scratch_root else self.simu_folder.resolve())):
                continue
            shutil.rmtree(resolved, ignore_errors=True)

    def _locate_csv(self, output_dir: str) -> Optional[str]:
        expected = Path(output_dir) / f"ts.{self.gauge_num}.crest.csv"
        if expected.exists():
            return str(expected)
        for fn in Path(output_dir).glob("ts.*.csv"):
            return str(fn)
        return None


def run_simulations_parallel(runner: SimulationRunner,
                             params_list: Sequence[ParameterSet],
                             round_label: str,
                             max_workers: Optional[int] = None) -> List[SimulationResult]:
    results: List[SimulationResult] = []
    worker_count = max_workers or min(len(params_list), os.cpu_count() or len(params_list))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(runner.run, params, round_label, idx): idx
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
