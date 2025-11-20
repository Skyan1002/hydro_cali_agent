"""Wrapper around the EF5 executable."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence


def run_ef5(control_file: str,
            ef5_executable: str = "./EF5/bin/ef5",
            cwd: Optional[str] = None,
            log_path: Optional[str] = None) -> subprocess.CompletedProcess:
    """Execute EF5 with the provided control file.

    The EF5 binary path is resolved so relative inputs still work even when
    ``cwd`` is set for the subprocess.
    """
    control_path = Path(control_file)
    if not control_path.exists():
        raise FileNotFoundError(f"Control file not found: {control_file}")

    ef5_path = Path(ef5_executable)
    attempted_paths = []
    if not ef5_path.is_absolute():
        if cwd:
            candidate = Path(cwd) / ef5_path
            attempted_paths.append(candidate)
            if candidate.exists():
                ef5_path = candidate
        if not ef5_path.is_absolute() or not ef5_path.exists():
            candidate = ef5_path.resolve()
            attempted_paths.append(candidate)
            ef5_path = candidate

    if not ef5_path.exists():
        attempted = ", ".join(str(p) for p in attempted_paths) or str(ef5_executable)
        raise FileNotFoundError(f"EF5 executable not found. Tried: {attempted}")

    try:
        if log_path:
            log_file = Path(log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("w", encoding="utf-8") as fh:
                result = subprocess.run(
                    [str(ef5_path), str(control_path)],
                    check=True,
                    cwd=cwd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            result = subprocess.run(
                [str(ef5_path), str(control_path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"EF5 executable not found at {ef5_path}") from exc
    return result
