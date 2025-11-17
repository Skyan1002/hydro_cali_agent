"""Wrapper around the EF5 executable."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence


def run_ef5(control_file: str,
            ef5_executable: str = "./EF5/bin/ef5",
            cwd: Optional[str] = None,
            log_path: Optional[str] = None) -> subprocess.CompletedProcess:
    """Execute EF5 with the provided control file."""
    control_path = Path(control_file)
    if not control_path.exists():
        raise FileNotFoundError(f"Control file not found: {control_file}")
    try:
        if log_path:
            log_file = Path(log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("w", encoding="utf-8") as fh:
                result = subprocess.run(
                    [ef5_executable, str(control_path)],
                    check=True,
                    cwd=cwd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            result = subprocess.run(
                [ef5_executable, str(control_path)],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"EF5 executable not found at {ef5_executable}") from exc
    return result
