"""Parameter utilities for calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

from .config import FROZEN_PARAMETERS, PARAM_BOUNDS


def safe_clip(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, value))


def apply_step_guard(name: str, old: float, new: float) -> float:
    return safe_clip(name, new)


@dataclass
class ParameterSet:
    """Container for model parameters."""

    values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_object(cls, obj: object) -> "ParameterSet":
        vals = {name: float(getattr(obj, name)) for name in PARAM_BOUNDS if hasattr(obj, name)}
        return cls(vals)

    def copy(self) -> "ParameterSet":
        return ParameterSet(self.values.copy())

    def apply_updates(self, updates: Dict[str, float]) -> "ParameterSet":
        new_vals = self.values.copy()
        for k, v in updates.items():
            if k not in new_vals or k in FROZEN_PARAMETERS:
                continue
            new_vals[k] = apply_step_guard(k, new_vals[k], float(v))
        return ParameterSet(new_vals)

    def to_object(self, obj: object) -> None:
        for k, v in self.values.items():
            if hasattr(obj, k):
                setattr(obj, k, float(v))

    def as_control_updates(self) -> Dict[str, float]:
        return self.values.copy()

    def items(self) -> Iterable:
        return self.values.items()

    def __getitem__(self, item: str) -> float:
        return self.values[item]

    def __setitem__(self, key: str, value: float) -> None:
        if key in FROZEN_PARAMETERS:
            return
        self.values[key] = safe_clip(key, float(value))

    def __repr__(self) -> str:
        return f"ParameterSet({self.values!r})"


__all__ = ["ParameterSet", "apply_step_guard", "safe_clip"]
