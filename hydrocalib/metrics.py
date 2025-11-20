"""Metrics utilities for hydrograph calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .peak_events import _read_series
from .config import EVENTS_FOR_AGGREGATE


def safe_corrcoef(sim: np.ndarray, obs: np.ndarray) -> float:
    if sim.size < 2 or obs.size < 2:
        return float("nan")
    sim_std = float(np.std(sim))
    obs_std = float(np.std(obs))
    if (not np.isfinite(sim_std) or not np.isfinite(obs_std)
            or sim_std <= 1e-12 or obs_std <= 1e-12):
        return float("nan")
    return float(np.corrcoef(sim, obs)[0, 1])


def _kge_components(sim: np.ndarray, obs: np.ndarray) -> Tuple[float, float, float]:
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    beta = sim_mean / obs_mean if obs_mean else np.nan
    r = safe_corrcoef(sim, obs)
    sim_std = np.std(sim)
    obs_std = np.std(obs)
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean) if sim_mean and obs_mean and obs_std else np.nan
    return r, beta, gamma


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    r, beta, gamma = _kge_components(sim, obs)
    if not np.isfinite(r) or not np.isfinite(beta) or not np.isfinite(gamma):
        return float("nan")
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)


def read_metrics_from_csv(csv_path: str,
                          time_col: str = "Time",
                          sim_col: str = "Discharge(m^3 s^-1)",
                          obs_col: str = "Observed(m^3 s^-1)") -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[time_col])
    s = df[sim_col].astype(float)
    o = df[obs_col].astype(float)

    valid = ~(s.isna() | o.isna())
    t, s, o = t[valid], s[valid], o[valid]

    den = np.sum((o - np.mean(o)) ** 2)
    nse = float(1.0 - np.sum((o - s) ** 2) / den) if den > 0 else float("nan")
    cc = safe_corrcoef(s.to_numpy(), o.to_numpy())
    kge_val = kge(s.to_numpy(), o.to_numpy())

    s_peak = float(np.max(s)) if len(s) else float("nan")
    o_peak = float(np.max(o)) if len(o) else float("nan")
    peak_ratio = float(s_peak / o_peak) if (o_peak and o_peak > 0) else float("inf")

    s_tpeak = t.iloc[int(np.argmax(s.to_numpy()))] if len(s) else pd.NaT
    o_tpeak = t.iloc[int(np.argmax(o.to_numpy()))] if len(o) else pd.NaT
    lag_hours = (
        float((s_tpeak - o_tpeak).total_seconds() / 3600.0)
        if (s_tpeak is not pd.NaT and o_tpeak is not pd.NaT)
        else float("nan")
    )

    return {
        "NSE": nse,
        "CC": cc,
        "KGE": kge_val,
        "sim_peak": s_peak,
        "obs_peak": o_peak,
        "peak_ratio": peak_ratio,
        "lag_hours_sim_minus_obs": lag_hours,
    }


def read_metrics_for_period(csv_path: str,
                            start: Any,
                            end: Any,
                            time_col: str = "Time",
                            sim_col: str = "Discharge(m^3 s^-1)",
                            obs_col: str = "Observed(m^3 s^-1)") -> Dict[str, Any]:
    """Compute NSE/CC/KGE and peak stats over a bounded time window."""
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df = df[(df[time_col] >= start_ts) & (df[time_col] <= end_ts)].copy()

    if df.empty:
        return {
            "NSE": float("nan"),
            "CC": float("nan"),
            "KGE": float("nan"),
            "sim_peak": float("nan"),
            "obs_peak": float("nan"),
            "peak_ratio": float("nan"),
            "lag_hours_sim_minus_obs": float("nan"),
        }

    s = df[sim_col].astype(float)
    o = df[obs_col].astype(float)

    valid = ~(s.isna() | o.isna())
    s = s[valid]
    o = o[valid]
    t = df.loc[valid, time_col]

    den = np.sum((o - np.mean(o)) ** 2)
    nse = float(1.0 - np.sum((o - s) ** 2) / den) if den > 0 else float("nan")
    cc = safe_corrcoef(s.to_numpy(), o.to_numpy())
    kge_val = kge(s.to_numpy(), o.to_numpy())

    s_peak = float(np.max(s)) if len(s) else float("nan")
    o_peak = float(np.max(o)) if len(o) else float("nan")
    peak_ratio = float(s_peak / o_peak) if (o_peak and o_peak > 0) else float("inf")

    s_tpeak = t.iloc[int(np.argmax(s.to_numpy()))] if len(s) else pd.NaT
    o_tpeak = t.iloc[int(np.argmax(o.to_numpy()))] if len(o) else pd.NaT
    lag_hours = (
        float((s_tpeak - o_tpeak).total_seconds() / 3600.0)
        if (s_tpeak is not pd.NaT and o_tpeak is not pd.NaT)
        else float("nan")
    )

    return {
        "NSE": nse,
        "CC": cc,
        "KGE": kge_val,
        "sim_peak": s_peak,
        "obs_peak": o_peak,
        "peak_ratio": peak_ratio,
        "lag_hours_sim_minus_obs": lag_hours,
    }


def _series_window(df: pd.DataFrame,
                   start: pd.Timestamp,
                   end: pd.Timestamp,
                   sim_col: str,
                   obs_col: str,
                   precip_col: Optional[str]) -> pd.DataFrame:
    sub = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if sub.empty:
        return sub
    for c in (sim_col, obs_col):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if precip_col and precip_col in sub.columns:
        sub[precip_col] = pd.to_numeric(sub[precip_col], errors="coerce")
    sub = sub.dropna(subset=[sim_col, obs_col], how="any")
    return sub


def compute_event_metrics(csv_path: str,
                          windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
                          time_col: str = "Time",
                          sim_col: str = "Discharge(m^3 s^-1)",
                          obs_col: str = "Observed(m^3 s^-1)",
                          precip_col: str = "Precip(mm h^-1)") -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, low_memory=False)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

    metrics: List[Dict[str, Any]] = []
    for (start, end) in windows:
        sub = _series_window(df, start, end, sim_col, obs_col, precip_col)
        if sub.empty:
            metrics.append({
                "start": str(start),
                "end": str(end),
                "NSE": np.nan,
                "CC": np.nan,
                "KGE": np.nan,
                "peak_ratio": np.nan,
                "lag_hours": np.nan,
            })
            continue

        sim = sub[sim_col].to_numpy()
        obs = sub[obs_col].to_numpy()
        den = float(np.sum((obs - np.mean(obs)) ** 2))
        nse = float(1.0 - np.sum((obs - sim) ** 2) / den) if den > 0 else np.nan
        cc = safe_corrcoef(sim, obs)
        kge_val = kge(sim, obs)

        s_pk = float(np.max(sim)) if len(sim) else np.nan
        o_pk = float(np.max(obs)) if len(obs) else np.nan
        pk_ratio = float(s_pk / o_pk) if (o_pk and o_pk > 0) else np.nan

        t_idx = sub.index
        s_tpk = t_idx[int(np.argmax(sim))] if len(sim) else pd.NaT
        o_tpk = t_idx[int(np.argmax(obs))] if len(obs) else pd.NaT
        lag_h = (
            float((s_tpk - o_tpk).total_seconds() / 3600.0)
            if (s_tpk is not pd.NaT and o_tpk is not pd.NaT)
            else np.nan
        )

        metrics.append({
            "start": str(start),
            "end": str(end),
            "NSE": nse,
            "CC": cc,
            "KGE": kge_val,
            "peak_ratio": pk_ratio,
            "lag_hours": lag_h,
        })
    return metrics


def aggregate_event_metrics(event_metrics: Sequence[Dict[str, Any]],
                            top_n: int = EVENTS_FOR_AGGREGATE) -> Dict[str, float]:
    if not event_metrics:
        return {"NSE": np.nan, "CC": np.nan, "KGE": np.nan, "peak_ratio": np.nan, "lag_hours": np.nan}

    finite_events = [m for m in event_metrics if np.isfinite(m.get("NSE", np.nan))]
    events = finite_events or event_metrics
    events = events[:top_n]

    agg: Dict[str, float] = {}
    for key in ("NSE", "CC", "KGE", "peak_ratio", "lag_hours"):
        vals = [m.get(key, np.nan) for m in events]
        finite = [v for v in vals if np.isfinite(v)]
        agg[key] = float(np.mean(finite)) if finite else float("nan")
    return agg


__all__ = [
    "read_metrics_from_csv",
    "compute_event_metrics",
    "aggregate_event_metrics",
    "kge",
    "safe_corrcoef",
    "read_metrics_for_period",
]
