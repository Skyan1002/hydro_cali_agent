"""Peak detection utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class PeakEvent:
    peak_time: pd.Timestamp
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_value: float


def _read_series(csv_path: str,
                 time_col: str = "Time",
                 obs_col: str = "Observed(m^3 s^-1)") -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not found. Available: {list(df.columns)}")
    if obs_col not in df.columns:
        alt = [c for c in df.columns if ("observed" in c.lower() or "obs" in c.lower())]
        hint = f" Did you mean one of {alt}?" if alt else ""
        raise KeyError(f"Observed column '{obs_col}' not found.{hint}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=False)
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    df = df.set_index(time_col)
    df[obs_col] = pd.to_numeric(df[obs_col], errors="coerce")
    df = df.dropna(subset=[obs_col])
    return df


def _guess_sim_columns(df: pd.DataFrame, obs_col: str) -> List[str]:
    sims = []
    for c in df.columns:
        if c == obs_col:
            continue
        cl = c.lower()
        if ("sim" in cl) or c.startswith("Run") or c.startswith("CREST"):
            sims.append(c)
    return sims


def _median_dt_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    dts = index.to_series().diff().dropna().dt.total_seconds().values
    if len(dts) == 0:
        return 1.0
    return float(np.median(dts) / 3600.0)


def _local_maxima(series: pd.Series, order: int, threshold: float) -> List[int]:
    y = series.values
    n = len(y)
    idx = []
    for i in range(n):
        lo = max(0, i - order)
        hi = min(n, i + order + 1)
        if y[i] >= threshold and y[i] == np.nanmax(y[lo:hi]):
            idx.append(i)
    dedup: List[int] = []
    last = -10**9
    for i in idx:
        if dedup and i - last <= 1 and series.iloc[i] == series.iloc[last]:
            continue
        dedup.append(i)
        last = i
    return dedup


def pick_peak_events(csv_path: str,
                     n: int = 5,
                     time_col: str = "Time",
                     obs_col: str = "Observed(m^3 s^-1)",
                     min_separation_hours: float = 24.0,
                     baseflow_percentile: float = 0.20,
                     frac_of_peak: float = 0.20,
                     padding_hours: float = 6.0,
                     max_search_hours: float = 240.0,
                     smooth_window_hours: Optional[float] = 3.0,
                     windows_csv_out: Optional[str] = None) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    df = _read_series(csv_path, time_col=time_col, obs_col=obs_col).copy()
    dt_h = _median_dt_hours(df.index)
    k_pad = max(1, int(round(padding_hours / dt_h)))
    k_sep = max(1, int(round(min_separation_hours / dt_h)))
    k_search = max(1, int(round(max_search_hours / dt_h)))

    s = df[obs_col].copy()
    if smooth_window_hours and smooth_window_hours > 0:
        k_smooth = max(1, int(round(smooth_window_hours / dt_h)))
        s = s.rolling(window=k_smooth, center=True, min_periods=1).mean()

    threshold = np.nanpercentile(s.values, 90)
    order = max(1, int(round(k_sep / 2)))
    cand_idx = _local_maxima(s, order=order, threshold=threshold)
    if not cand_idx:
        return []

    cand_times = s.index.values[cand_idx]
    cand_vals = s.values[cand_idx]
    order_by_height = np.argsort(-cand_vals)
    selected_idx: List[int] = []
    for j in order_by_height:
        t = pd.Timestamp(cand_times[j])
        if all(abs((t - pd.Timestamp(cand_times[k])).total_seconds()) / 3600.0 >= min_separation_hours
               for k in selected_idx):
            selected_idx.append(j)
        if len(selected_idx) >= n:
            break
    if len(selected_idx) < n:
        for j in np.argsort(cand_times):
            if j in selected_idx:
                continue
            t = pd.Timestamp(cand_times[j])
            if all(abs((t - pd.Timestamp(cand_times[k])).total_seconds()) / 3600.0 >= min_separation_hours
                   for k in selected_idx):
                selected_idx.append(j)
            if len(selected_idx) >= n:
                break

    selected_times = [pd.Timestamp(cand_times[j]) for j in selected_idx]
    selected_vals = [float(cand_vals[j]) for j in selected_idx]
    baseflow = float(np.nanpercentile(s.values, baseflow_percentile * 100.0))

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for peak_t, peak_v in zip(selected_times, selected_vals):
        ip = s.index.get_loc(peak_t)
        stop_val = max(baseflow, frac_of_peak * peak_v)

        i0 = ip
        lo = max(0, ip - k_search)
        while i0 > lo and s.iloc[i0] > stop_val:
            i0 -= 1
        i0 = max(0, i0 - k_pad)

        i1 = ip
        hi = min(len(s) - 1, ip + k_search)
        while i1 < hi and s.iloc[i1] > stop_val:
            i1 += 1
        i1 = min(len(s) - 1, i1 + k_pad)

        start_t = s.index[i0]
        end_t = s.index[i1]
        windows.append((start_t, end_t))

    windows = [w for _, w in sorted(zip(selected_times, windows))]

    if windows_csv_out is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "peaks")
        os.makedirs(out_dir, exist_ok=True)
        windows_csv_out = os.path.join(out_dir, "windows.csv")
    pd.DataFrame([{"start": a, "end": b} for a, b in windows]).to_csv(windows_csv_out, index=False)
    return windows


__all__ = [
    "PeakEvent",
    "_read_series",
    "_guess_sim_columns",
    "pick_peak_events",
]
