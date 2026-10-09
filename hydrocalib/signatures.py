"""Numerical hydrologic signatures.

Provides the same diagnostic information the vision agent extracts from the 2x2
montage (FDC + 3 event hydrographs), but as numbers instead of an image. Used by

* the ``numeric`` perception mode of INCA (arm A1), and
* the non-LLM rule-based proposer (arm B1),

so that both receive information matched to what the image arm sees.

Signature families follow the diagnostic-calibration literature
(Yilmaz et al. 2008 WRR; Kavetski et al. 2018 WRR): water balance, flow-duration
curve shape, high/low flow bias, recession behaviour, and event-scale timing and
magnitude errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SIM_COL = "Discharge(m^3 s^-1)"
OBS_COL = "Observed(m^3 s^-1)"
PRECIP_COL = "Precip(mm h^-1)"
TIME_COL = "Time"


def _paired(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sim = pd.to_numeric(df[SIM_COL], errors="coerce").to_numpy(dtype=float)
    obs = pd.to_numeric(df[OBS_COL], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    return sim[mask], obs[mask]


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b not in (0.0,) and np.isfinite(b) else float("nan")


def _fdc(values: np.ndarray) -> np.ndarray:
    """Flow-duration curve: values sorted descending."""
    v = values[np.isfinite(values)]
    return np.sort(v)[::-1]


def _fdc_midslope(values: np.ndarray) -> float:
    """Slope of the FDC between the 30th and 70th exceedance percentiles (log space).

    Yilmaz et al. (2008) use this as the primary indicator of how well the model
    partitions water between quick and slow response.
    """
    v = _fdc(values)
    if v.size < 10:
        return float("nan")
    lo = v[int(0.70 * v.size)]
    hi = v[int(0.30 * v.size)]
    if lo <= 0 or hi <= 0:
        return float("nan")
    return float((np.log(hi) - np.log(lo)) / (0.70 - 0.30))


def _tail_bias(sim: np.ndarray, obs: np.ndarray, lo_q: float, hi_q: float) -> float:
    """Relative volume bias over an exceedance band of the observed FDC."""
    s, o = _fdc(sim), _fdc(obs)
    n = min(s.size, o.size)
    if n < 10:
        return float("nan")
    a, b = int(lo_q * n), max(int(hi_q * n), int(lo_q * n) + 1)
    so, oo = s[a:b].sum(), o[a:b].sum()
    return _safe_ratio(so - oo, oo)


def _baseflow_index(q: np.ndarray, alpha: float = 0.98, bfi_max: float = 0.80) -> float:
    """Eckhardt (2005) two-parameter recursive digital filter."""
    q = q[np.isfinite(q)]
    if q.size < 24:
        return float("nan")
    b = np.empty_like(q)
    b[0] = q[0]
    c1 = (1.0 - bfi_max) * alpha
    c2 = (1.0 - alpha) * bfi_max
    denom = 1.0 - alpha * bfi_max
    for i in range(1, q.size):
        b[i] = min((c1 * b[i - 1] + c2 * q[i]) / denom, q[i])
    return _safe_ratio(b.sum(), q.sum())


def _recession_constant(q: np.ndarray, min_len: int = 5) -> float:
    """Median exponential recession constant k in Q(t+1) = Q(t)*exp(-1/k)."""
    q = np.asarray(q, dtype=float)
    ks: List[float] = []
    run: List[float] = []
    for i in range(1, q.size):
        if np.isfinite(q[i]) and np.isfinite(q[i - 1]) and 0 < q[i] < q[i - 1]:
            if not run:
                run.append(q[i - 1])
            run.append(q[i])
        else:
            if len(run) >= min_len:
                seg = np.array(run)
                d = -np.diff(np.log(seg))
                d = d[np.isfinite(d) & (d > 0)]
                if d.size:
                    ks.append(1.0 / float(np.median(d)))
            run = []
    return float(np.median(ks)) if ks else float("nan")


def _lag1_autocorr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return _safe_ratio(float(np.dot(x[:-1], x[1:])), denom)


def _timestep_seconds(df: pd.DataFrame, default: float = 3600.0) -> float:
    """Median spacing of the Time column in seconds (falls back to hourly)."""
    if TIME_COL not in df.columns:
        return default
    t = pd.to_datetime(df[TIME_COL], errors="coerce").dropna()
    if t.size < 2:
        return default
    dt = float(np.nanmedian(np.diff(t.to_numpy()).astype("timedelta64[s]").astype(float)))
    return dt if np.isfinite(dt) and dt > 0 else default


def discharge_to_depth_mm(q_sum_m3s: float, dt_seconds: float, area_km2: float) -> float:
    """Convert a sum of discharge [m^3/s] over ``dt_seconds`` steps to runoff depth [mm].

    depth_mm = sum(Q) * dt * 1000 / (A_km2 * 1e6)
    """
    return float(q_sum_m3s) * dt_seconds * 1000.0 / (float(area_km2) * 1e6)


def global_signatures(csv_path: str, area_km2: Optional[float] = None) -> Dict[str, float]:
    """Whole-period signatures — the numeric analogue of the FDC panel.

    ``area_km2`` is required for a dimensionally correct runoff coefficient:
    discharge [m^3/s] is converted to basin-averaged depth [mm] before it is
    compared with precipitation [mm]. Without it the runoff-ratio entries are
    omitted (``volume_bias_frac`` already carries the sim/obs volume comparison).
    """
    df = pd.read_csv(csv_path)
    sim_all = pd.to_numeric(df[SIM_COL], errors="coerce").to_numpy(dtype=float)
    obs_all = pd.to_numeric(df[OBS_COL], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(sim_all) & np.isfinite(obs_all)
    sim, obs = sim_all[mask], obs_all[mask]
    if sim.size == 0:
        return {}
    out: Dict[str, float] = {}

    # --- water balance ---
    out["volume_bias_frac"] = _safe_ratio(sim.sum() - obs.sum(), obs.sum())
    out["mean_sim"] = float(sim.mean())
    out["mean_obs"] = float(obs.mean())
    if PRECIP_COL in df.columns and area_km2 is not None and np.isfinite(area_km2) and area_km2 > 0:
        # Same paired time steps as sim/obs so numerator and denominator cover one window.
        p = pd.to_numeric(df[PRECIP_COL], errors="coerce").to_numpy(dtype=float)[mask]
        dt_s = _timestep_seconds(df)
        p_tot = float(np.nansum(p)) * dt_s / 3600.0  # mm/h per step -> mm
        if p_tot > 0:
            depth_sim = discharge_to_depth_mm(float(sim.sum()), dt_s, area_km2)
            depth_obs = discharge_to_depth_mm(float(obs.sum()), dt_s, area_km2)
            out["runoff_ratio_sim"] = _safe_ratio(depth_sim, p_tot)
            out["runoff_ratio_obs"] = _safe_ratio(depth_obs, p_tot)
            out["runoff_ratio_bias"] = _safe_ratio(out["runoff_ratio_sim"], out["runoff_ratio_obs"])

    # --- FDC shape and tails ---
    out["fdc_midslope_sim"] = _fdc_midslope(sim)
    out["fdc_midslope_obs"] = _fdc_midslope(obs)
    out["fdc_midslope_bias"] = _safe_ratio(
        out["fdc_midslope_sim"] - out["fdc_midslope_obs"], abs(out["fdc_midslope_obs"])
    )
    out["high_flow_bias_top2pct"] = _tail_bias(sim, obs, 0.00, 0.02)
    out["mid_flow_bias_20_70pct"] = _tail_bias(sim, obs, 0.20, 0.70)
    out["low_flow_bias_bottom30pct"] = _tail_bias(sim, obs, 0.70, 1.00)

    # --- flow partitioning and dynamics ---
    out["bfi_sim"] = _baseflow_index(sim)
    out["bfi_obs"] = _baseflow_index(obs)
    out["bfi_bias"] = out["bfi_sim"] - out["bfi_obs"]
    out["recession_k_sim"] = _recession_constant(sim)
    out["recession_k_obs"] = _recession_constant(obs)
    out["recession_k_ratio"] = _safe_ratio(out["recession_k_sim"], out["recession_k_obs"])
    out["lag1_autocorr_sim"] = _lag1_autocorr(sim)
    out["lag1_autocorr_obs"] = _lag1_autocorr(obs)

    return {k: (float(v) if np.isfinite(v) else None) for k, v in out.items()}


def event_signatures(csv_path: str, windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-event signatures — the numeric analogue of the three hydrograph panels."""
    df = pd.read_csv(csv_path)
    if TIME_COL not in df.columns:
        return []
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    out: List[Dict[str, Any]] = []
    for i, w in enumerate(windows, start=1):
        start, end = pd.to_datetime(w.get("start")), pd.to_datetime(w.get("end"))
        seg = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)]
        if seg.empty:
            continue
        sim = pd.to_numeric(seg[SIM_COL], errors="coerce").to_numpy(dtype=float)
        obs = pd.to_numeric(seg[OBS_COL], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(sim) & np.isfinite(obs)
        if m.sum() < 3:
            continue
        s, o = sim[m], obs[m]
        t = seg[TIME_COL].to_numpy()[m]
        i_s, i_o = int(np.argmax(s)), int(np.argmax(o))
        lag_h = float((t[i_s] - t[i_o]) / np.timedelta64(1, "h"))
        rec: Dict[str, Any] = {
            "event": i,
            "start": str(start),
            "end": str(end),
            "peak_ratio": _safe_ratio(float(s.max()), float(o.max())),
            "peak_lag_hours_sim_minus_obs": lag_h,
            "event_volume_bias_frac": _safe_ratio(float(s.sum() - o.sum()), float(o.sum())),
            "rising_limb_ratio": _safe_ratio(
                float(s.max() - s[0]) / max(i_s, 1), float(o.max() - o[0]) / max(i_o, 1)
            ),
            # shorter minimum run: a 3-day event window rarely holds a 5-step recession
            "recession_k_sim": _recession_constant(s, min_len=3),
            "recession_k_obs": _recession_constant(o, min_len=3),
        }
        out.append({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rec.items()})
    return out


def render_signature_report(global_sig: Dict[str, Any],
                            event_sig: Sequence[Dict[str, Any]]) -> str:
    """Render signatures as the text block that replaces ``image_summary``.

    Kept deliberately close in length and framing to what the vision agent
    returns, so the information — not the verbosity — is what differs.
    """
    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)

    lines = ["NUMERICAL DIAGNOSTIC SIGNATURES (whole calibration period)"]
    groups = [
        ("Water balance", ["volume_bias_frac", "runoff_ratio_sim", "runoff_ratio_obs", "runoff_ratio_bias"]),
        ("Flow-duration curve",
         ["fdc_midslope_sim", "fdc_midslope_obs", "fdc_midslope_bias",
          "high_flow_bias_top2pct", "mid_flow_bias_20_70pct", "low_flow_bias_bottom30pct"]),
        ("Baseflow / recession",
         ["bfi_sim", "bfi_obs", "bfi_bias", "recession_k_sim", "recession_k_obs", "recession_k_ratio"]),
        ("Temporal structure", ["lag1_autocorr_sim", "lag1_autocorr_obs"]),
    ]
    for title, keys in groups:
        present = [k for k in keys if k in global_sig]
        if present:
            line = f"- {title}: " + ", ".join(f"{k}={fmt(global_sig.get(k))}" for k in present)
            if title == "Water balance" and "runoff_ratio_sim" not in global_sig:
                line += "; runoff coefficient not reported (basin area unavailable)"
            lines.append(line)

    if event_sig:
        lines.append("PER-EVENT SIGNATURES (same events the montage would show)")
        for e in event_sig:
            lines.append(
                f"- Event {e.get('event')} [{e.get('start')} .. {e.get('end')}]: "
                f"peak_ratio={fmt(e.get('peak_ratio'))}, "
                f"peak_lag_hours={fmt(e.get('peak_lag_hours_sim_minus_obs'))}, "
                f"volume_bias={fmt(e.get('event_volume_bias_frac'))}, "
                f"rising_limb_ratio={fmt(e.get('rising_limb_ratio'))}, "
                f"recession_k_sim/obs={fmt(e.get('recession_k_sim'))}/{fmt(e.get('recession_k_obs'))}"
            )

    lines.append(
        "Interpretation notes: values >0 for *_bias mean the simulation is too high; "
        "peak_ratio>1 means simulated peaks are too large; peak_lag_hours>0 means simulated "
        "peaks arrive late; recession_k_ratio<1 means simulated recessions are too fast"
        + ("; runoff_ratio_* are runoff coefficients (runoff depth / precipitation depth, "
           "dimensionless) and runoff_ratio_bias = ratio_sim/ratio_obs."
           if "runoff_ratio_sim" in global_sig else ".")
    )
    return "\n".join(lines)


def compute_signature_summary(csv_path: str,
                              windows: Optional[Sequence[Dict[str, Any]]] = None,
                              area_km2: Optional[float] = None) -> tuple[str, Dict[str, Any]]:
    """Convenience wrapper returning (text report, raw signature dict).

    ``area_km2`` (basin area) enables the dimensionally correct runoff
    coefficient; when it is None the runoff-ratio entries are dropped and the
    report says so.
    """
    g = global_signatures(csv_path, area_km2=area_km2)
    e = event_signatures(csv_path, windows or [])
    return render_signature_report(g, e), {"global": g, "events": e, "area_km2": area_km2}


__all__ = [
    "global_signatures",
    "event_signatures",
    "render_signature_report",
    "compute_signature_summary",
    "discharge_to_depth_mm",
]
