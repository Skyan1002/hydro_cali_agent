"""Plotting utilities for hydrographs."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import _series_window, safe_corrcoef
from .peak_events import _read_series


def plot_hydrograph_with_precipitation(csv_path: str, show: bool = True) -> str:
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'])

    valid_data = df.dropna(subset=['Discharge(m^3 s^-1)', 'Observed(m^3 s^-1)'])
    if valid_data.empty:
        cc = float('nan')
        nsce = float('nan')
    else:
        sim = valid_data['Discharge(m^3 s^-1)'].to_numpy()
        obs = valid_data['Observed(m^3 s^-1)'].to_numpy()
        cc = safe_corrcoef(sim, obs)
        mean_observed = float(np.mean(obs)) if obs.size else float('nan')
        if obs.size < 2 or not np.isfinite(mean_observed):
            nsce = float('nan')
        else:
            numerator = float(np.nansum((obs - sim) ** 2))
            denominator = float(np.nansum((obs - mean_observed) ** 2))
            nsce = 1 - (numerator / denominator) if denominator else float('nan')

    plt.rcParams.update({'font.family': 'serif', 'font.size': 16})
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    time_diffs = np.diff(df['Time'].values).astype('timedelta64[s]').astype(np.float64)
    if len(time_diffs) == 0 or np.std(time_diffs) > 0.01 * np.mean(time_diffs):
        width = 0.01
    else:
        avg_diff_seconds = np.mean(time_diffs)
        width = avg_diff_seconds / (24 * 60 * 60)

    ax1.plot(df['Time'], df['Discharge(m^3 s^-1)'], 'b-', label='Simulated Discharge', linewidth=2)
    ax1.scatter(df['Time'], df['Observed(m^3 s^-1)'], color='black', s=10, label='Observed Discharge')
    discharge_candidates = [df['Discharge(m^3 s^-1)'].max(), df['Observed(m^3 s^-1)'].max()]
    discharge_finite = [v for v in discharge_candidates if np.isfinite(v) and v is not None]
    if discharge_finite:
        max_discharge = max(discharge_finite)
        ax1.set_ylim(0, max_discharge * 2)
    precip_series = None
    max_precip = np.nan
    ax2 = None
    if 'Precip(mm h^-1)' in df.columns:
        precip_series = pd.to_numeric(df['Precip(mm h^-1)'], errors='coerce')
        if precip_series.notna().any() and (precip_series > 0).any():
            max_precip = float(precip_series.max())
            ax2 = ax1.twinx()
            ax2.bar(df['Time'], precip_series, width=width, color='skyblue', alpha=0.6, label='Precipitation')
            ax2.set_ylabel('Precipitation (mm/h)', color='skyblue', fontsize=18)
            ax2.set_ylim(max_precip * 2, 0)
    ax1.set_ylabel('Streamflow (m³/s)', color='b', fontsize=18)
    ax1.set_title('Hydrograph with Precipitation (Normal Scale)', fontsize=20)
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)
    else:
        ax1.legend(loc='upper right', fontsize=16)
    text_str = f'CC = {cc:.3f}\nNSCE = {nsce:.3f}'
    ax1.text(0.02, 0.85, text_str, transform=ax1.transAxes, fontsize=18,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    ax3.plot(df['Time'], df['Discharge(m^3 s^-1)'], 'b-', label='Simulated Discharge', linewidth=2)
    ax3.scatter(df['Time'], df['Observed(m^3 s^-1)'], color='black', s=4, label='Observed Discharge')
    ax3.set_yscale('log')
    ax4 = None
    if precip_series is not None and np.isfinite(max_precip) and max_precip > 0:
        ax4 = ax3.twinx()
        ax4.bar(df['Time'], precip_series, width=width, color='skyblue', alpha=0.6, label='Precipitation')
        ax4.set_ylabel('Precipitation (mm/h)', color='skyblue', fontsize=18)
        ax4.set_ylim(max_precip * 2, 0)
    ax3.set_xlabel('Time', fontsize=18)
    ax3.set_ylabel('Streamflow (m³/s) - Log Scale', color='b', fontsize=18)
    ax3.set_title('Hydrograph with Precipitation (Log Scale)', fontsize=20)
    lines3, labels3 = ax3.get_legend_handles_labels()
    if ax4 is not None:
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=16)
    else:
        ax3.legend(loc='upper right', fontsize=16)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:00'))
    fig.autofmt_xdate()

    for ax in [ax1, ax3]:
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, alpha=0.3)
    for ax in [ax2, ax4]:
        ax.tick_params(axis='y', labelsize=16)
    plt.xticks(fontsize=16)
    plt.tight_layout()

    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'hydrograph.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_event_windows(csv_path: str,
                       windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
                       time_col: str = "Time",
                       obs_col: str = "Observed(m^3 s^-1)",
                       discharge_col: str = "Discharge(m^3 s^-1)",
                       precip_col: str = "Precip(mm h^-1)",
                       out_dir: Optional[str] = None,
                       rotate_xticks: int | float = 30) -> List[str]:
    df = _read_series(csv_path, time_col=time_col, obs_col=obs_col)

    has_discharge = discharge_col in df.columns
    if has_discharge:
        df[discharge_col] = pd.to_numeric(df[discharge_col], errors="coerce")

    has_precip = precip_col in df.columns
    if has_precip:
        df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")

    sim_cols = []
    for c in df.columns:
        if c in (obs_col, discharge_col, precip_col):
            continue
        cl = c.lower()
        if ("sim" in cl) or c.startswith("Run") or c.startswith("CREST"):
            sim_cols.append(c)

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "peaks")
    os.makedirs(out_dir, exist_ok=True)

    def _bar_width_days(index: pd.DatetimeIndex) -> float:
        if len(index) < 2:
            return 1.0 / 1440.0
        diffs_sec = index.to_series().diff().dropna().dt.total_seconds().values
        if len(diffs_sec) == 0:
            return 1.0 / 1440.0
        med_sec = float(np.median(diffs_sec))
        return max((med_sec / 86400.0) * 0.8, 1.0 / 1440.0)

    saved: List[str] = []
    for k, (t0, t1) in enumerate(windows, start=1):
        sub = df.loc[(df.index >= t0) & (df.index <= t1)].copy()
        if sub.empty:
            continue

        width_days = _bar_width_days(sub.index)
        max_precip = sub[precip_col].max() if has_precip else np.nan
        obs_series = sub[obs_col].dropna()
        obs_peak_time = obs_series.idxmax() if not obs_series.empty else (sub.index[0] if not sub.empty else None)

        sub_log = sub.copy()
        sub_log[obs_col] = sub_log[obs_col].where(sub_log[obs_col] > 0, np.nan)
        if has_discharge:
            sub_log[discharge_col] = sub_log[discharge_col].where(sub_log[discharge_col] > 0, np.nan)
        for c in sim_cols:
            sub_log[c] = sub_log[c].where(sub_log[c] > 0, np.nan)

        plt.rcParams.update({"font.family": "serif", "font.size": 13})
        fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ymax_candidates = [sub[obs_col].max()]
        ymax_candidates.extend(sub[c].max() for c in sim_cols)
        if has_discharge:
            ymax_candidates.append(sub[discharge_col].max())
        finite_ymax = [v for v in ymax_candidates if np.isfinite(v) and v is not None]
        ymax_lin = max(finite_ymax) if finite_ymax else np.nan
        ax_lin.plot(sub.index, sub[obs_col], label="Observed", linewidth=2, color="k")
        if has_discharge and sub[discharge_col].notna().any():
            ax_lin.plot(sub.index, sub[discharge_col], label=discharge_col, linewidth=1.8)
        for c in sim_cols:
            ax_lin.plot(sub.index, sub[c], label=c, linewidth=1.6)
        if np.isfinite(ymax_lin) and ymax_lin > 0:
            ax_lin.set_ylim(0, ymax_lin * 2)
        if obs_peak_time is not None:
            ax_lin.axvline(obs_peak_time, linestyle="--", alpha=0.7, color="k", label="Obs peak")
        if obs_peak_time is not None and np.isfinite(ymax_lin) and ymax_lin > 0:
            for c in ([discharge_col] if has_discharge else []) + sim_cols:
                series = sub[c].dropna() if c in sub else pd.Series(dtype=float)
                if series.empty:
                    continue
                tpk = series.idxmax()
                lag_h = (tpk - obs_peak_time).total_seconds() / 3600.0 if obs_peak_time is not None else np.nan
                ax_lin.axvline(tpk, linestyle=":", alpha=0.7)
                if np.isfinite(lag_h):
                    ax_lin.text(tpk, ymax_lin * 2, f"lag={lag_h:+.1f} h",
                                rotation=90, va="bottom", ha="right", fontsize=8)

        if has_precip and sub[precip_col].notna().any():
            axp_lin = ax_lin.twinx()
            axp_lin.bar(sub.index, sub[precip_col], width=width_days, color="skyblue", alpha=0.6, label="Precipitation")
            axp_lin.set_ylabel("Precipitation (mm/h)", color="skyblue")
            if np.isfinite(max_precip) and max_precip > 0:
                axp_lin.set_ylim(max_precip * 2, 0)
        ax_lin.set_ylabel("Streamflow (m³/s)")

        ax_lin.set_title(f"Event {k}: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} (linear)")
        ax_lin.grid(True, alpha=0.3)
        lines1, labels1 = ax_lin.get_legend_handles_labels()
        if has_precip and sub[precip_col].notna().any():
            lines2, labels2 = axp_lin.get_legend_handles_labels()
            ax_lin.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)
        else:
            ax_lin.legend(loc="upper right", fontsize=11)

        ymax_candidates_log = [sub_log[obs_col].max()]
        ymax_candidates_log.extend(sub_log[c].max() for c in sim_cols)
        if has_discharge:
            ymax_candidates_log.append(sub_log[discharge_col].max())
        finite_ymax_log = [v for v in ymax_candidates_log if np.isfinite(v) and v is not None]
        ymax_log = max(finite_ymax_log) if finite_ymax_log else np.nan
        ax_log.plot(sub_log.index, sub_log[obs_col], label="Observed", linewidth=2, color="k")
        if has_discharge and sub_log[discharge_col].notna().any():
            ax_log.plot(sub_log.index, sub_log[discharge_col], label=discharge_col, linewidth=1.8)
        for c in sim_cols:
            ax_log.plot(sub_log.index, sub_log[c], label=c, linewidth=1.6)
        ax_log.set_yscale("log")
        if np.isfinite(ymax_log) and ymax_log > 0:
            ymin_log = np.nanmin([
                sub_log[obs_col].min(),
                *(sub_log[c].min() for c in sim_cols),
                *( [sub_log[discharge_col].min()] if has_discharge else [] ),
            ])
            if np.isfinite(ymin_log) and ymin_log > 0:
                ax_log.set_ylim(ymin_log, ymax_log * 2)
            else:
                ax_log.set_ylim(bottom=None, top=ymax_log * 2)
        if obs_peak_time is not None:
            ax_log.axvline(obs_peak_time, linestyle="--", alpha=0.7, color="k", label="Obs peak")
        if obs_peak_time is not None and np.isfinite(ymax_log) and ymax_log > 0:
            for c in ([discharge_col] if has_discharge else []) + sim_cols:
                series = sub[c].dropna() if c in sub else pd.Series(dtype=float)
                if series.empty:
                    continue
                tpk = series.idxmax()
                lag_h = (tpk - obs_peak_time).total_seconds() / 3600.0 if obs_peak_time is not None else np.nan
                ax_log.axvline(tpk, linestyle=":", alpha=0.7)
                if np.isfinite(lag_h):
                    ax_log.text(tpk, ymax_log * 2, f"lag={lag_h:+.1f} h",
                                rotation=90, va="bottom", ha="right", fontsize=8)

        if has_precip and sub[precip_col].notna().any():
            axp_log = ax_log.twinx()
            axp_log.bar(sub.index, sub[precip_col], width=width_days, color="skyblue", alpha=0.6, label="Precipitation")
            axp_log.set_ylabel("Precipitation (mm/h)", color="skyblue")
            if np.isfinite(max_precip) and max_precip > 0:
                axp_log.set_ylim(max_precip * 2, 0)
        ax_log.set_ylabel("Streamflow (m³/s) [log]")
        ax_log.set_xlabel("Time")
        ax_log.set_title(f"Event {k}: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} (log)")
        ax_log.grid(True, alpha=0.3)

        time_range = (t1 - t0).total_seconds() / 3600.0
        if time_range <= 48:
            ax_log.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_range / 10))))
            ax_log.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        elif time_range <= 240:
            interval = 6 if time_range <= 120 else 12
            ax_log.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
            ax_log.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        else:
            ax_log.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_range / 240))))
            ax_log.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        for tick in ax_log.get_xticklabels():
            tick.set_rotation(rotate_xticks)
            tick.set_horizontalalignment("right")

        lines3, labels3 = ax_log.get_legend_handles_labels()
        if has_precip and sub[precip_col].notna().any():
            lines4, labels4 = axp_log.get_legend_handles_labels()
            ax_log.legend(lines3 + lines4, labels3 + labels4, loc="upper right", fontsize=11)
        else:
            ax_log.legend(loc="upper right", fontsize=11)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"event_{k:02d}_{t0:%Y%m%d%H}_{t1:%Y%m%d%H}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    if saved:
        pd.DataFrame({"figure": saved}).to_csv(os.path.join(out_dir, "fig_index.csv"), index=False)
    return saved


def plot_flow_duration_curve(csv_path: str, show: bool = True) -> str:
    df = pd.read_csv(csv_path)
    sim_col = "Discharge(m^3 s^-1)"
    obs_col = "Observed(m^3 s^-1)"

    def _prepare(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        vals = pd.to_numeric(series, errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size == 0:
            return np.array([]), np.array([])
        sorted_vals = np.sort(vals)[::-1]
        ranks = np.arange(1, sorted_vals.size + 1)
        exceed_prob = ranks / (sorted_vals.size + 1) * 100.0
        return exceed_prob, sorted_vals

    sim_exceed, sim_vals = _prepare(df[sim_col]) if sim_col in df.columns else (np.array([]), np.array([]))
    obs_exceed, obs_vals = _prepare(df[obs_col]) if obs_col in df.columns else (np.array([]), np.array([]))

    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig, ax = plt.subplots(figsize=(8, 6))

    if sim_vals.size:
        ax.plot(sim_exceed, sim_vals, label="Simulated", linewidth=2, color="tab:blue")
    if obs_vals.size:
        ax.plot(obs_exceed, obs_vals, label="Observed", linewidth=2, color="k")

    if sim_vals.size or obs_vals.size:
        ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("Flow Duration Curve")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, "fdc.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


__all__ = [
    "plot_hydrograph_with_precipitation",
    "plot_event_windows",
    "plot_flow_duration_curve",
]
