"""Configuration constants for calibration."""

from __future__ import annotations

LLM_MODEL_DEFAULT = "gemini-3-pro"
LLM_MODEL_REASONING = "gemini-3-pro"

# Calibration behaviour
MAX_STEPS_DEFAULT = 20
IMPROVE_PATIENCE = 100
EVENTS_FOR_AGGREGATE = 3

# Parameter bounds (min, max)
PARAM_BOUNDS = {
    "wm": (0.1, 10.0),
    "b": (0.0, 3.0),
    "im": (0.0, 1.0),
    "ke": (0.8, 1.2),
    "fc": (0.1, 2.0),
    "iwu": (25.0, 25.0),
    "under": (0.1, 10.0),
    "leaki": (0.1, 10.0),
    "th": (10.0, 1000.0),
    "isu": (0.0, 0.01),
    "alpha": (0.1, 3.0),
    "beta": (0.1, 3.0),
    "alpha0": (0.0, 3.0),
}

# Parameters that remain fixed during calibration
FROZEN_PARAMETERS = {"th", "iwu", "isu"}

# Default simulation layout
DEFAULT_SIM_FOLDER = "cali_set/ky_03302000_2018"
DEFAULT_GAUGE_NUM = "03302000"

# Event selection defaults
DEFAULT_PEAK_PICK_KWARGS = dict(
    min_separation_hours=48,
    baseflow_percentile=0.20,
    frac_of_peak=0.20,
    padding_hours=12,
    max_search_hours=240,
    smooth_window_hours=3.0,
)
