"""Non-LLM optimizer baselines for EF5/CREST calibration.

This module provides the control arms for the INCA revision experiments.  Every
optimizer sees exactly the same search space (the free entries of
``PARAM_BOUNDS``), the same forward model (``SimulationRunner``), the same
objective function, and the same evaluation budget as the LLM-driven
calibration loop, so that differences in convergence can only come from the
proposal mechanism.

Implemented methods
-------------------
``sceua``
    Shuffled Complex Evolution with the competitive complex evolution (CCE)
    step exactly as published by Duan, Sorooshian & Gupta (1992) and Duan et
    al. (1994): reflection ``r = g + (g - u_q)`` through the centroid of the
    sub-complex, contraction to the midpoint ``(g + u_q)/2``, and random
    re-sampling inside the smallest hypercube containing the complex when a
    reflection leaves the feasible box (no clipping).  Default sizes follow
    Duan et al. (1994): ``m = 2n+1`` points per complex, ``q = n+1`` points per
    sub-complex, ``beta = 2n+1`` evolution steps per shuffle, ``p = 2``
    complexes.  The only restructuring is that the offspring of all complexes
    for one CCE stage are evaluated in a single parallel batch.
``cmaes``
    Textbook (mu/mu_w, lambda)-CMA-ES (Hansen 2016 tutorial) written directly
    on top of numpy: weighted recombination, evolution paths, rank-one and
    rank-mu covariance updates, and cumulative step-size adaptation.  Sampling
    happens in a normalised [0, 1]^d cube that is mapped back onto the real
    bounds.
``bo``
    Gaussian-process Bayesian optimisation.  ARD Matern-5/2 kernel with
    analytic log-marginal-likelihood gradients (scipy only, no sklearn),
    Latin-hypercube initial design, and Expected Improvement maximised by
    multi-start L-BFGS-B.  Batches are produced with the Kriging-believer
    heuristic so that the EF5 runs can still be parallelised.
``eki``
    Ensemble Kalman Inversion in the form of Iglesias, Law and Stuart (2013).
    The observation operator is a vector of standardised hydrologic
    diagnostics (all with target value zero), which is what makes the Kalman
    update meaningful; a scalar operator ``G = 1 - NSE`` (target 0, i.e. NSE
    maximisation) is available via ``observation_mode="scalar"``.
``rule``
    Deterministic diagnostic proposer.  It receives *exactly* the same numeric
    diagnostics as the INCA numeric arm (``signatures.compute_signature_summary``)
    and the same physical knowledge (the EF5 parameter guide used in the LLM
    prompts), but translates them into parameter moves through a fixed lookup
    table instead of an LLM.  It keeps the INCA outer loop: k candidates per
    round, best candidate becomes the new baseline, early stop after
    ``failure_patient`` non-improving rounds (optionally followed by random
    local restarts until the budget is spent, ``fill_budget=True``).
``random_local``
    Uninformed null control for ``rule``: identical outer loop, batch size,
    step grid, de-duplication and stopping rule, but the parameters to move and
    the step multipliers/directions are drawn at random instead of from the
    diagnostics.  Any gap between ``rule`` and ``random_local`` is therefore
    attributable to the diagnostic information alone.

Objectives
----------
``nse``
    Full-period NSE, identical to the INCA manager's ``_objective_value``
    (which reads ``full_metrics["NSE"]``).
``composite``
    NSE minus three dimensionless, capped tolerance penalties::

        NSE - w * clip(mean_e |peak lag_e| / 6 h,        0, 3)
            - w * clip(|log(mean peak ratio)| / 0.20,    0, 3)
            - w * clip(|mean event volume bias| / 0.15,  0, 3)

    with ``w = 0.05`` for all three terms (``COMPOSITE_*_WEIGHT``), the
    tolerances taken from ``DIAGNOSTICS`` (the same scales the rule/EKI/numeric
    arms use) and the peak lag averaged over events *after* taking absolute
    values.  The total penalty is bounded by 0.45, so the NSE term always
    dominates.

All optimizers *maximise* the objective value.  Whatever the objective, the
history payload also carries ``running_best_nse`` / ``best_by_nse`` (the
cumulative best full-period NSE), so equal-budget comparisons across arms are
always made in NSE units.


RULE TABLE PROVENANCE
=====================
Every entry of ``RULE_TABLE`` is a literal translation of one sentence of the
EF5 parameter guide that the LLM agents receive
(``hydrocalib/agents/physics_info.py``).  No hydrology beyond those sentences is
invented here.  The sentences are reproduced verbatim in ``GUIDE_SENTENCES``
(including the em dashes and Greek letters of the original) and referenced by
tag; :func:`check_rule_provenance` verifies at run time that every sentence is
a literal substring of the guide the LLM agents receive and warns otherwise.

G1  "WM controls the total soil water storage capacity; higher WM increases
     infiltration and reduces runoff."
       -> too much runoff (volume / mid-flow / event volume too high): wm up.
       -> flashy FDC mid-segment: wm up (more storage damps the response).
G2  "B defines the shape of the variable infiltration curve; larger B yields
     more surface runoff for a given soil moisture."
       -> too much runoff or too-high high flows / event peaks: b down.
       -> too little quick response (BFI too high, FDC too flat): b up.
G3  "IM is the impervious area fraction—higher values reduce infiltration and
     increase runoff."
       -> too much runoff or too-high high flows / event peaks: im down.
       -> BFI too high (response not flashy enough): im up.
G4  "KE scales potential evapotranspiration (PET); larger KE increases
     evaporation and decreases runoff."
       -> total or low-flow volume too high: ke up.
G5  "FC is the saturated hydraulic conductivity; higher FC allows faster
     infiltration, reducing surface flow."
       -> too much runoff or too-high high flows / event peaks: fc up.
       -> BFI too high / FDC too flat: fc down (more surface flow).
G8  "UNDER controls interflow velocity—higher values accelerate subsurface
     flow."
       -> recessions too fast: under down;  recessions too slow: under up.
       -> peaks late: under up;  peaks early: under down.
       -> BFI too high: under up;  low flows too high: under up.
G9  "LEAKI defines the leakage rate from the interflow layer; higher LEAKI
     speeds lateral drainage."
       -> same directions as G8 (recession, timing, BFI, low flow).
G11 "ALPHA and BETA are routing parameters in the discharge equation
     Q = αA^β; increasing either slows wave propagation and broadens flood
     peaks."
       -> high flows / event peaks too high: alpha up, beta up (broadening
          lowers and widens the peak).
       -> peaks arrive late: alpha down, beta down (faster propagation).
       -> recessions too fast / FDC too steep: alpha up, beta up.
G12 "ALPHA0 applies the same relationship for non-channel cells."
       -> alpha0 follows alpha/beta for the peak-magnitude and timing rules.
G13 "During calibration, adjust WM, B, IM, and FC to shape runoff volume; tune
     KE for evapotranspiration balance; and modify ALPHA, BETA, UNDER, and
     LEAKI to match hydrograph timing and attenuation."
       -> used to set the *weights*: volume diagnostics act on wm/b/im/fc(/ke),
          timing and attenuation diagnostics act on alpha/beta/under/leaki.

G6 (IWU), G7 (TH) and G10 (ISU) describe the three frozen parameters and are
therefore not encoded.
"""

from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

from .config import (
    DEFAULT_GAUGE_NUM,
    DEFAULT_PEAK_PICK_KWARGS,
    FROZEN_PARAMETERS,
    PARAM_BOUNDS,
)
from .metrics import aggregate_event_metrics, compute_event_metrics, read_metrics_from_csv
from .parameters import ParameterSet, safe_clip
from .zoned import base_name, is_zoned, zone_of
from .peak_events import pick_peak_events
from .signatures import compute_signature_summary
from .simulation import SimulationRunner, run_simulations_parallel

METHODS = ("rule", "random_local", "zone_random_local", "sceua", "cmaes", "bo", "eki", "dream",
           # structure-aware classical baselines for zoned problems (BlockCoordinateOptimizer)
           "blk_seq_eki", "blk_cyc_eki", "blk_lump_eki", "blk_cyc_cmaes",
           # the rule proposer applied zone by zone (ZoneRuleProposer); _mz adds candidates
           # that apply every zone's rule pushes at once
           "zone_rule", "zone_rule_mz")
OBJECTIVES = ("nse", "composite", "multisite_nse")

# Stable integer ids used to derive an independent random sub-stream per
# optimizer: ``default_rng([seed, METHOD_IDS[method]])``.  Without this every
# method shared ``default_rng(seed)`` and BO/EKI evaluated the very same LHS
# design for a given seed.  Never renumber.
METHOD_IDS: Dict[str, int] = {
    "rule": 1, "sceua": 2, "cmaes": 3, "bo": 4, "eki": 5, "random_local": 6, "dream": 7,
    "zone_random_local": 8,
    "blk_seq_eki": 9, "blk_cyc_eki": 10, "blk_lump_eki": 11, "blk_cyc_cmaes": 12,
    "zone_rule": 13, "zone_rule_mz": 14,
}

# --- composite objective: NSE minus dimensionless, capped tolerance penalties ---
# score = NSE - w_lag * clip(mean_e|lag_e| / tol_lag, 0, cap)
#             - w_peak * clip(|log(mean peak ratio)| / tol_peak, 0, cap)
#             - w_vol * clip(|mean event volume bias| / tol_vol, 0, cap)
# The tolerances are read from DIAGNOSTICS (event_peak_lag 6 h,
# event_peak_ratio 0.20, event_volume_bias 0.15) so every arm uses one scale.
COMPOSITE_LAG_WEIGHT = 0.05
COMPOSITE_PEAK_WEIGHT = 0.05
COMPOSITE_VOLUME_WEIGHT = 0.05
COMPOSITE_PENALTY_CAP = 3.0      # each term is capped at cap*weight (= 0.15)

# --- rule proposer ---
RULE_STEP_GRID: Tuple[float, ...] = (0.25, 0.5, 0.67, 1.5, 2.0, 4.0)
RULE_ADDITIVE_FRACTION = {0.25: -0.40, 0.5: -0.25, 0.67: -0.10,
                          1.5: 0.10, 2.0: 0.25, 4.0: 0.40}
RULE_MIN_DEVIATION = 0.15        # ignore diagnostics closer than this to target
# Standardised deviations are soft-saturated as S*tanh(dev/S): the ordering
# and (up to ~2S) the magnitude of the raw deviation are preserved, whereas the
# former hard clip at +-3 saturated all ten diagnostics at the INCA baseline.
RULE_DEVIATION_SOFT_SCALE = 10.0
RULE_DEVIATION_CLIP = 3.0        # legacy hard clip (kept for the config record)
# Adaptive step tier: multiplier 4**(|push|/RULE_ADAPTIVE_PUSH_SCALE), capped
# at RULE_MAX_STEP, floored at RULE_MIN_ADAPTIVE_STEP.
RULE_ADAPTIVE_PUSH_SCALE = 10.0
RULE_MAX_STEP = 4.0
RULE_MIN_ADAPTIVE_STEP = 1.10
RULE_RANDOM_LOCAL_MAX_SUBSET = 5  # random_local / fill_budget: 1..5 params per move

# What the proposer looked like before the design audit (recorded in
# RuleProposer.config() next to the current settings so the appendix can show
# both).
LEGACY_RULE_PROPOSER: Dict[str, Any] = {
    "step_grid": [0.5, 0.67, 1.5, 2.0],
    "deviation_saturation": "hard clip",
    "deviation_clip": 3.0,
    "deterministic_subset_sizes": [1, 1, 2, 3, 4],
    "mixed_plan": True,
    "random_subset_max": {"normal": 4, "aggressive": 5},
    "adaptive_step": False,
    "full_push_candidate": False,
    "fill_budget": False,
}
RULE_PROPOSER_SETTINGS: Dict[str, Any] = {
    "step_grid": list(RULE_STEP_GRID),
    "deviation_saturation": "soft (S*tanh(dev/S))",
    "deviation_soft_scale": RULE_DEVIATION_SOFT_SCALE,
    "deterministic_plans": [
        "top1@moderate", "top1@strong", "top2@strong", "top3@moderate",
        "all_pushed@adaptive", "all_pushed@moderate", "top3@extreme",
        "top1@strong+top2@moderate", "momentum",
    ],
    "random_subset_max": "all pushed parameters",
    "adaptive_step": f"4**(|push|/{RULE_ADAPTIVE_PUSH_SCALE}), "
                     f"in [{RULE_MIN_ADAPTIVE_STEP}, {RULE_MAX_STEP}]",
    "full_push_candidate": True,
}

# --- SCE-UA (Duan et al. 1994 recommendations, as a function of dimension) ---
SCEUA_DEFAULT_COMPLEXES = 2


def sceua_defaults(dim: int) -> Dict[str, int]:
    """Duan et al. (1994): m = 2n+1, q = n+1, beta = 2n+1; p = 2 complexes."""
    n = max(1, int(dim))
    return {"complexes": SCEUA_DEFAULT_COMPLEXES, "complex_size": 2 * n + 1,
            "subset_size": n + 1, "evolutions_per_complex": 2 * n + 1}


# --- CMA-ES ---
CMAES_SIGMA0 = 0.30
CMAES_BOX_PENALTY = 1.0

# --- Bayesian optimisation ---
BO_INIT_MULTIPLIER = 2           # initial LHS design size = multiplier * dim
BO_MAX_GP_POINTS = 300
BO_HYPER_REFIT_EVERY = 10
BO_EI_XI = 0.01
BO_EI_RANDOM_STARTS = 2000
BO_EI_LOCAL_STARTS = 5
BO_DEFAULT_BATCH = 4

# --- Ensemble Kalman Inversion ---
# DREAM-ZS (Ter Braak & Vrugt 2008).  EF5's built-in DREAM hardwires its dimension
# to the model's parameter count (DREAM.cpp:76 `pointerMCMC->n = numParams`, i.e.
# 6 CREST + 7 KW = 13), takes one chain per dimension (seq = n), and reads its bounds
# from one [CrestCaliParams] section carrying a single GAUGE=, so it cannot address the
# zoned 10*(k+1) vector at all.  This re-implementation keeps what makes the comparison
# to the published chain meaningful -- the same box and the same acceptance density,
# log L = -(MaxT/2) * log(1 - NSCE) (dream_functions.cpp:616-631 with Gamma = 0,
# MaxT = totalTimeStepsOutsideWarm) -- but uses the ZS variant (few chains + a
# past-states archive Z), which is what makes DREAM usable above ~20 dimensions.
DREAM_DEFAULT_CHAINS = 3          # ZS standard; EF5 instead uses seq = dim
DREAM_DEFAULT_NCR = 3             # crossover levels, as EF5 (DREAM.cpp:126)
DREAM_DEFAULT_DELTA = 3           # DE pairs, as EF5 (DREAM.cpp:128)
DREAM_SNOOKER_PROB = 0.10         # Ter Braak & Vrugt 2008 recommend 0.1
DREAM_INIT_PER_DIM = 3            # Z archive is seeded with 3*dim Latin-hypercube points
DREAM_GAMMA_JUMP_EVERY = 5        # every 5th generation takes gamma = 1 (mode jumping)
DREAM_LIKELIHOODS = ("ef5_sse", "nse_half")
DREAM_SSE_N = 8760                # EF5's MaxT = totalTimeStepsOutsideWarm (2018, hourly)


EKI_DEFAULT_ENSEMBLE = 20
EKI_INFLATION = 1.02
EKI_RESIDUAL_CLIP = 5.0


GUIDE_SENTENCES: Dict[str, str] = {
    "G1": ("WM controls the total soil water storage capacity; higher WM increases "
           "infiltration and reduces runoff."),
    "G2": ("B defines the shape of the variable infiltration curve; larger B yields "
           "more surface runoff for a given soil moisture."),
    "G3": ("IM is the impervious area fraction—higher values reduce infiltration and "
           "increase runoff."),
    "G4": ("KE scales potential evapotranspiration (PET); larger KE increases "
           "evaporation and decreases runoff."),
    "G5": ("FC is the saturated hydraulic conductivity; higher FC allows faster "
           "infiltration, reducing surface flow."),
    "G8": ("UNDER controls interflow velocity—higher values accelerate subsurface flow."),
    "G9": ("LEAKI defines the leakage rate from the interflow layer; higher LEAKI "
           "speeds lateral drainage."),
    "G11": ("ALPHA and BETA are routing parameters in the discharge equation "
            "Q = αA^β; increasing either slows wave propagation and broadens "
            "flood peaks."),
    "G12": ("ALPHA0 applies the same relationship for non-channel cells."),
    "G13": ("During calibration, adjust WM, B, IM, and FC to shape runoff volume; tune "
            "KE for evapotranspiration balance; and modify ALPHA, BETA, UNDER, and "
            "LEAKI to match hydrograph timing and attenuation."),
}

_GUIDE_ATTR = "PHYSICS_PARAMETER_GUIDE"
_PHYSICS_INFO_PATH = Path(__file__).resolve().parent / "agents" / "physics_info.py"


def guide_text() -> str:
    """Return the parameter guide handed to the LLM agents (empty if unavailable).

    The string literal is read from ``agents/physics_info.py`` with :mod:`ast`
    so that no LLM client (and no API key) is needed; importing the package is
    only a fallback.
    """
    try:
        tree = ast.parse(_PHYSICS_INFO_PATH.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == _GUIDE_ATTR for t in node.targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, str) and value:
                    return value
    except Exception:  # noqa: BLE001 - fall through to the import path
        pass
    try:
        from .agents import physics_info as _physics_info
    except Exception:  # pragma: no cover - the agents package may pull in openai
        return ""
    text = getattr(_physics_info, _GUIDE_ATTR, None)
    return text if isinstance(text, str) else ""


def _normalize_guide_text(text: str) -> str:
    """Typographic normalisation used only as a *fallback* comparison."""
    for src, dst in (("—", "-"), ("–", "-"), ("−", "-"),
                     ("α", "alpha"), ("β", "beta"),
                     ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


def check_rule_provenance(verbose: bool = True) -> Dict[str, Any]:
    """Verify that every ``GUIDE_SENTENCES`` entry is a literal substring of the guide.

    Returns ``{"ok": bool|None, "verbatim": [...], "normalized_only": [...],
    "missing": [...], "guide_sha256": str|None}``.  ``ok`` is ``None`` when the
    guide text could not be located.  Sentences that only match after
    typographic normalisation (dashes, Greek letters, whitespace) are reported
    separately; sentences that do not match at all are printed as WARN.
    """
    text = guide_text()
    result: Dict[str, Any] = {"ok": None, "verbatim": [], "normalized_only": [],
                              "missing": [], "guide_sha256": None,
                              "rules_with_missing_guide": []}
    if not text:
        if verbose:
            print("[WARN] rule provenance: parameter guide text unavailable; "
                  "GUIDE_SENTENCES could not be verified", flush=True)
        return result
    result["guide_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    norm_text = _normalize_guide_text(text)
    for tag, sentence in GUIDE_SENTENCES.items():
        if sentence in text:
            result["verbatim"].append(tag)
        elif _normalize_guide_text(sentence) in norm_text:
            result["normalized_only"].append(tag)
        else:
            result["missing"].append(tag)
    missing_or_loose = set(result["missing"]) | set(result["normalized_only"])
    result["rules_with_missing_guide"] = sorted({
        f"{r.diagnostic}->{r.param}" for r in RULE_TABLE if r.guide in missing_or_loose
    })
    result["ok"] = not result["missing"] and not result["normalized_only"]
    if verbose:
        for tag in result["normalized_only"]:
            print(f"[WARN] rule provenance: {tag} matches the guide only after "
                  f"typographic normalisation (not verbatim)", flush=True)
        for tag in result["missing"]:
            print(f"[WARN] rule provenance: {tag} is NOT found in the parameter guide: "
                  f"{GUIDE_SENTENCES[tag]!r}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# Provenance fingerprints
# --------------------------------------------------------------------------- #
def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=10, check=False)
    except Exception:  # noqa: BLE001 - git may be missing entirely
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def rule_table_sha256() -> str:
    """Fingerprint of the knowledge the rule arm has: rules + diagnostics + guide sentences."""
    blob = json.dumps({
        "rule_table": [[r.diagnostic, r.param, r.response, r.weight, r.guide] for r in RULE_TABLE],
        "diagnostics": [[d.key, d.source, d.field, d.target, d.scale, d.log_transform]
                        for d in DIAGNOSTICS],
        "guide_sentences": GUIDE_SENTENCES,
    }, sort_keys=True, ensure_ascii=False)
    return _sha256_text(blob)


def provenance_block(verbose: bool = False) -> Dict[str, Any]:
    """Fingerprints written into every history payload (git HEAD, file hashes, guide)."""
    here = Path(__file__).resolve()
    repo = here.parent.parent
    head = _git(repo, "rev-parse", "HEAD")
    dirty: Optional[bool] = None
    if head is not None:
        status = _git(repo, "status", "--porcelain")
        dirty = bool(status) if status is not None else None
    guide = guide_text()
    check = check_rule_provenance(verbose=verbose)
    return {
        "git_head": head,
        "git_dirty": dirty,
        "repo_dir": str(repo),
        "optimizers_py_sha256": _sha256_file(here),
        "rule_table_sha256": rule_table_sha256(),
        "guide_sha256": _sha256_text(guide) if guide else None,
        "guide_source": str(_PHYSICS_INFO_PATH) if guide else None,
        "guide_provenance_ok": check["ok"],
        "guide_sentences_not_verbatim": check["normalized_only"] + check["missing"],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
class SearchSpace:
    """The free calibration parameters and the maps between real and unit space."""

    def __init__(self, names: Optional[Sequence[str]] = None):
        self.names: List[str] = list(names) if names else [
            n for n in PARAM_BOUNDS if n not in FROZEN_PARAMETERS
        ]
        # zoned names ('wm@V1') take the bounds of their base parameter
        self.lower = np.array([PARAM_BOUNDS[base_name(n)][0] for n in self.names], dtype=float)
        self.upper = np.array([PARAM_BOUNDS[base_name(n)][1] for n in self.names], dtype=float)
        self.width = np.maximum(self.upper - self.lower, 1e-12)

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_unit(self, x_real: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(x_real, dtype=float) - self.lower) / self.width, 0.0, 1.0)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        return self.lower + u * self.width

    def clip_real(self, x_real: Sequence[float]) -> np.ndarray:
        return np.array([safe_clip(n, float(v)) for n, v in zip(self.names, x_real)], dtype=float)

    def vector_of(self, params: ParameterSet) -> np.ndarray:
        return np.array([float(params.values[n]) for n in self.names], dtype=float)

    def build_params(self, base: ParameterSet, x_real: Sequence[float]) -> ParameterSet:
        """Return a copy of ``base`` with the free parameters replaced by ``x_real``."""
        values = base.values.copy()
        for name, val in zip(self.names, x_real):
            values[name] = safe_clip(name, float(val))
        return ParameterSet(values)

    def params_from_unit(self, base: ParameterSet, u: Sequence[float]) -> ParameterSet:
        return self.build_params(base, self.from_unit(np.asarray(u, dtype=float)))


def latin_hypercube(rng: np.random.Generator, n_samples: int, n_dims: int) -> np.ndarray:
    """Latin hypercube sample on the unit cube (no external dependency)."""
    out = np.empty((n_samples, n_dims), dtype=float)
    jitter = rng.random((n_samples, n_dims))
    for j in range(n_dims):
        out[:, j] = (rng.permutation(n_samples) + jitter[:, j]) / float(n_samples)
    return out


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars and non-finite floats into JSON values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# --------------------------------------------------------------------------- #
# Evaluation bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationRecord:
    """One EF5 forward evaluation."""

    eval_index: int
    iteration: int
    candidate_index: int
    params: Dict[str, float]
    objective_value: float
    full_metrics: Dict[str, float] = field(default_factory=dict)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0
    failed: bool = False
    error: Optional[str] = None
    round_label: str = ""
    output_dir: str = ""
    csv_path: str = ""
    signatures: Optional[Dict[str, Any]] = None
    # daily normalised residual vector (EKI 'series' observation operator); never written
    # to the history file -- it is recomputed from the kept incumbent when needed
    residual_series: Optional[Any] = None

    def to_json(self, include_signatures: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "eval_index": self.eval_index,
            "iteration": self.iteration,
            "candidate_index": self.candidate_index,
            "params": self.params,
            "objective_value": self.objective_value,
            "full_metrics": self.full_metrics,
            "aggregate_metrics": self.aggregate_metrics,
            "wall_time_s": self.wall_time_s,
        }
        if self.failed:
            payload["failed"] = True
            payload["error"] = self.error
        if self.round_label:
            payload["round_label"] = self.round_label
        if include_signatures and self.signatures is not None:
            payload["signatures"] = self.signatures
        return _jsonable(payload)


class BaseEvaluator:
    """Budget-aware batch evaluator shared by every optimizer.

    Subclasses implement :meth:`_run_batch`; the base class owns the budget,
    the evaluation log and the running-best trace.
    """

    def __init__(self,
                 method: str,
                 objective: str = "nse",
                 budget: int = 1000,
                 verbose: bool = True):
        self.method = method
        self.objective = objective
        self.budget = int(budget)
        self.verbose = verbose
        self.records: List[EvaluationRecord] = []
        self.running_best: List[float] = []
        self.best: Optional[EvaluationRecord] = None
        self.best_value: float = float("-inf")
        # NSE-denominated trace kept regardless of the objective so that every
        # arm can be read on the same equal-budget axis (B6 optimises a
        # composite score; its running_best is not an NSE)
        self.running_best_nse: List[float] = []
        self.best_by_nse: Optional[EvaluationRecord] = None
        self.best_nse_value: float = float("-inf")
        # optional checkpoint hook called after every batch; failures are ignored
        self.on_batch: Optional[Callable[["BaseEvaluator"], None]] = None

    # -- budget ------------------------------------------------------------ #
    @property
    def n_evaluations(self) -> int:
        return len(self.records)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.n_evaluations)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    # -- API --------------------------------------------------------------- #
    def evaluate(self,
                 params_list: Sequence[ParameterSet],
                 iteration: int,
                 stage: int = 0) -> List[EvaluationRecord]:
        """Run one parallel batch, truncated to the remaining budget."""
        if self.exhausted or not params_list:
            return []
        batch = list(params_list)[: self.remaining]
        round_label = f"{self.method}_{iteration:04d}"
        if stage:
            round_label = f"{round_label}_s{stage}"
        records = self._run_batch(batch, round_label, iteration)
        for rec in records:
            rec.eval_index = len(self.records)
            self.records.append(rec)
            if math.isfinite(rec.objective_value) and rec.objective_value > self.best_value:
                self.best_value = rec.objective_value
                self.best = rec
            self.running_best.append(self.best_value)
            nse = _record_nse(rec)
            if math.isfinite(nse) and nse > self.best_nse_value:
                self.best_nse_value = nse
                self.best_by_nse = rec
            self.running_best_nse.append(self.best_nse_value)
        if self.verbose:
            shown = ", ".join(
                f"{r.objective_value:.4f}" if math.isfinite(r.objective_value) else "fail"
                for r in records
            )
            print(f"  [{round_label}] {len(records)} eval(s): [{shown}] | "
                  f"best={self.best_value:.4f} | used {self.n_evaluations}/{self.budget}",
                  flush=True)
        if self.on_batch is not None:
            try:
                self.on_batch(self)
            except Exception as exc:  # noqa: BLE001 - checkpointing must never abort a run
                print(f"  [warn] checkpoint hook failed: {exc!r}", flush=True)
        return records

    def _run_batch(self,
                   params_list: Sequence[ParameterSet],
                   round_label: str,
                   iteration: int) -> List[EvaluationRecord]:
        raise NotImplementedError


@dataclass
class _FailedRun:
    """Stand-in for ``SimulationResult`` when EF5 raised."""

    round_label: str
    candidate_index: int
    params: ParameterSet
    error: str
    output_dir: str = ""
    csv_path: str = ""


class _SafeRunner:
    """Wrap a ``SimulationRunner`` so a crashed EF5 run cannot abort the batch."""

    def __init__(self, runner: SimulationRunner):
        self._runner = runner

    def run(self, params: ParameterSet, round_label: str, candidate_index: int):
        try:
            return self._runner.run(params, round_label, candidate_index)
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            print(f"    [Sim] Candidate {candidate_index} FAILED: {exc!r}", flush=True)
            return _FailedRun(round_label, candidate_index, params.copy(), repr(exc))


def _record_nse(rec: "EvaluationRecord") -> float:
    """Full-period NSE of a record (nan when unavailable)."""
    try:
        val = float((rec.full_metrics or {}).get("NSE", float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def composite_tolerances() -> Dict[str, float]:
    """Tolerances of the composite penalties, taken from ``DIAGNOSTICS``."""
    return {
        "lag_hours": DIAGNOSTIC_BY_KEY["event_peak_lag"].scale,          # 6.0 h
        "log_peak_ratio": DIAGNOSTIC_BY_KEY["event_peak_ratio"].scale,   # 0.20
        "event_volume_bias": DIAGNOSTIC_BY_KEY["event_volume_bias"].scale,  # 0.15
    }


def composite_penalties(full_metrics: Dict[str, float],
                        aggregate_metrics: Dict[str, float],
                        signatures: Optional[Dict[str, Any]],
                        event_metrics: Optional[Sequence[Dict[str, Any]]] = None
                        ) -> Dict[str, float]:
    """The three dimensionless penalty terms of the composite objective.

    Each term is ``weight * clip(|deviation| / tolerance, 0, COMPOSITE_PENALTY_CAP)``:

    * ``lag``: the peak timing error is taken *per event* in absolute value
      and then averaged (``event_metrics[i]["lag_hours"]``; falling back to the
      per-event ``peak_lag_hours_sim_minus_obs`` of the signatures, and only
      then to the signed aggregate mean), so an early and a late event no
      longer cancel;
    * ``peak``: ``|log(mean event peak ratio)|`` from ``aggregate_event_metrics``;
    * ``volume``: ``|mean event_volume_bias_frac|`` from the signatures.

    Missing or non-finite inputs contribute a zero penalty.
    """
    tol = composite_tolerances()
    cap = COMPOSITE_PENALTY_CAP

    lags: List[float] = []
    if event_metrics:
        for m in event_metrics:
            try:
                v = float(m.get("lag_hours", float("nan")))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                lags.append(abs(v))
    if not lags and signatures:
        for e in (signatures.get("events") or []):
            v = e.get("peak_lag_hours_sim_minus_obs")
            if v is not None and math.isfinite(float(v)):
                lags.append(abs(float(v)))
    if not lags:
        v = float((aggregate_metrics or {}).get("lag_hours", float("nan")))
        if math.isfinite(v):
            lags.append(abs(v))
    lag_term = 0.0
    if lags:
        lag_term = COMPOSITE_LAG_WEIGHT * min(cap, float(np.mean(lags)) / tol["lag_hours"])

    peak_term = 0.0
    peak_ratio = float((aggregate_metrics or {}).get("peak_ratio", float("nan")))
    if math.isfinite(peak_ratio) and peak_ratio > 0:
        peak_term = COMPOSITE_PEAK_WEIGHT * min(cap, abs(math.log(peak_ratio)) / tol["log_peak_ratio"])

    vol_term = 0.0
    vol = _mean_event_signature(signatures, "event_volume_bias_frac")
    if math.isfinite(vol):
        vol_term = COMPOSITE_VOLUME_WEIGHT * min(cap, abs(vol) / tol["event_volume_bias"])

    return {"lag": float(lag_term), "peak": float(peak_term), "volume": float(vol_term)}


def composite_objective(full_metrics: Dict[str, float],
                        aggregate_metrics: Dict[str, float],
                        signatures: Optional[Dict[str, Any]],
                        event_metrics: Optional[Sequence[Dict[str, Any]]] = None) -> float:
    """``NSE`` minus three capped, dimensionless tolerance penalties.

    ``score = NSE - w_lag  * clip(mean_e |lag_e| / 6 h, 0, 3)
                  - w_peak * clip(|log(mean peak ratio)| / 0.20, 0, 3)
                  - w_vol  * clip(|mean event volume bias| / 0.15, 0, 3)``

    with ``w = 0.05`` each (``COMPOSITE_*_WEIGHT``), so the penalty never
    exceeds 0.45 and no single term can dominate the NSE.  See
    :func:`composite_penalties` for the term definitions.  A non-finite NSE
    gives ``-inf``.
    """
    nse = float(full_metrics.get("NSE", float("nan")))
    if not math.isfinite(nse):
        return float("-inf")
    pen = composite_penalties(full_metrics, aggregate_metrics, signatures, event_metrics)
    return float(nse - pen["lag"] - pen["peak"] - pen["volume"])


def _mean_event_signature(signatures: Optional[Dict[str, Any]], key: str) -> float:
    """Mean of a per-event signature over the selected events."""
    if not signatures:
        return float("nan")
    events = signatures.get("events") or []
    vals = [float(e[key]) for e in events
            if e.get(key) is not None and math.isfinite(float(e[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def daily_normalised_residuals(csv_path: str, n_gauges: int = 1) -> np.ndarray:
    """Daily-mean residuals (sim - obs) / sqrt(n_gauges * S), S = sum_d (obs_d - mean obs)^2.

    Chosen so that the squared norm of the concatenated vector over all gauges equals
    1 - mean_g NSE_g (daily), i.e. the EKI 'series' operator drives exactly the multi-site
    objective (with daily instead of hourly aggregation, to keep the observation vector at
    a few thousand entries). Same rows as read_metrics_from_csv (both series finite).
    """
    import pandas as pd  # local import: pandas is already a dependency of metrics.py
    df = pd.read_csv(csv_path, usecols=["Time", "Discharge(m^3 s^-1)", "Observed(m^3 s^-1)"])
    t = pd.to_datetime(df["Time"])
    sim = pd.to_numeric(df["Discharge(m^3 s^-1)"], errors="coerce")
    obs = pd.to_numeric(df["Observed(m^3 s^-1)"], errors="coerce")
    ok = sim.notna() & obs.notna()
    day = t[ok].dt.floor("D")
    sd = sim[ok].groupby(day.values).mean().to_numpy(dtype=float)
    od = obs[ok].groupby(day.values).mean().to_numpy(dtype=float)
    S = float(np.sum((od - od.mean()) ** 2))
    if S <= 0 or len(od) == 0:
        return np.zeros(0, dtype=float)
    return (sd - od) / math.sqrt(max(1, n_gauges) * S)


class EF5Evaluator(BaseEvaluator):
    """Runs EF5 in parallel and turns each output CSV into an objective value."""

    def __init__(self,
                 runner: SimulationRunner,
                 method: str,
                 objective: str = "nse",
                 budget: int = 1000,
                 n_peaks: int = 3,
                 max_workers: int = 8,
                 peak_pick_kwargs: Optional[Dict[str, Any]] = None,
                 need_signatures: bool = False,
                 multisite_gauges: Optional[Sequence[str]] = None,
                 residual_series: bool = False,
                 verbose: bool = True):
        super().__init__(method=method, objective=objective, budget=budget, verbose=verbose)
        # EKI 'series' operator: keep the daily normalised residual vector of every run
        # (computed here, before the run's outputs are discarded)
        self.residual_series = bool(residual_series)
        self.runner = runner
        self.safe_runner = _SafeRunner(runner)
        self.n_peaks = n_peaks
        self.max_workers = max_workers
        self.peak_pick_kwargs = dict(peak_pick_kwargs or DEFAULT_PEAK_PICK_KWARGS)
        self.need_signatures = need_signatures or objective == "composite"
        # With real interior gauges the objective is the mean NSE over all of them; the
        # outlet's own metrics are still recorded separately so the number stays
        # comparable with the single-gauge runs.
        self.multisite_gauges: List[str] = [str(g) for g in (multisite_gauges or [])]
        # incumbent whose outputs are kept on disk; everything else is deleted as soon
        # as it has been scored (see _discard_superseded)
        self._kept: Optional[Tuple[str, int]] = None
        self._kept_value: float = float("-inf")

    def _run_batch(self,
                   params_list: Sequence[ParameterSet],
                   round_label: str,
                   iteration: int) -> List[EvaluationRecord]:
        start = time.time()
        results = run_simulations_parallel(
            self.safe_runner, list(params_list), round_label, self.max_workers
        )
        batch_wall = time.time() - start
        per_run = batch_wall / max(1, len(results))

        records: List[EvaluationRecord] = []
        for res in results:
            params = res.params if hasattr(res, "params") else params_list[res.candidate_index]
            if isinstance(res, _FailedRun):
                records.append(EvaluationRecord(
                    eval_index=-1,
                    iteration=iteration,
                    candidate_index=res.candidate_index,
                    params=dict(params.values),
                    objective_value=float("-inf"),
                    wall_time_s=per_run,
                    failed=True,
                    error=res.error,
                    round_label=round_label,
                ))
                continue
            records.append(self._score(res, iteration, round_label, per_run))
        self._discard_superseded(records)
        return records

    def _discard_superseded(self, records: List[EvaluationRecord]) -> None:
        """Keep the incumbent's outputs, delete everyone else's.

        Only the incumbent's series is ever read again (the job copies it back at the end),
        so holding 2000 evaluations on the node's local disk buys nothing and makes the run
        fail if another job fills /tmp halfway through -- which is how 18 of the first 125
        multi-site runs died. Disabled by INCA_KEEP_EVALS=1 when a full trace is wanted.
        """
        if os.getenv("INCA_KEEP_EVALS") == "1":
            return
        for rec in records:
            v = rec.objective_value
            if rec.failed or v is None or not math.isfinite(float(v)):
                keep = False
            else:
                keep = float(v) > self._kept_value
            if keep:
                prev = self._kept
                self._kept = (rec.round_label, rec.candidate_index)
                self._kept_value = float(v)
                if prev is not None and prev != self._kept:
                    self.runner.discard(prev[0], prev[1])
            else:
                if (rec.round_label, rec.candidate_index) != self._kept:
                    self.runner.discard(rec.round_label, rec.candidate_index)

    def _gauge_nse(self, res) -> Dict[str, float]:
        """NSE at every gauge of a multi-site run, read from its own ts.<id>.crest.csv.

        EF5 lower-cases the gauge id in the file name (Simulator.cpp), and a gauge with no
        OBS gets a NaN Observed column, which read_metrics_from_csv turns into a NaN NSE --
        so a missing record shows up as non-finite rather than as a silent zero.
        """
        out = Path(getattr(res, "output_dir", "") or "")
        vals: Dict[str, float] = {}
        for gid in self.multisite_gauges:
            path = out / f"ts.{gid.lower()}.crest.csv"
            if not path.exists():
                vals[gid] = float("nan")
                continue
            try:
                vals[gid] = float(read_metrics_from_csv(str(path)).get("NSE", float("nan")))
            except Exception:  # noqa: BLE001
                vals[gid] = float("nan")
        return vals

    def _score(self, res, iteration: int, round_label: str, wall_time: float) -> EvaluationRecord:
        params = dict(res.params.values)
        try:
            full_metrics = read_metrics_from_csv(res.csv_path)
            windows = pick_peak_events(res.csv_path, n=self.n_peaks, **self.peak_pick_kwargs)
            event_metrics = compute_event_metrics(res.csv_path, windows)
            aggregate = aggregate_event_metrics(event_metrics, top_n=self.n_peaks)
            signatures: Optional[Dict[str, Any]] = None
            if self.need_signatures:
                window_dicts = [{"start": str(a), "end": str(b)} for a, b in windows]
                _, signatures = compute_signature_summary(res.csv_path, window_dicts)
            per_gauge = self._gauge_nse(res) if self.multisite_gauges else {}
            if per_gauge:
                aggregate = dict(aggregate)
                aggregate.update({f"NSE@{g}": v for g, v in per_gauge.items()})
            if self.objective == "multisite_nse":
                vals = [per_gauge.get(g, float("nan")) for g in self.multisite_gauges]
                # every gauge must have scored: one missing series would otherwise raise
                # the mean of the rest and look like an improvement
                value = (sum(vals) / len(vals)
                         if vals and all(math.isfinite(v) for v in vals) else float("-inf"))
            elif self.objective == "composite":
                value = composite_objective(full_metrics, aggregate, signatures, event_metrics)
                pen = composite_penalties(full_metrics, aggregate, signatures, event_metrics)
                aggregate = dict(aggregate)
                aggregate.update({f"composite_pen_{k}": v for k, v in pen.items()})
            else:
                nse = float(full_metrics.get("NSE", float("nan")))
                value = nse if math.isfinite(nse) else float("-inf")
            failed = not math.isfinite(value)
            resid = None
            if self.residual_series:
                try:
                    if self.multisite_gauges:
                        out = Path(getattr(res, "output_dir", "") or "")
                        parts = [daily_normalised_residuals(str(out / f"ts.{g.lower()}.crest.csv"),
                                                            len(self.multisite_gauges))
                                 for g in self.multisite_gauges]
                        resid = np.concatenate(parts) if parts else None
                    else:
                        resid = daily_normalised_residuals(res.csv_path, 1)
                except Exception:  # noqa: BLE001
                    resid = None
            return EvaluationRecord(
                eval_index=-1,
                iteration=iteration,
                candidate_index=res.candidate_index,
                params=params,
                objective_value=float(value),
                full_metrics=full_metrics,
                aggregate_metrics=aggregate,
                wall_time_s=wall_time,
                failed=failed,
                error="non-finite objective" if failed else None,
                round_label=round_label,
                output_dir=getattr(res, "output_dir", ""),
                csv_path=getattr(res, "csv_path", ""),
                signatures=signatures,
                residual_series=resid,
            )
        except Exception as exc:  # noqa: BLE001 - metrics must never abort a run
            return EvaluationRecord(
                eval_index=-1,
                iteration=iteration,
                candidate_index=res.candidate_index,
                params=params,
                objective_value=float("-inf"),
                wall_time_s=wall_time,
                failed=True,
                error=f"metric failure: {exc!r}",
                round_label=round_label,
                output_dir=getattr(res, "output_dir", ""),
                csv_path=getattr(res, "csv_path", ""),
            )


# --------------------------------------------------------------------------- #
# 1. SCE-UA
# --------------------------------------------------------------------------- #
class SceUaOptimizer:
    """Shuffled Complex Evolution (Duan, Sorooshian & Gupta 1992; Duan et al. 1994).

    The competitive complex evolution (CCE) step is implemented as published:

    1. the population of ``p * m`` points is sorted and dealt into ``p``
       complexes by rank (point ``k`` goes to complex ``k mod p``);
    2. for each of ``beta`` evolution steps a sub-complex of ``q`` points is
       drawn from the (re-sorted) complex with the trapezoidal probability
       ``P_i = 2(m+1-i) / (m(m+1))`` so that better points are more likely;
    3. the worst point ``u_q`` of the sub-complex is reflected through the
       centroid ``g`` of the other ``q-1`` points, ``r = g + (g - u_q)``.  If
       ``r`` lies outside the feasible box it is replaced by a random point in
       the smallest hypercube ``H`` containing the complex (Duan's rule; no
       clipping to the box faces);
    4. if ``r`` does not beat ``u_q`` the contraction ``c = (g + u_q) / 2`` is
       tried; if that fails too, ``u_q`` is replaced by a random point in ``H``;
    5. after ``beta`` steps for every complex the population is shuffled and
       re-partitioned.

    Default sizes follow Duan et al. (1994): ``m = 2n+1``, ``q = n+1``,
    ``beta = 2n+1`` (``alpha = 1`` offspring per step), with ``p = 2``
    complexes (see :func:`sceua_defaults`).  The only restructuring is that
    the offspring of all ``p`` complexes for one CCE stage (reflection,
    contraction, random) are evaluated in a single parallel batch.
    """

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 complexes: Optional[int] = None,
                 complex_size: Optional[int] = None,
                 subset_size: Optional[int] = None,
                 evolutions_per_complex: Optional[int] = None,
                 reflection_coef: float = 1.0,
                 contraction_coef: float = 0.5):
        self.ev = evaluator
        self.space = space
        self.base_params = base_params
        self.rng = rng
        defaults = sceua_defaults(space.dim)
        self.complexes = max(1, int(complexes or defaults["complexes"]))
        self.complex_size = max(3, int(complex_size or defaults["complex_size"]))
        q = int(subset_size or min(defaults["subset_size"], self.complex_size))
        self.subset_size = max(2, min(q, self.complex_size))
        self.evolutions_per_complex = max(1, int(evolutions_per_complex
                                                 or defaults["evolutions_per_complex"]))
        self.reflection_coef = float(reflection_coef)      # 1.0 -> r = 2g - u_q (Duan)
        self.contraction_coef = float(contraction_coef)    # 0.5 -> (g + u_q)/2  (Duan)

    def config(self) -> Dict[str, Any]:
        return {
            "complexes": self.complexes,
            "complex_size": self.complex_size,
            "subset_size": self.subset_size,
            "evolutions_per_complex": self.evolutions_per_complex,
            "population_size": self.complexes * self.complex_size,
            "reflection_coef": self.reflection_coef,
            "contraction_coef": self.contraction_coef,
            "reflection": "r = g + reflection_coef*(g - u_q)",
            "contraction": "c = u_q + contraction_coef*(g - u_q)",
            "infeasible_reflection": "random point in the smallest hypercube containing the complex",
            "duan_defaults_for_dim": sceua_defaults(self.space.dim),
        }

    def _params(self, vec: np.ndarray) -> ParameterSet:
        return self.space.build_params(self.base_params, vec)

    def _evaluate(self, vectors: Sequence[np.ndarray], iteration: int, stage: int) -> List[float]:
        recs = self.ev.evaluate([self._params(v) for v in vectors], iteration, stage=stage)
        scores = [r.objective_value for r in recs]
        scores.extend([float("-inf")] * (len(vectors) - len(scores)))
        return scores

    def run(self) -> Dict[str, Any]:
        space = self.space
        pop_size = self.complexes * self.complex_size
        population = space.lower + self.rng.random((pop_size, space.dim)) * space.width
        population[0] = space.clip_real(space.vector_of(self.base_params))

        scores = np.array(self._evaluate(list(population), 0, 0), dtype=float)
        if len(scores) < pop_size:  # budget ran out during initialisation
            scores = np.concatenate([scores, np.full(pop_size - len(scores), -np.inf)])

        iteration = 1
        while not self.ev.exhausted:
            used_before = self.ev.n_evaluations
            # shuffle: sort the whole population and deal it into complexes by rank
            order = np.argsort(-_finite_scores(scores))
            population, scores = population[order], scores[order]
            groups = [list(range(i, pop_size, self.complexes)) for i in range(self.complexes)]

            for _ in range(self.evolutions_per_complex):
                if self.ev.exhausted:
                    break
                pending: List[Tuple[int, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], float]] = []
                for members in groups:
                    if len(members) < 2:
                        continue
                    hull = (population[members].min(axis=0), population[members].max(axis=0))
                    subset = self._sample_subset(members, scores)
                    reflected, worst_idx, centroid = self._reflect(population, scores, subset, hull)
                    pending.append((worst_idx, reflected, centroid, hull, scores[worst_idx]))
                if not pending:
                    break

                new_scores = self._evaluate([p[1] for p in pending], iteration, 0)
                retry = []
                for (worst_idx, vec, centroid, hull, old_score), new_score in zip(pending, new_scores):
                    if _better(new_score, old_score):
                        population[worst_idx] = vec
                        scores[worst_idx] = new_score
                    else:
                        retry.append((worst_idx, centroid, hull, old_score))

                if retry and not self.ev.exhausted:
                    contracted = [self._contract(centroid, population[worst_idx])
                                  for worst_idx, centroid, _, _ in retry]
                    con_scores = self._evaluate(contracted, iteration, 1)
                    still = []
                    for (worst_idx, _, hull, old_score), vec, sc in zip(retry, contracted, con_scores):
                        if _better(sc, old_score):
                            population[worst_idx] = vec
                            scores[worst_idx] = sc
                        else:
                            still.append((worst_idx, hull))

                    if still and not self.ev.exhausted:
                        randoms = [self._random_in_hull(hull) for _, hull in still]
                        rand_scores = self._evaluate(randoms, iteration, 2)
                        for (worst_idx, _), vec, sc in zip(still, randoms, rand_scores):
                            population[worst_idx] = vec   # Duan step (f): unconditional
                            scores[worst_idx] = sc
            if self.ev.n_evaluations == used_before:
                # degenerate configuration (complexes too small): nothing to evolve
                break
            iteration += 1
        return {"iterations": iteration - 1, **self.config()}

    def _sample_subset(self, members: Sequence[int], scores: np.ndarray) -> List[int]:
        """Draw ``q`` members with Duan's trapezoidal probability over the *current* ranks."""
        ranked = sorted(members, key=lambda i: -_finite_scores(scores[i]))
        m = len(ranked)
        probs = np.array([2.0 * (m + 1 - i) / (m * (m + 1)) for i in range(1, m + 1)])
        probs /= probs.sum()
        size = min(self.subset_size, m)
        return [int(i) for i in self.rng.choice(ranked, size=size, replace=False, p=probs)]

    def _random_in_hull(self, hull: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Uniform sample inside the smallest hypercube containing a complex."""
        lo, hi = hull
        lo = np.clip(lo, self.space.lower, self.space.upper)
        hi = np.clip(hi, self.space.lower, self.space.upper)
        return lo + self.rng.random(self.space.dim) * np.maximum(hi - lo, 0.0)

    def _reflect(self,
                 population: np.ndarray,
                 scores: np.ndarray,
                 subset: Sequence[int],
                 hull: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, int, np.ndarray]:
        """Duan step (c): ``r = g + (g - u_q)``; infeasible -> random point in ``H``."""
        idx = list(subset)
        order = np.argsort(-_finite_scores(scores[idx]))
        worst_global = idx[order[-1]]
        centroid = np.mean(population[idx][order[:-1]], axis=0)
        worst = population[worst_global]
        candidate = centroid + self.reflection_coef * (centroid - worst)
        feasible = (np.all(np.isfinite(candidate))
                    and np.all(candidate >= self.space.lower)
                    and np.all(candidate <= self.space.upper))
        if not feasible:
            candidate = self._random_in_hull(hull)
        return candidate, worst_global, centroid

    def _contract(self, centroid: np.ndarray, worst: np.ndarray) -> np.ndarray:
        """Duan step (e): ``c = (g + u_q) / 2`` (a convex combination, always feasible)."""
        contracted = worst + self.contraction_coef * (centroid - worst)
        if not np.all(np.isfinite(contracted)):
            contracted = self.space.lower + self.rng.random(self.space.dim) * self.space.width
        return np.clip(contracted, self.space.lower, self.space.upper)


def _finite_scores(scores):
    """``-inf`` for non-finite entries so failed runs always rank last."""
    arr = np.asarray(scores, dtype=float)
    return np.where(np.isfinite(arr), arr, -np.inf)


def _better(new: float, old: float) -> bool:
    if not math.isfinite(new):
        return False
    if not math.isfinite(old):
        return True
    return new > old


# --------------------------------------------------------------------------- #
# 2. CMA-ES
# --------------------------------------------------------------------------- #
class CmaEsOptimizer:
    """(mu/mu_w, lambda)-CMA-ES on the normalised [0, 1]^d cube.

    Follows Hansen (2016), "The CMA Evolution Strategy: A Tutorial": weighted
    intermediate recombination, cumulative step-size adaptation with the
    conjugate evolution path, and a covariance update combining the rank-one
    and rank-mu terms.  Box constraints are handled by evaluating the repaired
    (clipped) sample while updating the distribution with the raw sample and
    ranking by a quadratically penalised objective.
    """

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 pop_size: Optional[int] = None,
                 sigma0: float = CMAES_SIGMA0,
                 x0: Optional[np.ndarray] = None):
        self.ev = evaluator
        self.space = space
        self.base_params = base_params
        self.rng = rng
        n = space.dim
        self.n = n
        self.lam = int(pop_size) if pop_size else int(4 + math.floor(3 * math.log(n)))
        self.lam = max(4, self.lam)
        self.mu = max(1, self.lam // 2)

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = float(1.0 / np.sum(self.weights ** 2))

        self.c_c = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.c_s = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.c_mu = min(1.0 - self.c_1,
                        2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff))
        self.d_s = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.c_s
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        start = x0 if x0 is not None else space.to_unit(space.vector_of(base_params))
        self.mean = np.clip(np.asarray(start, dtype=float), 0.0, 1.0)
        self.sigma0 = float(sigma0)
        self.sigma = float(sigma0)
        self.p_c = np.zeros(n)
        self.p_s = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.inv_sqrt_C = np.eye(n)
        self.eigen_eval = 0
        self.generation = 0

    def config(self) -> Dict[str, Any]:
        return {"lambda": self.lam, "mu": self.mu, "mueff": self.mueff,
                "sigma0": self.sigma0, "box_penalty": CMAES_BOX_PENALTY}

    def _update_eigen(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:  # pragma: no cover - numerical fallback
            self.C = np.eye(self.n)
            eigvals, eigvecs = np.ones(self.n), np.eye(self.n)
        eigvals = np.maximum(eigvals, 1e-20)
        self.D = np.sqrt(eigvals)
        self.B = eigvecs
        self.inv_sqrt_C = self.B @ np.diag(1.0 / self.D) @ self.B.T

    def run(self) -> Dict[str, Any]:
        n = self.n
        while not self.ev.exhausted:
            self.generation += 1
            z = self.rng.standard_normal((self.lam, n))
            y = z @ (self.B * self.D).T
            x = self.mean + self.sigma * y
            x_rep = np.clip(x, 0.0, 1.0)

            params = [self.space.params_from_unit(self.base_params, xr) for xr in x_rep]
            recs = self.ev.evaluate(params, self.generation, stage=0)
            if not recs:
                break
            k = len(recs)
            raw = np.array([r.objective_value for r in recs], dtype=float)
            penalty = CMAES_BOX_PENALTY * np.sum((x[:k] - x_rep[:k]) ** 2, axis=1)
            ranked = np.where(np.isfinite(raw), raw - penalty, -np.inf)
            if not np.any(np.isfinite(ranked)):
                continue  # every candidate failed: resample without updating

            order = np.argsort(-ranked)
            mu = min(self.mu, k)
            weights = self.weights[:mu] / self.weights[:mu].sum()
            sel = order[:mu]
            y_w = weights @ y[sel]

            self.mean = np.clip(self.mean + self.sigma * y_w, 0.0, 1.0)
            mueff = float(1.0 / np.sum(weights ** 2))

            self.p_s = ((1.0 - self.c_s) * self.p_s
                        + math.sqrt(self.c_s * (2.0 - self.c_s) * mueff) * (self.inv_sqrt_C @ y_w))
            denom = math.sqrt(max(1e-12, 1.0 - (1.0 - self.c_s) ** (2 * self.generation)))
            h_sig = (np.linalg.norm(self.p_s) / denom / self.chi_n
                     < 1.4 + 2.0 / (n + 1.0))
            self.p_c = ((1.0 - self.c_c) * self.p_c
                        + (1.0 if h_sig else 0.0)
                        * math.sqrt(self.c_c * (2.0 - self.c_c) * mueff) * y_w)

            delta_h = (0.0 if h_sig else 1.0) * self.c_c * (2.0 - self.c_c)
            rank_mu = (y[sel] * weights[:, None]).T @ y[sel]
            self.C = ((1.0 - self.c_1 - self.c_mu) * self.C
                      + self.c_1 * (np.outer(self.p_c, self.p_c) + delta_h * self.C)
                      + self.c_mu * rank_mu)

            self.sigma *= math.exp(min(1.0, (self.c_s / self.d_s)
                                       * (np.linalg.norm(self.p_s) / self.chi_n - 1.0)))
            self.sigma = float(np.clip(self.sigma, 1e-8, 10.0))

            if (self.generation - self.eigen_eval
                    > 1.0 / max(1e-12, (self.c_1 + self.c_mu)) / n / 10.0):
                self.eigen_eval = self.generation
                self._update_eigen()
            if not np.all(np.isfinite(self.C)):  # pragma: no cover - numerical guard
                self.C = np.eye(n)
                self._update_eigen()
        return {"generations": self.generation, **self.config()}


# --------------------------------------------------------------------------- #
# 3. Bayesian optimisation (GP + EI)
# --------------------------------------------------------------------------- #
class MaternGP:
    """Zero-mean GP with an ARD Matern-5/2 kernel and Gaussian noise.

    Hyperparameters (log lengthscales, log signal variance, log noise variance)
    are fitted by maximising the exact log marginal likelihood with L-BFGS-B and
    analytic gradients.  Implemented with numpy/scipy only.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.theta = np.concatenate([np.log(np.full(dim, 0.3)), [np.log(1.0)], [np.log(1e-4)]])
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._chol = None

    # -- kernel ------------------------------------------------------------ #
    @staticmethod
    def _sq_scaled(X1: np.ndarray, X2: np.ndarray, ls: np.ndarray) -> np.ndarray:
        a = X1 / ls
        b = X2 / ls
        d2 = (np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :]
              - 2.0 * a @ b.T)
        return np.maximum(d2, 0.0)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, theta: np.ndarray) -> np.ndarray:
        ls = np.exp(theta[: self.dim])
        sf2 = math.exp(theta[self.dim])
        r = np.sqrt(self._sq_scaled(X1, X2, ls))
        s5r = math.sqrt(5.0) * r
        return sf2 * (1.0 + s5r + (5.0 / 3.0) * r * r) * np.exp(-s5r)

    def _nll_and_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        n = X.shape[0]
        ls = np.exp(theta[: self.dim])
        sf2 = math.exp(theta[self.dim])
        sn2 = math.exp(theta[self.dim + 1])
        r = np.sqrt(self._sq_scaled(X, X, ls))
        s5r = math.sqrt(5.0) * r
        e = np.exp(-s5r)
        K_sig = sf2 * (1.0 + s5r + (5.0 / 3.0) * r * r) * e
        K = K_sig + (sn2 + 1e-10) * np.eye(n)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12, np.zeros_like(theta)
        alpha = cho_solve(c, y)
        logdet = 2.0 * np.sum(np.log(np.diag(c[0])))
        nll = 0.5 * float(y @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)

        K_inv = cho_solve(c, np.eye(n))
        W = np.outer(alpha, alpha) - K_inv          # dL/dK = 0.5 * tr(W dK)
        grad = np.zeros_like(theta)
        common = (5.0 / 3.0) * sf2 * e * (1.0 + s5r)
        for i in range(self.dim):
            diff2 = (X[:, i][:, None] - X[:, i][None, :]) ** 2
            dK = common * diff2 / (ls[i] ** 2)
            grad[i] = -0.5 * float(np.sum(W * dK))
        grad[self.dim] = -0.5 * float(np.sum(W * K_sig))
        grad[self.dim + 1] = -0.5 * sn2 * float(np.trace(W))
        return nll, grad

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            rng: np.random.Generator,
            optimize: bool = True,
            n_restarts: int = 2) -> None:
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if optimize and self.X.shape[0] >= 4:
            bounds = ([(math.log(1e-2), math.log(1e2))] * self.dim
                      + [(math.log(1e-3), math.log(1e3)), (math.log(1e-8), math.log(1e0))])
            starts = [self.theta.copy()]
            for _ in range(max(0, n_restarts - 1)):
                starts.append(np.concatenate([
                    np.log(rng.uniform(0.05, 2.0, size=self.dim)),
                    [math.log(rng.uniform(0.2, 5.0))],
                    [math.log(rng.uniform(1e-6, 1e-2))],
                ]))
            best_theta, best_nll = self.theta, math.inf
            for start in starts:
                try:
                    res = minimize(self._nll_and_grad, start, args=(self.X, self.y),
                                   jac=True, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 80})
                except Exception:  # noqa: BLE001 - hyperparameters are non-critical
                    continue
                if res.fun < best_nll and np.all(np.isfinite(res.x)):
                    best_nll, best_theta = float(res.fun), res.x
            self.theta = best_theta
        self._factorize()

    def _factorize(self) -> None:
        assert self.X is not None and self.y is not None
        n = self.X.shape[0]
        sn2 = math.exp(self.theta[self.dim + 1])
        K = self._kernel(self.X, self.X, self.theta) + (sn2 + 1e-8) * np.eye(n)
        jitter = 1e-8
        for _ in range(6):
            try:
                self._chol = cho_factor(K, lower=True)
                break
            except np.linalg.LinAlgError:  # pragma: no cover - numerical guard
                K[np.diag_indices_from(K)] += jitter
                jitter *= 10.0
        self._alpha = cho_solve(self._chol, self.y)

    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at ``Xs``."""
        assert self.X is not None and self._chol is not None
        Ks = self._kernel(Xs, self.X, self.theta)
        mean = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        sf2 = math.exp(self.theta[self.dim])
        var = np.maximum(sf2 - np.sum(Ks * v.T, axis=1), 1e-12)
        return mean, np.sqrt(var)

    def append(self, x: np.ndarray, y_value: float) -> None:
        """Add a pseudo-observation (Kriging believer) and refactorise."""
        assert self.X is not None and self.y is not None
        self.X = np.vstack([self.X, np.atleast_2d(x)])
        self.y = np.concatenate([self.y, [y_value]])
        self._factorize()


class BayesianOptimizer:
    """GP-based Bayesian optimisation with Expected Improvement.

    Initial design: Latin hypercube on [0, 1]^d.  Each round fits the GP on the
    standardised objective values and maximises EI with multi-start L-BFGS-B;
    batches of ``batch_size`` points are built with the Kriging-believer
    heuristic so that EF5 runs stay parallel.
    """

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 n_init: Optional[int] = None,
                 batch_size: int = BO_DEFAULT_BATCH,
                 xi: float = BO_EI_XI):
        self.ev = evaluator
        self.space = space
        self.base_params = base_params
        self.rng = rng
        self.n_init = int(n_init) if n_init else BO_INIT_MULTIPLIER * space.dim
        self.batch_size = max(1, int(batch_size))
        self.xi = float(xi)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.gp = MaternGP(space.dim)

    def config(self) -> Dict[str, Any]:
        return {"n_init": self.n_init, "batch_size": self.batch_size, "ei_xi": self.xi,
                "kernel": "matern52_ard", "max_gp_points": BO_MAX_GP_POINTS,
                "hyper_refit_every": BO_HYPER_REFIT_EVERY}

    def _observe(self, points: Sequence[np.ndarray], iteration: int, stage: int = 0) -> None:
        params = [self.space.params_from_unit(self.base_params, u) for u in points]
        recs = self.ev.evaluate(params, iteration, stage=stage)
        for u, rec in zip(points, recs):
            self.X.append(np.asarray(u, dtype=float))
            self.y.append(rec.objective_value)

    def _training_set(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        X = np.array(self.X, dtype=float)
        y = np.array(self.y, dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            return X, np.zeros(len(y)), 0.0, 1.0
        worst = float(np.min(y[finite]))
        y = np.where(finite, y, worst)  # failed runs are imputed with the worst value
        if X.shape[0] > BO_MAX_GP_POINTS:
            order = np.argsort(-y)
            keep = list(order[: BO_MAX_GP_POINTS // 2])
            rest = order[BO_MAX_GP_POINTS // 2:]
            extra = self.rng.choice(rest, size=BO_MAX_GP_POINTS - len(keep), replace=False)
            idx = np.concatenate([np.array(keep, dtype=int), np.asarray(extra, dtype=int)])
            X, y = X[idx], y[idx]
        mu, sd = float(np.mean(y)), float(np.std(y))
        sd = sd if sd > 1e-12 else 1.0
        return X, (y - mu) / sd, mu, sd

    def _ei(self, Xs: np.ndarray, best: float) -> np.ndarray:
        mean, sd = self.gp.predict(Xs)
        imp = mean - best - self.xi
        z = imp / sd
        return imp * norm.cdf(z) + sd * norm.pdf(z)

    def _maximize_ei(self, best: float) -> np.ndarray:
        cand = self.rng.random((BO_EI_RANDOM_STARTS, self.space.dim))
        vals = np.concatenate([self._ei(chunk, best)
                               for chunk in np.array_split(cand, max(1, len(cand) // 250))])
        order = np.argsort(-vals)
        best_x, best_val = cand[order[0]], float(vals[order[0]])
        bounds = [(0.0, 1.0)] * self.space.dim
        for start in cand[order[:BO_EI_LOCAL_STARTS]]:
            try:
                res = minimize(lambda z: -float(self._ei(z[None, :], best)[0]),
                               start, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 40})
            except Exception:  # noqa: BLE001 - fall back to the random best
                continue
            if np.all(np.isfinite(res.x)) and -float(res.fun) > best_val:
                best_val, best_x = -float(res.fun), np.clip(res.x, 0.0, 1.0)
        return best_x

    def run(self) -> Dict[str, Any]:
        init = latin_hypercube(self.rng, self.n_init, self.space.dim)
        # start from the same reference point the other arms are seeded with
        init[0] = self.space.to_unit(self.space.vector_of(self.base_params))
        for start in range(0, len(init), max(1, self.batch_size)):
            if self.ev.exhausted:
                break
            self._observe(init[start:start + max(1, self.batch_size)], 0, stage=start // max(1, self.batch_size))

        iteration = 1
        since_refit = BO_HYPER_REFIT_EVERY
        while not self.ev.exhausted:
            X, y_std, _, _ = self._training_set()
            refit = since_refit >= BO_HYPER_REFIT_EVERY
            self.gp.fit(X, y_std, self.rng, optimize=refit)
            since_refit = 0 if refit else since_refit

            best_std = float(np.max(y_std))
            batch: List[np.ndarray] = []
            for _ in range(min(self.batch_size, self.ev.remaining)):
                x_next = self._maximize_ei(best_std)
                batch.append(x_next)
                if len(batch) < self.batch_size:  # Kriging believer
                    mean, _ = self.gp.predict(x_next[None, :])
                    self.gp.append(x_next, float(mean[0]))
            self._observe(batch, iteration)
            since_refit += len(batch)
            iteration += 1
        return {"iterations": iteration - 1, **self.config()}


# --------------------------------------------------------------------------- #
# 4. Ensemble Kalman Inversion
# --------------------------------------------------------------------------- #
EKI_RESIDUAL_SPEC: Tuple[Tuple[str, str, float, float], ...] = (
    # (name, source key, target, scale)
    ("volume_bias", "volume_bias_frac", 0.0, 0.10),
    ("high_flow_bias", "high_flow_bias_top2pct", 0.0, 0.15),
    ("mid_flow_bias", "mid_flow_bias_20_70pct", 0.0, 0.15),
    ("low_flow_bias", "low_flow_bias_bottom30pct", 0.0, 0.20),
    ("fdc_midslope_bias", "fdc_midslope_bias", 0.0, 0.20),
    ("bfi_bias", "bfi_bias", 0.0, 0.10),
    ("log_recession_ratio", "recession_k_ratio", 1.0, 0.25),   # log-transformed
)
EKI_EVENT_SPEC: Tuple[Tuple[str, str, float, float], ...] = (
    ("log_peak_ratio", "peak_ratio", 1.0, 0.20),               # log-transformed
    ("peak_lag_days", "peak_lag_hours_sim_minus_obs", 0.0, 0.25),
    ("event_volume_bias", "event_volume_bias_frac", 0.0, 0.15),
)
# global + event diagnostics + one (1 - NSE) term
EKI_RESIDUAL_DIM = len(EKI_RESIDUAL_SPEC) + len(EKI_EVENT_SPEC) + 1


def eki_multisite_residuals(record: EvaluationRecord, gauges: Sequence[str]) -> np.ndarray:
    """Observation operator for a multi-site objective: one ``1 - NSE`` per gauge.

    The single-site operator below is built from the outlet series alone -- its signature
    diagnostics and its NSE both come from ``csv_path`` -- so under a multi-site objective
    EKI would keep inverting for the outlet and ignore the records it is supposed to be
    fitting. That is not a subtle bias: the 100-parameter multi-site and outlet-only arms
    of HD4 returned bit-identical EKI solutions because of it. Assimilating one residual
    per gauge is also the textbook ensemble-Kalman way to use several observations at once.
    """
    per_gauge = {k[4:]: v for k, v in (record.aggregate_metrics or {}).items()
                 if k.startswith("NSE@")}
    out: List[float] = []
    for g in gauges:
        nse = per_gauge.get(g)
        try:
            val = float(nse)
        except (TypeError, ValueError):
            val = float("nan")
        out.append(float(np.clip(1.0 - val, 0.0, EKI_RESIDUAL_CLIP))
                   if math.isfinite(val) else EKI_RESIDUAL_CLIP)
    return np.array(out, dtype=float)


def eki_residuals(record: EvaluationRecord, mode: str = "residual") -> np.ndarray:
    """Standardised observation operator ``G(theta)`` with target ``y = 0``.

    The residual form uses the same diagnostics the rule proposer and the INCA
    numeric arm consume, each divided by a tolerance scale, plus ``1 - NSE`` so
    that the inversion also tracks the calibration objective.  Missing
    diagnostics contribute zero (no information).

    The scalar form is ``G = clip(1 - NSE, 0, EKI_RESIDUAL_CLIP)`` (a failed
    run gives the clip value): with target ``y = 0`` the Kalman update then
    drives NSE towards 1, i.e. it maximises the objective.  (The earlier
    ``G = -objective`` would have minimised ``NSE**2`` and pushed NSE to 0.)
    """
    if mode == "scalar":
        nse = _record_nse(record)
        val = (float(np.clip(1.0 - nse, 0.0, EKI_RESIDUAL_CLIP)) if math.isfinite(nse)
               else EKI_RESIDUAL_CLIP)
        return np.array([val], dtype=float)
    if mode != "residual":
        raise ValueError(f"unknown EKI observation mode {mode!r}; expected 'residual' or 'scalar'")
    if record.failed:
        # a crashed run carries no information: make it maximally inconsistent
        return np.full(EKI_RESIDUAL_DIM, EKI_RESIDUAL_CLIP, dtype=float)

    sig = record.signatures or {}
    glob = sig.get("global") or {}
    out: List[float] = []
    for name, key, target, scale in EKI_RESIDUAL_SPEC:
        raw = glob.get(key)
        val = float(raw) if raw is not None and math.isfinite(float(raw)) else None
        if val is None:
            out.append(0.0)
            continue
        if name.startswith("log_"):
            val = math.log(val) if val > 0 else float("nan")
            target = math.log(target) if target > 0 else 0.0
        out.append(0.0 if not math.isfinite(val) else (val - target) / scale)
    for name, key, target, scale in EKI_EVENT_SPEC:
        val = _mean_event_signature(sig, key)
        if not math.isfinite(val):
            out.append(0.0)
            continue
        if name == "peak_lag_days":
            val = val / 24.0
        elif name.startswith("log_"):
            val = math.log(val) if val > 0 else float("nan")
            target = math.log(target) if target > 0 else 0.0
        out.append(0.0 if not math.isfinite(val) else (val - target) / scale)

    nse = float(record.full_metrics.get("NSE", float("nan")))
    out.append(float(np.clip(1.0 - nse, 0.0, EKI_RESIDUAL_CLIP)) if math.isfinite(nse)
               else EKI_RESIDUAL_CLIP)
    return np.clip(np.array(out, dtype=float), -EKI_RESIDUAL_CLIP, EKI_RESIDUAL_CLIP)


class EkiOptimizer:
    """Ensemble Kalman Inversion (Iglesias, Law & Stuart 2013).

    The ensemble lives in the normalised cube.  With standardised residuals the
    observation covariance is the identity, so the update reads

        theta_j <- theta_j + C_thetaG (C_GG + I)^{-1} (0 - G_j - eta_j),
        eta_j ~ N(0, I),

    followed by a small multiplicative inflation to counter ensemble collapse
    and a projection back into the box.
    """

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 ensemble_size: int = EKI_DEFAULT_ENSEMBLE,
                 observation_mode: str = "residual",
                 multisite_gauges: Optional[Sequence[str]] = None,
                 inflation: float = EKI_INFLATION,
                 init_spread: float = 1.0):
        # Only when the objective really is the multi-site one: under an outlet-only
        # objective EKI must keep inverting for the outlet, even though the run records
        # every gauge for diagnostics.
        self.multisite_gauges: List[str] = [str(g) for g in (multisite_gauges or [])]
        self.ev = evaluator
        self.space = space
        self.base_params = base_params
        self.rng = rng
        self.J = max(4, int(ensemble_size))
        self.observation_mode = observation_mode
        self.inflation = float(inflation)
        # init_spread < 1 draws the initial ensemble in a box of that half-width (unit
        # cube) around the starting point instead of over the whole cube -- the block
        # coordinate baseline refines one zone from the incumbent and must not restart
        # globally at every visit. The ensemble is kept between run() calls so that a
        # zone revisited later continues from its own previous ensemble.
        self.init_spread = float(init_spread)
        self.theta: Optional[np.ndarray] = None

    def config(self) -> Dict[str, Any]:
        return {"ensemble_size": self.J, "observation_mode": self.observation_mode,
                "inflation": self.inflation, "init_spread": self.init_spread}

    def run(self) -> Dict[str, Any]:
        if self.theta is not None and self.theta.shape == (self.J, self.space.dim):
            theta = self.theta
        else:
            theta = latin_hypercube(self.rng, self.J, self.space.dim)
            center = self.space.to_unit(self.space.vector_of(self.base_params))
            if self.init_spread < 1.0:
                theta = np.clip(center + (theta - 0.5) * 2.0 * self.init_spread, 0.0, 1.0)
            theta[0] = center

        iteration = 0
        while not self.ev.exhausted:
            params = [self.space.params_from_unit(self.base_params, t) for t in theta]
            recs = self.ev.evaluate(params, iteration, stage=0)
            if len(recs) < 2:
                break
            J = len(recs)
            if self.observation_mode == "series":
                # full daily residual series (all gauges concatenated); a failed run gets a
                # vector of bounded norm so it neither dominates nor vanishes
                vecs = [r.residual_series for r in recs]
                m = max((len(v) for v in vecs if v is not None and len(v)), default=0)
                if m == 0:
                    break
                G = np.array([(np.clip(v, -EKI_RESIDUAL_CLIP, EKI_RESIDUAL_CLIP) if v is not None and len(v) == m
                               else np.full(m, EKI_RESIDUAL_CLIP / math.sqrt(m))) for v in vecs], dtype=float)
            elif self.multisite_gauges:
                G = np.array([eki_multisite_residuals(r, self.multisite_gauges) for r in recs],
                             dtype=float)
            else:
                G = np.array([eki_residuals(r, self.observation_mode) for r in recs], dtype=float)
            th = theta[:J]

            th_mean, G_mean = th.mean(axis=0), G.mean(axis=0)
            dth, dG = th - th_mean, G - G_mean
            m = G.shape[1]
            noise = self.rng.standard_normal((J, m))
            Y = -G - noise
            if m > 500:
                # ensemble-space form (Woodbury): C_tG (C_GG + I)^-1 with B = dG/sqrt(J-1)
                B = dG / math.sqrt(max(1, J - 1))                      # J x m
                A = dth.T / math.sqrt(max(1, J - 1))                   # n x J
                inner = np.linalg.solve(B @ B.T + np.eye(J), B @ Y.T)  # J x J_members
                update = (A @ (B @ Y.T - (B @ B.T) @ inner)).T          # J x n
            else:
                C_tG = dth.T @ dG / max(1, J - 1)
                C_GG = dG.T @ dG / max(1, J - 1)
                try:
                    gain = np.linalg.solve(C_GG + np.eye(m), Y.T).T
                except np.linalg.LinAlgError:  # pragma: no cover - numerical guard
                    gain = np.linalg.lstsq(C_GG + np.eye(m), Y.T, rcond=None)[0].T
                update = gain @ C_tG.T
            if not np.all(np.isfinite(update)):
                update = np.zeros_like(th)

            new_theta = th + update
            new_mean = new_theta.mean(axis=0)
            new_theta = new_mean + self.inflation * (new_theta - new_mean)
            theta = np.clip(new_theta, 0.0, 1.0)
            if theta.shape[0] < self.J:  # keep the ensemble size after truncation
                extra = latin_hypercube(self.rng, self.J - theta.shape[0], self.space.dim)
                theta = np.vstack([theta, extra])
            iteration += 1
        self.theta = theta
        return {"iterations": iteration, **self.config()}


class DreamZsOptimizer:
    """DREAM-ZS (Ter Braak & Vrugt 2008), sampling in the normalised cube.

    Why this exists rather than EF5's own ``STYLE=cali_dream``: that implementation
    fixes the sampled dimension to the model's parameter count
    (``DREAM.cpp:76  pointerMCMC->n = numParams`` -> 6 CREST + 7 KW = 13) and takes one
    chain per dimension (``seq = n``), reading its bounds from a single
    ``[CrestCaliParams]`` section that carries one ``GAUGE=``.  There is no path in it
    to the zoned ``10*(k+1)`` vector, so the high-dimensional question cannot be asked
    of it at all.

    Two things are kept faithful to EF5 so the numbers stay comparable with the
    published chain: the proposal box is the same ``PARAM_BOUNDS``, and the acceptance
    density is EF5's own.  That density is *not* the ``NSCE/2`` value EF5 stores in the
    output CSV (``DREAM.cpp:651``); the Metropolis test is ``metrop(..., option 3)``,
    which forms ``((1-NSE_new)/(1-NSE_old)) ** (-MaxT*(1+Gamma)/2)``
    (``dream_functions.cpp:616-631``) with ``Gamma = 0`` (``DREAM.cpp:83``) and
    ``MaxT = totalTimeStepsOutsideWarm`` (``Simulator.h:31``) -- i.e. the standard
    Gaussian/SSE likelihood ``log L = -(MaxT/2) log(1-NSE)``.  Everything else follows
    the ZS variant, which is what lets DREAM run above ~20 dimensions: a small number of
    chains plus an archive ``Z`` of past states supplying the difference vectors, so the
    number of chains no longer has to scale with the dimension.

    One generation costs ``chains`` forward runs, so the budget maps cleanly onto the
    other baselines' equal-budget accounting.  The archive is seeded with
    ``DREAM_INIT_PER_DIM * dim`` Latin-hypercube points, which at dim 100 is already
    300 of a 2000 evaluation budget -- that cost is part of the answer, not an artefact.
    """

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 chains: Optional[int] = None,
                 n_cr: int = DREAM_DEFAULT_NCR,
                 delta: int = DREAM_DEFAULT_DELTA,
                 snooker_prob: float = DREAM_SNOOKER_PROB,
                 init_per_dim: int = DREAM_INIT_PER_DIM,
                 likelihood: str = "ef5_sse",
                 sse_n: int = DREAM_SSE_N):
        self.ev = evaluator
        self.space = space
        self.base_params = base_params
        self.rng = rng
        self.chains = max(2, int(chains or DREAM_DEFAULT_CHAINS))
        self.n_cr = max(1, int(n_cr))
        self.delta = max(1, int(delta))
        self.snooker_prob = float(snooker_prob)
        self.init_per_dim = max(1, int(init_per_dim))
        if likelihood not in DREAM_LIKELIHOODS:
            raise ValueError(f"unknown dream likelihood {likelihood!r}; expected {DREAM_LIKELIHOODS}")
        self.likelihood = likelihood
        self.sse_n = max(2, int(sse_n))
        self.n_accepted = 0
        self.n_proposed = 0

    def config(self) -> Dict[str, Any]:
        return {
            "chains": self.chains,
            "n_cr": self.n_cr,
            "delta_pairs": self.delta,
            "snooker_prob": self.snooker_prob,
            "init_archive_per_dim": self.init_per_dim,
            "likelihood": self.likelihood,
            "likelihood_formula": (f"log_p = -(n/2)*log(1-NSE), n={self.sse_n} "
                                   "(EF5 dream_functions.cpp:616-631, Gamma=0)"
                                   if self.likelihood == "ef5_sse"
                                   else "log_p = NSE/2 (the stored nsce/2 diagnostic column, "
                                        "NOT EF5's acceptance density)"),
            "sse_n": self.sse_n,
            "variant": "DREAM-ZS (archive of past states); EF5 uses classic DREAM with seq=dim",
            "acceptance_rate": (self.n_accepted / self.n_proposed) if self.n_proposed else None,
        }

    # -- helpers ----------------------------------------------------------- #
    def _logp(self, rec: "EvaluationRecord") -> float:
        """Log-density of one evaluation; -inf when the forward run failed.

        ``ef5_sse`` is EF5's own acceptance density: metrop option 3 forms
        ``((1-NSE_new)/(1-NSE_old)) ** (-MaxT*(1+Gamma)/2)`` with Gamma = 0, i.e.
        ``log L = -(MaxT/2) * log(1-NSE)``.  Since ``1-NSE`` is the sum of squares
        divided by the (run-independent) observed variance, this is the standard
        Gaussian likelihood with the variance integrated out, up to a constant that
        cancels in the ratio.  It is sharp: a 0.05 NSE loss near NSE 0.8 is accepted
        with probability ~1e-548.

        ``nse_half`` is the flat ``NSE/2`` form -- the value EF5 *stores* as the
        ``nsce/2`` column (DREAM.cpp:651) but never uses for acceptance.  Kept as a
        contrast arm: under it the same 0.05 loss is accepted with probability 0.975,
        so the chain random-walks.
        """
        v = float(rec.objective_value)
        if not math.isfinite(v):
            return float("-inf")
        if self.likelihood == "nse_half":
            return 0.5 * v
        resid = max(1.0 - v, 1e-9)      # (1-NSE) is proportional to the sum of squares
        return -0.5 * self.sse_n * math.log(resid)

    def _evaluate_unit(self, U: np.ndarray, iteration: int) -> List["EvaluationRecord"]:
        params = [self.space.params_from_unit(self.base_params, u) for u in U]
        return self.ev.evaluate(params, iteration)

    def _cr_probs(self, delta_tot: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Adaptive crossover probabilities (Vrugt et al. 2009, eq. 6)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            w = delta_tot / np.maximum(counts, 1)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            return np.full(self.n_cr, 1.0 / self.n_cr)
        p = w / w.sum()
        p = np.clip(p, 0.05, None)          # never let a level die completely
        return p / p.sum()

    # -- main loop --------------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        d = self.space.dim
        iteration = 0

        # --- seed the archive Z ------------------------------------------- #
        m0 = min(max(self.init_per_dim * d, 2 * self.chains), max(1, self.ev.remaining))
        U0 = latin_hypercube(self.rng, m0, d)
        U0[0] = self.space.to_unit(self.space.vector_of(self.base_params))
        recs = self._evaluate_unit(U0, iteration)
        iteration += 1
        if not recs:
            return {"generations": 0, "archive_size": 0, **self.config()}
        Z = np.array([self.space.to_unit(np.array([r.params[n] for n in self.space.names], dtype=float))
                      for r in recs], dtype=float)
        Zlp = np.array([self._logp(r) for r in recs], dtype=float)

        # chains start at the best archive members
        order = np.argsort(-np.where(np.isfinite(Zlp), Zlp, -np.inf))
        idx = order[: self.chains]
        X = Z[idx].copy()
        Xlp = Zlp[idx].copy()

        cr_grid = (np.arange(1, self.n_cr + 1)) / float(self.n_cr)
        pcr = np.full(self.n_cr, 1.0 / self.n_cr)
        delta_tot = np.zeros(self.n_cr)
        cr_counts = np.zeros(self.n_cr)
        gen = 0

        while not self.ev.exhausted:
            gen += 1
            proposals = np.empty_like(X)
            cr_idx = np.empty(self.chains, dtype=int)
            log_jac = np.zeros(self.chains)   # snooker Jacobian, 0 for symmetric moves

            for c in range(self.chains):
                snooker = self.rng.random() < self.snooker_prob
                if snooker and Z.shape[0] >= 3:
                    # snooker update: project onto the line through an anchor point
                    a, b, zr = Z[self.rng.choice(Z.shape[0], size=3, replace=False)]
                    diff = X[c] - zr
                    nrm = float(diff @ diff)
                    if nrm <= 1e-12:
                        prop = X[c] + 1e-3 * self.rng.standard_normal(d)
                    else:
                        proj = ((a - b) @ diff) / nrm
                        gamma = 1.2 + self.rng.random() * (2.2 - 1.2)
                        prop = X[c] + gamma * proj * diff
                        # non-symmetric proposal: carry (|prop-zr| / |x-zr|)^(d-1)
                        num = float(np.linalg.norm(prop - zr))
                        den = float(np.linalg.norm(X[c] - zr))
                        if num > 1e-12 and den > 1e-12:
                            log_jac[c] = (d - 1) * (math.log(num) - math.log(den))
                    cr_idx[c] = self.n_cr - 1        # snooker moves every coordinate
                else:
                    # parallel-direction update from `delta` random pairs in Z
                    k = min(2 * self.delta, Z.shape[0])
                    pick = self.rng.choice(Z.shape[0], size=k, replace=False)
                    pairs = pick.reshape(-1, 2)
                    diff = sum(Z[i] - Z[j] for i, j in pairs)
                    j_cr = int(self.rng.choice(self.n_cr, p=pcr))
                    cr_idx[c] = j_cr
                    mask = self.rng.random(d) < cr_grid[j_cr]
                    if not mask.any():
                        mask[self.rng.integers(d)] = True
                    d_eff = max(1, int(mask.sum()))
                    gamma = 2.38 / math.sqrt(2.0 * len(pairs) * d_eff)
                    if gen % DREAM_GAMMA_JUMP_EVERY == 0:
                        gamma = 1.0
                    e = 1.0 + 1e-2 * (2.0 * self.rng.random(d) - 1.0)
                    eps = 1e-6 * self.rng.standard_normal(d)
                    step = e * gamma * diff + eps
                    prop = X[c].copy()
                    prop[mask] = prop[mask] + step[mask]
                # reflect at the box edges rather than clipping: clipping piles mass
                # on the bounds, which is exactly the pathology we are measuring
                prop = np.abs(prop)
                prop = np.where(prop > 1.0, 2.0 - prop, prop)
                proposals[c] = np.clip(prop, 0.0, 1.0)

            recs = self._evaluate_unit(proposals, iteration)
            iteration += 1
            if not recs:
                break

            for c, rec in enumerate(recs):
                lp = self._logp(rec)
                self.n_proposed += 1
                if not math.isfinite(lp):
                    alpha = 0.0                       # a failed run is never accepted
                elif not math.isfinite(Xlp[c]):
                    alpha = 1.0                       # anything beats a failed incumbent
                else:
                    alpha = math.exp(min(0.0, lp - Xlp[c] + log_jac[c]))
                if self.rng.random() < alpha:
                    jump = proposals[c] - X[c]
                    X[c] = proposals[c]
                    Xlp[c] = lp
                    self.n_accepted += 1
                    j = cr_idx[c]
                    var = np.var(Z, axis=0)
                    var = np.where(var > 1e-12, var, 1.0)
                    delta_tot[j] += float(np.sum((jump ** 2) / var))
                cr_counts[cr_idx[c]] += 1

            # append the accepted states to the archive every generation (ZS)
            Z = np.vstack([Z, X])
            Zlp = np.concatenate([Zlp, Xlp])
            if gen <= 20 or gen % 10 == 0:      # adapt CR during the early phase only
                pcr = self._cr_probs(delta_tot, cr_counts)

        return {"generations": gen, "archive_size": int(Z.shape[0]),
                "dim": d, **self.config()}


# --------------------------------------------------------------------------- #
# 5. Rule-based diagnostic proposer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Diagnostic:
    """A numeric signature compared against its perfect-model target."""

    key: str                  # name used in reports
    source: str               # "global" or "event"
    field: str                # key inside the signature dict
    target: float
    scale: float
    log_transform: bool = False
    description: str = ""


DIAGNOSTICS: Tuple[Diagnostic, ...] = (
    Diagnostic("volume_bias", "global", "volume_bias_frac", 0.0, 0.10,
               description="total simulated runoff volume vs observed"),
    Diagnostic("high_flow_bias", "global", "high_flow_bias_top2pct", 0.0, 0.15,
               description="volume of the top 2% of the FDC"),
    Diagnostic("mid_flow_bias", "global", "mid_flow_bias_20_70pct", 0.0, 0.15,
               description="volume of the 20-70% FDC band"),
    Diagnostic("low_flow_bias", "global", "low_flow_bias_bottom30pct", 0.0, 0.20,
               description="volume of the lowest 30% of the FDC"),
    Diagnostic("fdc_midslope_bias", "global", "fdc_midslope_bias", 0.0, 0.20,
               description="slope of the FDC mid-segment (quick/slow partitioning)"),
    Diagnostic("bfi_bias", "global", "bfi_bias", 0.0, 0.10,
               description="baseflow index difference (sim - obs)"),
    Diagnostic("recession_ratio", "global", "recession_k_ratio", 1.0, 0.25, log_transform=True,
               description="recession constant ratio; <1 means recessions too fast"),
    Diagnostic("event_peak_ratio", "event", "peak_ratio", 1.0, 0.20, log_transform=True,
               description="mean simulated/observed event peak magnitude"),
    Diagnostic("event_peak_lag", "event", "peak_lag_hours_sim_minus_obs", 0.0, 6.0,
               description="mean event peak timing error in hours (>0 = late)"),
    Diagnostic("event_volume_bias", "event", "event_volume_bias_frac", 0.0, 0.15,
               description="mean event runoff volume bias"),
)
DIAGNOSTIC_BY_KEY = {d.key: d for d in DIAGNOSTICS}


@dataclass(frozen=True)
class RuleEntry:
    """One deterministic (diagnostic -> parameter direction) rule.

    ``response`` is the direction the parameter must move when the diagnostic is
    *above* its target (i.e. the simulation is "too high" in that aspect).
    """

    diagnostic: str
    param: str
    response: int      # +1 = increase, -1 = decrease
    weight: float
    guide: str         # tag into GUIDE_SENTENCES
    note: str = ""


# --------------------------------------------------------------------------- #
# The table below is the entire hydrologic knowledge available to the ``rule``
# arm.  Each row cites the guide sentence it was translated from; see the module
# docstring for the sentence-by-sentence provenance.
# --------------------------------------------------------------------------- #
RULE_TABLE: Tuple[RuleEntry, ...] = (
    # --- D1: total runoff volume (G13: WM, B, IM, FC shape volume; KE for ET) ---
    RuleEntry("volume_bias", "wm", +1, 1.0, "G1", "higher WM -> more infiltration, less runoff"),
    RuleEntry("volume_bias", "b", -1, 0.9, "G2", "larger B -> more surface runoff"),
    RuleEntry("volume_bias", "im", -1, 0.8, "G3", "higher IM -> less infiltration, more runoff"),
    RuleEntry("volume_bias", "ke", +1, 0.8, "G4", "larger KE -> more evaporation, less runoff"),
    RuleEntry("volume_bias", "fc", +1, 0.7, "G5", "higher FC -> faster infiltration, less surface flow"),
    # --- D2: high-flow (top 2% FDC) magnitude ---
    RuleEntry("high_flow_bias", "b", -1, 1.0, "G2", "surface runoff generation"),
    RuleEntry("high_flow_bias", "im", -1, 0.9, "G3", "impervious fraction drives runoff"),
    RuleEntry("high_flow_bias", "fc", +1, 0.8, "G5", "infiltration removes surface flow"),
    RuleEntry("high_flow_bias", "alpha", +1, 0.7, "G11", "slower routing broadens (lowers) peaks"),
    RuleEntry("high_flow_bias", "beta", +1, 0.7, "G11", "slower routing broadens (lowers) peaks"),
    RuleEntry("high_flow_bias", "alpha0", +1, 0.4, "G12", "same relation off-channel"),
    # --- D3: event peak magnitude ---
    RuleEntry("event_peak_ratio", "b", -1, 1.0, "G2", "surface runoff generation"),
    RuleEntry("event_peak_ratio", "im", -1, 0.9, "G3", "impervious fraction drives runoff"),
    RuleEntry("event_peak_ratio", "fc", +1, 0.8, "G5", "infiltration removes surface flow"),
    RuleEntry("event_peak_ratio", "alpha", +1, 0.9, "G11", "slower routing broadens (lowers) peaks"),
    RuleEntry("event_peak_ratio", "beta", +1, 0.9, "G11", "slower routing broadens (lowers) peaks"),
    RuleEntry("event_peak_ratio", "alpha0", +1, 0.5, "G12", "same relation off-channel"),
    # --- D4: recession speed (value < target means recessions too fast) ---
    RuleEntry("recession_ratio", "under", +1, 1.0, "G8", "higher UNDER accelerates subsurface flow"),
    RuleEntry("recession_ratio", "leaki", +1, 1.0, "G9", "higher LEAKI speeds lateral drainage"),
    RuleEntry("recession_ratio", "alpha", -1, 0.5, "G11", "faster routing shortens the falling limb"),
    RuleEntry("recession_ratio", "beta", -1, 0.5, "G11", "faster routing shortens the falling limb"),
    # --- D5: baseflow index (too high = response not flashy enough) ---
    RuleEntry("bfi_bias", "under", +1, 0.9, "G8", "faster interflow makes the response flashier"),
    RuleEntry("bfi_bias", "leaki", +1, 0.9, "G9", "faster lateral drainage empties the slow store"),
    RuleEntry("bfi_bias", "b", +1, 0.6, "G2", "more surface runoff lowers the baseflow share"),
    RuleEntry("bfi_bias", "im", +1, 0.5, "G3", "more impervious area lowers the baseflow share"),
    RuleEntry("bfi_bias", "fc", -1, 0.5, "G5", "less infiltration lowers the baseflow share"),
    # --- D6: FDC mid-segment slope (too steep = too flashy) ---
    RuleEntry("fdc_midslope_bias", "wm", +1, 0.8, "G1", "more storage damps the response"),
    RuleEntry("fdc_midslope_bias", "b", -1, 0.8, "G2", "less surface runoff flattens the FDC"),
    RuleEntry("fdc_midslope_bias", "fc", +1, 0.6, "G5", "more infiltration flattens the FDC"),
    RuleEntry("fdc_midslope_bias", "under", -1, 0.6, "G8", "slower interflow damps variability"),
    RuleEntry("fdc_midslope_bias", "leaki", -1, 0.6, "G9", "slower drainage damps variability"),
    RuleEntry("fdc_midslope_bias", "alpha", +1, 0.4, "G11", "routing attenuation flattens the FDC"),
    RuleEntry("fdc_midslope_bias", "beta", +1, 0.4, "G11", "routing attenuation flattens the FDC"),
    # --- D7: event peak timing (>0 = simulated peaks late) ---
    RuleEntry("event_peak_lag", "alpha", -1, 1.0, "G11", "increasing ALPHA slows wave propagation"),
    RuleEntry("event_peak_lag", "beta", -1, 1.0, "G11", "increasing BETA slows wave propagation"),
    RuleEntry("event_peak_lag", "alpha0", -1, 0.6, "G12", "same relation off-channel"),
    RuleEntry("event_peak_lag", "under", +1, 0.6, "G8", "faster interflow advances the peak"),
    RuleEntry("event_peak_lag", "leaki", +1, 0.5, "G9", "faster drainage advances the peak"),
    # --- D8: event runoff volume ---
    RuleEntry("event_volume_bias", "wm", +1, 0.9, "G1", "higher WM -> more infiltration, less runoff"),
    RuleEntry("event_volume_bias", "b", -1, 0.9, "G2", "larger B -> more surface runoff"),
    RuleEntry("event_volume_bias", "im", -1, 0.7, "G3", "higher IM -> more runoff"),
    RuleEntry("event_volume_bias", "ke", +1, 0.5, "G4", "larger KE -> less runoff"),
    RuleEntry("event_volume_bias", "fc", +1, 0.7, "G5", "higher FC -> less surface flow"),
    # --- D9: mid-range flows ---
    RuleEntry("mid_flow_bias", "wm", +1, 0.7, "G1", "storage capacity controls mid-range volume"),
    RuleEntry("mid_flow_bias", "b", -1, 0.6, "G2", "infiltration-curve shape controls runoff"),
    RuleEntry("mid_flow_bias", "fc", +1, 0.5, "G5", "conductivity controls surface flow"),
    # --- D10: low flows ---
    RuleEntry("low_flow_bias", "under", +1, 0.8, "G8", "faster interflow leaves less water for low flows"),
    RuleEntry("low_flow_bias", "leaki", +1, 0.8, "G9", "faster lateral drainage empties the slow store"),
    RuleEntry("low_flow_bias", "wm", +1, 0.5, "G1", "higher WM reduces total runoff"),
    RuleEntry("low_flow_bias", "ke", +1, 0.5, "G4", "higher KE reduces total runoff"),
)

RULES_BY_DIAGNOSTIC: Dict[str, List[RuleEntry]] = {}
for _entry in RULE_TABLE:
    RULES_BY_DIAGNOSTIC.setdefault(_entry.diagnostic, []).append(_entry)


def diagnostic_deviations(signatures: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Standardised deviations ``(value - target)/scale`` for every diagnostic."""
    if not signatures:
        return {}
    glob = signatures.get("global") or {}
    out: Dict[str, float] = {}
    for diag in DIAGNOSTICS:
        if diag.source == "global":
            raw = glob.get(diag.field)
            val = float(raw) if raw is not None else float("nan")
        else:
            val = _mean_event_signature(signatures, diag.field)
        if not math.isfinite(val):
            continue
        target = diag.target
        if diag.log_transform:
            if val <= 0 or target <= 0:
                continue
            val, target = math.log(val), math.log(target)
        out[diag.key] = _soft_saturate((val - target) / diag.scale)
    return out


def _soft_saturate(dev: float, scale: float = RULE_DEVIATION_SOFT_SCALE) -> float:
    """``scale * tanh(dev / scale)``: sign, ordering and (for |dev| << scale) magnitude survive."""
    return float(scale * math.tanh(float(dev) / scale))


def parameter_pushes(deviations: Dict[str, float]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Aggregate the rule table into a signed push per parameter.

    Returns ``(pushes, fired)`` where ``pushes[param]`` is the summed
    ``weight * response * deviation`` and ``fired`` documents which rules were
    active (for the run log, so the paper can show what the arm "knew").
    """
    pushes: Dict[str, float] = {}
    fired: List[Dict[str, Any]] = []
    for diag_key, dev in deviations.items():
        if abs(dev) < RULE_MIN_DEVIATION:
            continue
        for rule in RULES_BY_DIAGNOSTIC.get(diag_key, []):
            if rule.param in FROZEN_PARAMETERS:
                continue
            contribution = rule.weight * rule.response * dev
            pushes[rule.param] = pushes.get(rule.param, 0.0) + contribution
            fired.append({
                "diagnostic": diag_key,
                "deviation": round(float(dev), 4),
                "param": rule.param,
                "direction": "increase" if contribution > 0 else "decrease",
                "contribution": round(float(contribution), 4),
                "guide": rule.guide,   # sentence text lives in GUIDE_SENTENCES
            })
    return pushes, fired


def _apply_step(name: str, value: float, step: float) -> float:
    """Multiplicative step with an additive fallback for (near-)zero values."""
    lo, hi = PARAM_BOUNDS[base_name(name)]
    span = max(hi - lo, 1e-12)
    if abs(value) < 1e-9 or abs(value) < 1e-6 * span:
        if step in RULE_ADDITIVE_FRACTION:
            frac = RULE_ADDITIVE_FRACTION[step]
        else:  # continuous (adaptive) multiplier: scale the fraction by log-step
            frac = 0.25 * math.log(max(step, 1e-9)) / math.log(2.0)
            frac = float(np.clip(frac, -0.40, 0.40))
        return safe_clip(name, value + frac * span)
    return safe_clip(name, value * step)


_MODERATE_STEP = {+1: 1.5, -1: 0.67}
_STRONG_STEP = {+1: 2.0, -1: 0.5}
_EXTREME_STEP = {+1: RULE_MAX_STEP, -1: 1.0 / RULE_MAX_STEP}


def adaptive_step(push: float) -> float:
    """Multiplier growing with |push|: ``4**(|push|/RULE_ADAPTIVE_PUSH_SCALE)`` capped at x4."""
    frac = min(1.0, abs(float(push)) / RULE_ADAPTIVE_PUSH_SCALE)
    mult = max(RULE_MIN_ADAPTIVE_STEP, RULE_MAX_STEP ** frac)
    return float(mult if push > 0 else 1.0 / mult)


def _step_for(push: float, tier: str) -> float:
    direction = 1 if push > 0 else -1
    if tier == "moderate":
        return _MODERATE_STEP[direction]
    if tier == "strong":
        return _STRONG_STEP[direction]
    if tier == "extreme":
        return _EXTREME_STEP[direction]
    if tier == "adaptive":
        return adaptive_step(push)
    raise ValueError(tier)


class RuleProposer:
    """Deterministic diagnostic-to-parameter proposer (the non-LLM twin of INCA).

    Each round it reads the numeric signatures of the current best simulation,
    aggregates :data:`RULE_TABLE` into a signed push per parameter, and emits
    ``n_candidates`` parameter sets.  The deterministic ladder covers single-
    and few-parameter moves at a *moderate* (x1.5 / x0.67), *strong* (x2 /
    x0.5) and *extreme* (x4 / x0.25) tier, one candidate that moves **every**
    pushed parameter with a step that grows with the push magnitude
    (:func:`adaptive_step`), one that moves every pushed parameter moderately,
    and the momentum of the last accepted move; randomised subsets (any size
    up to all pushed parameters) fill the remaining slots.  The outer loop
    mirrors INCA exactly: evaluate the batch, keep the best, stop after
    ``failure_patient`` consecutive non-improving rounds.  With
    ``fill_budget=True`` the remaining budget is then spent on random local
    restarts from the incumbent (the ``random_local`` move set), so that an
    equal-budget reading exists for the arm.
    """

    informed = True   # random_local flips this

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 n_candidates: int = 8,
                 failure_patient: int = 5,
                 fill_budget: bool = False):
        self.ev = evaluator
        self.space = space
        self.base_params = base_params
        self.rng = rng
        self.n_candidates = max(1, int(n_candidates))
        self.failure_patient = max(1, int(failure_patient))
        self.fill_budget = bool(fill_budget)
        self.log: List[Dict[str, Any]] = []
        self.stopped_early = False
        self.stop_reason = ""
        self.n_eval_at_stop: Optional[int] = None
        # parameter vectors already spent, so a non-improving round never
        # re-runs the very same deterministic ladder on the next round
        self.evaluated_keys: set = set()
        self._provenance = check_rule_provenance(verbose=True) if self.informed else None

    def config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "n_candidates": self.n_candidates,
            "failure_patient": self.failure_patient,
            "fill_budget": self.fill_budget,
            "informed": self.informed,
            "step_grid": list(RULE_STEP_GRID),
            "proposer_settings": dict(RULE_PROPOSER_SETTINGS),
            "legacy_proposer_settings": dict(LEGACY_RULE_PROPOSER),
            "random_local_max_subset": RULE_RANDOM_LOCAL_MAX_SUBSET,
        }
        if self.informed:
            cfg.update({
                "n_rules": len(RULE_TABLE),
                "n_diagnostics": len(DIAGNOSTICS),
                "deviation_soft_scale": RULE_DEVIATION_SOFT_SCALE,
                "min_deviation": RULE_MIN_DEVIATION,
                "guide_sentences": dict(GUIDE_SENTENCES),
                "guide_text": guide_text(),
                "guide_provenance": self._provenance,
                "rule_table_sha256": rule_table_sha256(),
                "rule_table": [
                    {"diagnostic": r.diagnostic, "param": r.param,
                     "response": r.response, "weight": r.weight,
                     "guide": r.guide, "note": r.note}
                    for r in RULE_TABLE
                ],
                "diagnostics": [
                    {"key": d.key, "source": d.source, "field": d.field,
                     "target": d.target, "scale": d.scale,
                     "log_transform": d.log_transform, "description": d.description}
                    for d in DIAGNOSTICS
                ],
            })
        return cfg

    # -- candidate construction ------------------------------------------- #
    def propose(self,
                current: ParameterSet,
                pushes: Dict[str, float],
                momentum: Optional[Dict[str, float]],
                aggressive: bool) -> List[Tuple[ParameterSet, Dict[str, Any]]]:
        """Build up to ``n_candidates`` distinct parameter sets from the pushes."""
        ranked = sorted(((p, v) for p, v in pushes.items() if abs(v) > 1e-9),
                        key=lambda kv: -abs(kv[1]))
        ranked = [(p, v) for p, v in ranked if p in self.space.names]

        plans: List[Tuple[str, List[Tuple[str, float]]]] = []
        if ranked:
            top = [p for p, _ in ranked]
            push = dict(ranked)

            def tiered(names: Sequence[str], tier: str) -> List[Tuple[str, float]]:
                return [(p, _step_for(push[p], tier)) for p in names]

            plans.append(("top1@moderate", tiered(top[:1], "moderate")))
            plans.append(("top1@strong", tiered(top[:1], "strong")))
            if len(top) >= 2:
                plans.append(("top2@strong", tiered(top[:2], "strong")))
            if len(top) >= 3:
                plans.append(("top3@moderate", tiered(top[:3], "moderate")))
            # every pushed parameter at once: the single-round reach INCA has
            plans.append(("all_pushed@adaptive", tiered(top, "adaptive")))
            if len(top) >= 2:
                plans.append(("all_pushed@moderate", tiered(top, "moderate")))
            plans.append(("top3@extreme", tiered(top[:3], "extreme")))
            if len(top) >= 2:
                plans.append(("top1@strong+top2@moderate",
                              [(top[0], _step_for(push[top[0]], "strong")),
                               (top[1], _step_for(push[top[1]], "moderate"))]))
        if momentum:
            plans.append(("momentum", [(p, s) for p, s in momentum.items() if p in self.space.names]))

        # randomised subsets keep the batch diverse when few rules fire and
        # supply fresh candidates after a round that reproduced known points
        pool = [p for p, _ in ranked] or list(self.space.names)
        tiers = ("moderate", "strong", "adaptive", "extreme")
        guard = 0
        while len(plans) < self.n_candidates * 4 and guard < 200:
            guard += 1
            size = int(self.rng.integers(1, len(pool) + 1))
            chosen = list(self.rng.choice(pool, size=size, replace=False))
            plan = []
            for p in chosen:
                if p in pushes and abs(pushes[p]) > 1e-9:
                    step = _step_for(pushes[p], str(self.rng.choice(tiers)))
                else:
                    step = float(self.rng.choice(RULE_STEP_GRID))
                plan.append((str(p), float(step)))
            plans.append(("random_subset", plan))
        return self._materialise(current, plans)

    def random_local_propose(self,
                             current: ParameterSet,
                             momentum: Optional[Dict[str, float]]
                             ) -> List[Tuple[ParameterSet, Dict[str, Any]]]:
        """Uninformed moves: 1..5 random free parameters, random grid multipliers."""
        plans: List[Tuple[str, List[Tuple[str, float]]]] = []
        if momentum:
            plans.append(("momentum", [(p, s) for p, s in momentum.items() if p in self.space.names]))
        names = list(self.space.names)
        max_subset = min(RULE_RANDOM_LOCAL_MAX_SUBSET, len(names))
        guard = 0
        while len(plans) < self.n_candidates * 4 and guard < 200:
            guard += 1
            size = int(self.rng.integers(1, max_subset + 1))
            chosen = list(self.rng.choice(names, size=size, replace=False))
            plans.append(("random_local",
                          [(str(p), float(self.rng.choice(RULE_STEP_GRID))) for p in chosen]))
        return self._materialise(current, plans)

    def _materialise(self,
                     current: ParameterSet,
                     plans: Sequence[Tuple[str, List[Tuple[str, float]]]]
                     ) -> List[Tuple[ParameterSet, Dict[str, Any]]]:
        candidates: List[Tuple[ParameterSet, Dict[str, Any]]] = []
        for label, plan in plans:
            if len(candidates) >= self.n_candidates:
                break
            values = current.values.copy()
            updates: Dict[str, float] = {}
            for name, step in plan:
                new_val = _apply_step(str(name), float(values[name]), float(step))
                if abs(new_val - values[name]) <= 1e-12 * max(1.0, abs(values[name])):
                    continue  # already at the bound: the move would waste a run
                values[name] = new_val
                updates[str(name)] = round(float(step), 6)
            if not updates:
                continue
            key = self._key(values)
            if key in self.evaluated_keys:
                continue  # never spend the budget twice on the same vector
            self.evaluated_keys.add(key)
            candidates.append((ParameterSet(values), {"steps": updates, "plan": label}))
        return candidates

    def _key(self, values: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(round(float(values[n]), 9) for n in self.space.names)

    # -- outer loop -------------------------------------------------------- #
    def _candidates_for_round(self,
                              best_rec: EvaluationRecord,
                              current: ParameterSet,
                              momentum: Optional[Dict[str, float]],
                              failures: int) -> Tuple[List[Tuple[ParameterSet, Dict[str, Any]]],
                                                      Dict[str, float], List[Dict[str, Any]]]:
        deviations = diagnostic_deviations(best_rec.signatures)
        pushes, fired = parameter_pushes(deviations)
        cands = self.propose(current, pushes, momentum, aggressive=failures >= 2)
        return cands, deviations, fired

    def run(self) -> Dict[str, Any]:
        baseline = self.ev.evaluate([self.base_params], 0, stage=0)
        if not baseline:
            return {"iterations": 0, "stopped_early": False, **self.config()}
        best_rec = baseline[0]
        current = ParameterSet(dict(best_rec.params))
        best_value = best_rec.objective_value
        self.evaluated_keys.add(self._key(current.values))
        momentum: Optional[Dict[str, float]] = None
        failures = 0
        iteration = 1
        phase = "informed" if self.informed else "random_local"

        while not self.ev.exhausted:
            if phase == "fill_budget":
                candidates = self.random_local_propose(current, momentum)
                deviations, fired = {}, []
            else:
                candidates, deviations, fired = self._candidates_for_round(
                    best_rec, current, momentum, failures)
            if not candidates:
                if phase != "fill_budget" and self.fill_budget:
                    self._enter_fill_phase("no new admissible candidate could be constructed "
                                           "(every proposal was already evaluated or hit a bound)")
                    phase = "fill_budget"
                    momentum = None
                    continue
                self.stopped_early = True
                self.stop_reason = ("no new admissible candidate could be constructed "
                                    "(every proposal was already evaluated or hit a bound)")
                break

            recs = self.ev.evaluate([c for c, _ in candidates], iteration, stage=0)
            if not recs:
                break
            assemble = getattr(self, "_assemble", None)
            if assemble is not None and phase != "fill_budget" and not self.ev.exhausted:
                merged = assemble(recs, best_rec)
                if merged is not None:
                    extra = self.ev.evaluate([merged], iteration, stage=1)
                    if extra:
                        recs = list(recs) + list(extra)
                        candidates = list(candidates) + [(merged, {"plan": "zone_assembly",
                                                                   "steps": {}})]
            round_best = max(recs, key=lambda r: r.objective_value)
            improved = _better(round_best.objective_value, best_value)

            self.log.append({
                "iteration": iteration,
                "phase": phase,
                "baseline_objective": best_value if math.isfinite(best_value) else None,
                "deviations": {k: round(v, 4) for k, v in deviations.items()},
                "fired_rules": fired,
                "candidates": [
                    {"candidate_index": rec.candidate_index,
                     "plan": meta.get("plan"),
                     "steps": meta["steps"],
                     "objective_value": rec.objective_value,
                     "params": rec.params}
                    for rec, meta in zip(recs, [m for _, m in candidates][:len(recs)])
                ],
                "best_candidate_index": round_best.candidate_index,
                "improved": bool(improved),
            })

            if improved:
                idx = recs.index(round_best)
                momentum = dict(candidates[idx][1]["steps"]) if idx < len(candidates) else None
                best_rec = round_best
                current = ParameterSet(dict(round_best.params))
                best_value = round_best.objective_value
                failures = 0
            else:
                failures += 1
                momentum = None
                if failures >= self.failure_patient and phase != "fill_budget":
                    reason = (f"no improvement for {failures} consecutive rounds "
                              f"(failure_patient={self.failure_patient})")
                    if self.fill_budget:
                        self._enter_fill_phase(reason)
                        phase = "fill_budget"
                        failures = 0
                    else:
                        self.stopped_early = True
                        self.stop_reason = reason
                        break
            iteration += 1

        if self.n_eval_at_stop is None and self.stopped_early:
            self.n_eval_at_stop = self.ev.n_evaluations
        return {"iterations": iteration - 1,
                "stopped_early": self.stopped_early,
                "stop_reason": self.stop_reason,
                "n_eval_at_stop": self.n_eval_at_stop,
                "budget_filled": self.fill_budget and self.n_eval_at_stop is not None,
                "rule_log": self.log,
                **self.config()}

    def _enter_fill_phase(self, reason: str) -> None:
        """Record the early stop and switch to random local restarts."""
        self.stopped_early = True
        self.stop_reason = reason + "; remaining budget spent on random local restarts"
        self.n_eval_at_stop = self.ev.n_evaluations


class ZoneRandomLocalProposer(RuleProposer):
    """``zone_random_local``: random_local given the same block structure INCA exploits.

    INCA's advantage on the multi-site problem could come from the language model
    reasoning about each sub-basin, or merely from the two mechanical things the agent
    loop does that no classical method here does: it proposes one zone at a time, and it
    assembles a candidate from whichever proposal fitted each zone's own gauge best. This
    proposer strips the language model away and keeps exactly those two, so the gap that
    remains is the part reasoning has to account for.

      * every candidate perturbs the free parameters of ONE zone (drawn uniformly),
        rather than 1..5 coordinates scattered across the whole vector;
      * after each batch, one extra candidate is assembled by taking each zone's block
        from whichever evaluation -- incumbent included -- scored best at that zone's own
        gauge, and it is simulated and judged on equal terms, exactly as in the agent.

    Everything else (batch size, step grid, de-duplication, early stopping, budget
    accounting) is inherited unchanged, so the only differences from ``random_local`` are
    the two listed above.
    """

    informed = False

    def __init__(self, *args, multisite_gauges=None, zone_ids=None, **kw):
        super().__init__(*args, **kw)
        self.multisite_gauges = [str(g) for g in (multisite_gauges or [])]
        self.zone_ids = [str(z) for z in (zone_ids or [])]

    def _candidates_for_round(self, best_rec, current, momentum, failures):
        zones = self.zone_ids or sorted({z for n in self.space.names if (z := zone_of(n))})
        if not zones:
            return self.random_local_propose(current, momentum), {}, []
        plans: List[Tuple[str, List[Tuple[str, float]]]] = []
        if momentum:
            plans.append(("momentum", [(p, s) for p, s in momentum.items() if p in self.space.names]))
        by_zone = {z: [n for n in self.space.names if zone_of(n) == z] for z in zones}
        guard = 0
        while len(plans) < self.n_candidates * 4 and guard < 400:
            guard += 1
            z = str(self.rng.choice(zones))
            names = by_zone.get(z) or []
            if not names:
                continue
            size = int(self.rng.integers(1, min(RULE_RANDOM_LOCAL_MAX_SUBSET, len(names)) + 1))
            chosen = list(self.rng.choice(names, size=size, replace=False))
            plans.append((f"zone_random_local:{z}",
                          [(str(p), float(self.rng.choice(RULE_STEP_GRID))) for p in chosen]))
        return self._materialise(current, plans), {}, []

    def _assemble(self, recs, incumbent_rec):
        """Per-zone recombination, the classical twin of the agent's zone assembly."""
        if not (self.multisite_gauges and self.zone_ids):
            return None
        sources = [("incumbent", incumbent_rec)] + [(str(r.candidate_index), r) for r in recs]
        picks, merged = {}, dict(incumbent_rec.params)
        for zone in self.zone_ids:
            key = f"NSE@{zone}"
            tag, val = None, float("-inf")
            for name, rec in sources:
                v = (rec.aggregate_metrics or {}).get(key)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v) and v > val:
                    tag, val = name, v
            if tag is None:
                return None
            picks[zone] = tag
        if len(set(picks.values())) <= 1:
            return None
        by_tag = dict(sources)
        for n in merged:
            z = zone_of(n)
            if z in picks:
                src = by_tag[picks[z]].params
                if n in src:
                    merged[n] = src[n]
        return ParameterSet(merged)


class ZoneRuleProposer(ZoneRandomLocalProposer):
    """``zone_rule``: the rule proposer applied zone by zone -- INCA's non-LLM twin on
    the multi-site problem.

    Each round it reads the numeric signatures of the incumbent's series at every gauge,
    turns each gauge's deviations into signed pushes through the same RULE_TABLE the
    single-zone proposer and the LLM guide share, and applies those pushes to that
    gauge's own parameter block. Zones are served worst-gauge-first; each zone gets its
    "move every pushed parameter" candidate first (the one-round reach INCA has), then the
    tiered single-parameter moves, until the batch is full. The per-zone assembly of the
    parent class recombines the batch afterwards. No language model, no free text: the
    diagnosis is the rule table's classification of the signature deviations, and the
    direction is the table's response.
    """

    informed = True

    def __init__(self, *args, signature_fn=None, multizone: bool = False, **kw):
        super().__init__(*args, **kw)
        # hook for tests: signature_fn(record, gauge_id) -> signatures dict or None
        self.signature_fn = signature_fn or self._signatures_from_disk
        self.round_diagnoses: List[Dict[str, Any]] = []
        # multizone: also propose candidates that apply every zone's pushes together (the
        # coordinated multi-zone move the LLM makes), at the adaptive and moderate tiers
        self.multizone = bool(multizone)

    def _signatures_from_disk(self, rec: EvaluationRecord, gauge: str) -> Optional[Dict[str, Any]]:
        out = Path(getattr(rec, "output_dir", "") or "")
        csv = out / f"ts.{gauge.lower()}.crest.csv"
        if not csv.exists():
            return None
        try:
            n_peaks = int(getattr(self.ev, "n_peaks", 3))
            kw = dict(getattr(self.ev, "peak_pick_kwargs", None) or DEFAULT_PEAK_PICK_KWARGS)
            windows = pick_peak_events(str(csv), n=n_peaks, **kw)
            _, sig = compute_signature_summary(str(csv), [{"start": str(a), "end": str(b)} for a, b in windows])
            return sig
        except Exception:  # noqa: BLE001 - a gauge without usable signatures just gets no rules
            return None

    def _candidates_for_round(self, best_rec, current, momentum, failures):
        zones = self.zone_ids or sorted({z for n in self.space.names if (z := zone_of(n))})
        if not zones:
            return self.random_local_propose(current, momentum), {}, []
        per_gauge = {k[4:]: v for k, v in (best_rec.aggregate_metrics or {}).items() if k.startswith("NSE@")}
        order = sorted(zones, key=lambda z: (per_gauge.get(z) if isinstance(per_gauge.get(z), (int, float))
                                             and math.isfinite(per_gauge.get(z)) else float("-inf")))
        zone_plans: Dict[str, List[Tuple[str, List[Tuple[str, float]]]]] = {}
        all_dev: Dict[str, float] = {}
        fired_all: List[Dict[str, Any]] = []
        diag_log: Dict[str, Any] = {}
        for z in order:
            sig = self.signature_fn(best_rec, z)
            dev = diagnostic_deviations(sig)
            pushes, fired = parameter_pushes(dev)
            names_z = [n for n in self.space.names if zone_of(n) == z]
            push_z = {f"{p}@{z}": v for p, v in pushes.items() if f"{p}@{z}" in names_z}
            all_dev.update({f"{k}@{z}": v for k, v in dev.items()})
            fired_all.extend([dict(f, zone=z) for f in fired])
            diag_log[z] = {"nse": per_gauge.get(z), "deviations": {k: round(v, 3) for k, v in dev.items()},
                           "pushes": {k: round(v, 3) for k, v in push_z.items()}}
            ranked = sorted(((p, v) for p, v in push_z.items() if abs(v) > 1e-9), key=lambda kv: -abs(kv[1]))
            plans: List[Tuple[str, List[Tuple[str, float]]]] = []
            if ranked:
                top = [p for p, _ in ranked]
                push = dict(ranked)
                tiered = lambda ns, tier: [(p, _step_for(push[p], tier)) for p in ns]  # noqa: E731
                plans.append((f"zone_rule:{z}:all_pushed@adaptive", tiered(top, "adaptive")))
                plans.append((f"zone_rule:{z}:top1@strong", tiered(top[:1], "strong")))
                if len(top) >= 2:
                    plans.append((f"zone_rule:{z}:top2@moderate", tiered(top[:2], "moderate")))
                plans.append((f"zone_rule:{z}:all_pushed@moderate", tiered(top, "moderate")))
                plans.append((f"zone_rule:{z}:top3@extreme", tiered(top[:3], "extreme")))
            else:
                # nothing fired at this gauge: fall back to a random move inside its block
                size = int(self.rng.integers(1, min(RULE_RANDOM_LOCAL_MAX_SUBSET, len(names_z)) + 1))
                chosen = list(self.rng.choice(names_z, size=size, replace=False))
                plans.append((f"zone_random_local:{z}", [(str(p), float(self.rng.choice(RULE_STEP_GRID))) for p in chosen]))
            zone_plans[z] = plans
        self.round_diagnoses.append(diag_log)
        # interleave: worst gauge's first plan, second-worst's first plan, ..., then second plans
        plans_out: List[Tuple[str, List[Tuple[str, float]]]] = []
        if momentum:
            plans_out.append(("momentum", [(p, s) for p, s in momentum.items() if p in self.space.names]))
        if self.multizone:
            for tier in ("adaptive", "moderate"):
                joint: List[Tuple[str, float]] = []
                for z in order:
                    for label, plan in zone_plans[z]:
                        if label.endswith(f"all_pushed@{tier}"):
                            joint.extend(plan)
                            break
                if joint:
                    plans_out.append((f"zone_rule:ALL:all_pushed@{tier}", joint))
        depth = max(len(v) for v in zone_plans.values()) if zone_plans else 0
        for d in range(depth):
            for z in order:
                if d < len(zone_plans[z]):
                    plans_out.append(zone_plans[z][d])
        # keep the batch diverse after a non-improving round: random in-block moves at the end
        guard = 0
        while len(plans_out) < self.n_candidates * 4 and guard < 200:
            guard += 1
            z = str(self.rng.choice(zones))
            names_z = [n for n in self.space.names if zone_of(n) == z]
            size = int(self.rng.integers(1, min(RULE_RANDOM_LOCAL_MAX_SUBSET, len(names_z)) + 1))
            chosen = list(self.rng.choice(names_z, size=size, replace=False))
            plans_out.append((f"zone_random_local:{z}", [(str(p), float(self.rng.choice(RULE_STEP_GRID))) for p in chosen]))
        return self._materialise(current, plans_out), all_dev, fired_all

    def config(self) -> Dict[str, Any]:
        cfg = super().config()
        cfg["zone_diagnoses"] = self.round_diagnoses
        return cfg


class RandomLocalProposer(RuleProposer):
    """``random_local``: the ``rule`` loop with the diagnostics switched off.

    Same batch size, step grid, de-duplication, momentum and early-stopping
    rule as :class:`RuleProposer`, but every round moves 1..5 randomly chosen
    free parameters by random multipliers from ``RULE_STEP_GRID`` (which
    contains both directions).  It is the uninformed null control for the
    question "does the diagnostic information matter?".
    """

    informed = False

    def _candidates_for_round(self, best_rec, current, momentum, failures):
        return self.random_local_propose(current, momentum), {}, []


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class _BlockView:
    """The evaluator a block optimizer sees: a slice of the zoned problem.

    The inner optimizer searches one zone's ten multipliers (or the ten lumped
    multipliers) and never sees the rest of the vector; every candidate it asks for
    is completed with the incumbent's values for the other zones, simulated by the
    real evaluator (so the run's records, budget and running-best trace are the
    usual ones, all in the multi-site objective), and handed back with
    ``objective_value`` replaced by the criterion this visit selects on -- the
    zone's own gauge for the sequential scheme, the multi-site mean otherwise.
    """

    def __init__(self, outer: BaseEvaluator, incumbent: Dict[str, float], zones: Sequence[str],
                 criterion: Callable[[EvaluationRecord], float], local_budget: int, iter_base: int):
        self.outer = outer
        self.inc = dict(incumbent)
        self.zones = list(zones)
        self.criterion = criterion
        self.budget = int(local_budget)
        self.iter_base = int(iter_base)
        self.used = 0
        self.records: List[EvaluationRecord] = []
        self.objective = outer.objective
        self.multisite_gauges = list(getattr(outer, "multisite_gauges", []) or [])
        self.need_signatures = False

    @property
    def remaining(self) -> int:
        return max(0, min(self.budget - self.used, self.outer.remaining))

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def n_evaluations(self) -> int:
        return self.used

    def full_params(self, sub: ParameterSet) -> ParameterSet:
        vals = dict(self.inc)
        for k, v in sub.values.items():
            if zone_of(k):
                vals[k] = float(v)
            else:  # a lumped name: broadcast to every zone
                for z in self.zones:
                    vals[f"{k}@{z}"] = float(v)
        return ParameterSet(vals)

    def evaluate(self, params_list: Sequence[ParameterSet], iteration: int, stage: int = 0) -> List[EvaluationRecord]:
        batch = list(params_list)[: self.remaining]
        if not batch:
            return []
        recs = self.outer.evaluate([self.full_params(p) for p in batch], self.iter_base + iteration, stage)
        self.used += len(recs)
        out: List[EvaluationRecord] = []
        for r in recs:
            c = self.criterion(r)
            c = float(c) if c is not None and math.isfinite(float(c)) else float("-inf")
            out.append(dataclasses.replace(r, objective_value=c))
        self.records.extend(out)
        return out


class BlockCoordinateOptimizer:
    """Structure-aware classical calibration of a zoned parameter set.

    This is the baseline a hydrologist would build by hand for a nested-gauge basin,
    and the fair answer to "INCA only wins because it is told the block structure":
    give a classical optimizer exactly the same structure. Three schemes:

    ``sequential``  one pass over the zones from the headwaters to the outlet; each
                    zone's ten multipliers are calibrated against that zone's own
                    gauge with the other zones held at the incumbent (the standard
                    upstream-to-downstream procedure). Budget split equally.
    ``cyclic``      block-coordinate ascent on the multi-site objective: visit the
                    zones in turn, spend ``per_visit`` evaluations on each, accept
                    the visit's best if it improves the incumbent, repeat until the
                    budget is gone. Each zone keeps its own inner-optimizer state
                    between visits.
    ``lumped``      a first stage calibrates the ten lumped multipliers (all zones
                    tied) for ``lumped_budget`` evaluations, then ``cyclic`` refines
                    zone by zone from that solution (calibrate lumped first, then
                    distribute).

    The inner optimizer is EKI (per-gauge residuals, ensemble initialised around the
    incumbent) or CMA-ES. Every evaluation is a full-basin run scored on the
    multi-site objective, so the budget axis is the same as for every other method.
    """

    def __init__(self,
                 evaluator: BaseEvaluator,
                 space: SearchSpace,
                 base_params: ParameterSet,
                 rng: np.random.Generator,
                 *,
                 inner: str = "eki",
                 scheme: str = "cyclic",
                 per_visit: Optional[int] = None,
                 lumped_budget: int = 0,
                 ensemble_size: Optional[int] = None,
                 sigma0: Optional[float] = None,
                 local_spread: float = 1.0):
        if not is_zoned(space.names):
            raise ValueError("block-coordinate baselines need a zoned parameter space (--zones/--param_zones)")
        if getattr(evaluator, "objective", "") != "multisite_nse":
            raise ValueError("block-coordinate baselines are defined for --objective multisite_nse")
        self.ev = evaluator
        self.space = space
        self.rng = rng
        self.inner = inner
        self.scheme = scheme
        self.gauges: List[str] = list(getattr(evaluator, "multisite_gauges", []) or [])
        # zone ids in the order the control lists them (outlet first, then interior
        # gauges by rank); zones without their own parameter block do not appear here
        seen: List[str] = []
        for n in space.names:
            z = zone_of(n)
            if z and z not in seen:
                seen.append(z)
        self.zones = seen
        self.bases = sorted({base_name(n) for n in space.names})
        # headwaters first, outlet last: a downstream gauge integrates everything
        # calibrated before it
        self.visit_order = list(reversed(self.zones))
        self.J = int(ensemble_size or EKI_DEFAULT_ENSEMBLE)
        self.sigma0 = float(sigma0) if sigma0 else CMAES_SIGMA0
        # each zone's first visit searches that zone's whole box (a normal EKI / CMA-ES
        # start on a 10-parameter problem); later visits continue from the zone's own
        # persisted ensemble / distribution, so locality comes from the inner method's
        # contraction rather than from a hand-set neighbourhood
        self.local_spread = float(local_spread)
        self.per_visit = int(per_visit) if per_visit else (self.J if inner == "eki" else 20)
        self.lumped_budget = int(lumped_budget)
        self.incumbent: Dict[str, float] = dict(base_params.values)
        self.inc_rec: Optional[EvaluationRecord] = None
        self._inner_state: Dict[str, Any] = {}
        self.visits: List[Dict[str, Any]] = []

    def config(self) -> Dict[str, Any]:
        return {"inner": self.inner, "scheme": self.scheme, "per_visit": self.per_visit,
                "lumped_budget": self.lumped_budget, "ensemble_size": self.J, "sigma0": self.sigma0,
                "local_spread": self.local_spread, "zones": self.zones, "visit_order": self.visit_order,
                "n_visits": len(self.visits), "visits": self.visits}

    # -- helpers ----------------------------------------------------------- #
    def _zone_names(self, zone: str) -> List[str]:
        return [n for n in self.space.names if zone_of(n) == zone]

    @staticmethod
    def _own_gauge(zone: str) -> Callable[[EvaluationRecord], float]:
        key = f"NSE@{zone}"
        return lambda r: (r.aggregate_metrics or {}).get(key, float("nan"))

    @staticmethod
    def _multisite(r: EvaluationRecord) -> float:
        return r.objective_value

    def _inner_for(self, key: str, view: _BlockView, sub_space: SearchSpace, sub_base: ParameterSet,
                   residual_gauges: Sequence[str], spread: float):
        opt = self._inner_state.get(key)
        if opt is None:
            if self.inner == "eki":
                opt = EkiOptimizer(view, sub_space, sub_base, self.rng, ensemble_size=self.J,
                                   multisite_gauges=list(residual_gauges), init_spread=spread)
            elif self.inner == "cmaes":
                opt = CmaEsOptimizer(view, sub_space, sub_base, self.rng, sigma0=self.sigma0)
            else:
                raise ValueError(f"unknown inner optimizer {self.inner!r}")
            self._inner_state[key] = opt
        else:
            opt.ev = view
            opt.base_params = sub_base
        return opt

    def _visit(self, key: str, names: Sequence[str], sub_base_values: Dict[str, float],
               criterion: Callable[[EvaluationRecord], float], residual_gauges: Sequence[str],
               budget: int, spread: float, label: str) -> None:
        if budget <= 0 or self.ev.exhausted:
            return
        view = _BlockView(self.ev, self.incumbent, self.zones, criterion, budget,
                          iter_base=1000 * (len(self.visits) + 1))
        sub_space = SearchSpace(names=list(names))
        sub_base = ParameterSet(dict(sub_base_values))
        opt = self._inner_for(key, view, sub_space, sub_base, residual_gauges, spread)
        opt.run()
        # accept the visit's best if it beats the incumbent on this visit's criterion
        cands = [r for r in view.records if math.isfinite(r.objective_value)]
        inc_val = float("-inf")
        if self.inc_rec is not None:
            v = criterion(self.inc_rec)
            inc_val = float(v) if v is not None and math.isfinite(float(v)) else float("-inf")
        accepted = False
        if cands:
            best = max(cands, key=lambda r: r.objective_value)
            if best.objective_value > inc_val:
                # the record in view.records carries the criterion; the full parameter
                # vector and the true multi-site value live in the outer record
                full = next(r for r in self.ev.records if r.eval_index == best.eval_index)
                self.incumbent = dict(full.params)
                self.inc_rec = full
                accepted = True
        self.visits.append({"label": label, "n_eval": view.used, "accepted": accepted,
                            "incumbent_objective": (self.inc_rec.objective_value
                                                    if self.inc_rec is not None else None)})

    # -- schemes ----------------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        # score the starting point once so that the incumbent has a record to compare with
        recs = self.ev.evaluate([ParameterSet(dict(self.incumbent))], 0, stage=0)
        if recs and not recs[0].failed:
            self.inc_rec = recs[0]

        if self.scheme == "lumped" and self.lumped_budget > 0:
            outlet = self.zones[0]
            lumped_base = {b: self.incumbent[f"{b}@{outlet}"] for b in self.bases}
            self._visit("lumped", self.bases, lumped_base, self._multisite, self.gauges,
                        min(self.lumped_budget, self.ev.remaining), spread=1.0, label="lumped")

        if self.scheme == "sequential":
            n = len(self.visit_order)
            share = max(self.J, self.ev.remaining // max(1, n))
            for z in self.visit_order:
                names = self._zone_names(z)
                self._visit(f"zone:{z}", names, {k: self.incumbent[k] for k in names},
                            self._own_gauge(z), [z], min(share, self.ev.remaining),
                            spread=self.local_spread, label=f"seq:{z}")
            # rounding may leave a few evaluations: spend them cyclically on the objective
        cycle = 0
        while not self.ev.exhausted:
            cycle += 1
            for z in self.visit_order:
                if self.ev.exhausted:
                    break
                names = self._zone_names(z)
                self._visit(f"zone:{z}", names, {k: self.incumbent[k] for k in names},
                            self._multisite, self.gauges, min(self.per_visit, self.ev.remaining),
                            spread=self.local_spread, label=f"cyc{cycle}:{z}")
            if cycle > 10_000:  # pragma: no cover - safety
                break
        return {"cycles": cycle, **self.config()}


def build_optimizer(method: str,
                    evaluator: BaseEvaluator,
                    space: SearchSpace,
                    base_params: ParameterSet,
                    rng: np.random.Generator,
                    *,
                    n_candidates: int = 8,
                    failure_patient: int = 5,
                    pop_size: Optional[int] = None,
                    max_workers: int = 8,
                    fill_budget: bool = False,
                    eki_observation: str = "residual",
                    ensemble_size: Optional[int] = None,
                    sigma0: Optional[float] = None,
                    complexes: Optional[int] = None,
                    complex_size: Optional[int] = None,
                    subset_size: Optional[int] = None,
                    evolutions_per_complex: Optional[int] = None,
                    dream_likelihood: str = "ef5_sse",
                    lumped_budget: Optional[int] = None):
    """Instantiate one of the baseline optimizers."""
    if method == "rule" and is_zoned(space.names):
        # RULE_TABLE pushes are keyed by base parameter name; on a zoned space they
        # would silently act on an arbitrary subset of zones. Not a meaningful baseline.
        raise ValueError("the rule proposer is single-zone only; use sceua/cmaes/bo/eki/random_local with --zones")
    if method == "sceua":
        if complex_size is None and pop_size:
            p = complexes or SCEUA_DEFAULT_COMPLEXES
            complex_size = max(3, int(round(pop_size / p)))
        return SceUaOptimizer(evaluator, space, base_params, rng,
                              complexes=complexes, complex_size=complex_size,
                              subset_size=subset_size,
                              evolutions_per_complex=evolutions_per_complex)
    if method == "cmaes":
        return CmaEsOptimizer(evaluator, space, base_params, rng, pop_size=pop_size,
                              sigma0=float(sigma0) if sigma0 else CMAES_SIGMA0)
    if method == "bo":
        batch = int(pop_size) if pop_size else max(1, min(BO_DEFAULT_BATCH, max_workers))
        return BayesianOptimizer(evaluator, space, base_params, rng, batch_size=batch)
    if method == "eki":
        size = int(ensemble_size or pop_size or EKI_DEFAULT_ENSEMBLE)
        return EkiOptimizer(evaluator, space, base_params, rng,
                            ensemble_size=size, observation_mode=eki_observation,
                            multisite_gauges=(getattr(evaluator, "multisite_gauges", None)
                                              if getattr(evaluator, "objective", "") == "multisite_nse"
                                              else None))
    if method == "dream":
        # --pop_size / --ensemble_size double as the chain count so the existing
        # sensitivity flags work; EF5's own choice would be chains = dim.
        return DreamZsOptimizer(evaluator, space, base_params, rng,
                                chains=(ensemble_size or pop_size or None),
                                likelihood=dream_likelihood)
    if method == "rule":
        return RuleProposer(evaluator, space, base_params, rng,
                            n_candidates=n_candidates, failure_patient=failure_patient,
                            fill_budget=fill_budget)
    if method == "zone_random_local":
        return ZoneRandomLocalProposer(
            evaluator, space, base_params, rng,
            n_candidates=n_candidates, failure_patient=failure_patient,
            fill_budget=fill_budget,
            multisite_gauges=getattr(evaluator, "multisite_gauges", None),
            zone_ids=sorted({z for n in space.names if (z := zone_of(n))}))
    if method == "random_local":
        return RandomLocalProposer(evaluator, space, base_params, rng,
                                   n_candidates=n_candidates, failure_patient=failure_patient,
                                   fill_budget=fill_budget)
    if method in ("zone_rule", "zone_rule_mz"):
        return ZoneRuleProposer(
            evaluator, space, base_params, rng,
            n_candidates=n_candidates, failure_patient=failure_patient, fill_budget=fill_budget,
            multisite_gauges=getattr(evaluator, "multisite_gauges", None),
            zone_ids=sorted({z for n in space.names if (z := zone_of(n))}),
            multizone=(method == "zone_rule_mz"))
    if method.startswith("blk_"):
        scheme = {"blk_seq_eki": "sequential", "blk_cyc_eki": "cyclic",
                  "blk_lump_eki": "lumped", "blk_cyc_cmaes": "cyclic"}[method]
        return BlockCoordinateOptimizer(
            evaluator, space, base_params, rng,
            inner="cmaes" if method.endswith("cmaes") else "eki", scheme=scheme,
            per_visit=pop_size,
            # 200 evaluations of lumped calibration before distributing (a tenth of the
            # usual 2000 budget); scaled down for small test budgets
            lumped_budget=((int(lumped_budget) if lumped_budget else
                            min(200, max(int(ensemble_size or EKI_DEFAULT_ENSEMBLE), evaluator.budget // 5)))
                           if scheme == "lumped" else 0),
            ensemble_size=ensemble_size, sigma0=sigma0)
    raise ValueError(f"Unknown method '{method}'; expected one of {METHODS}")


def method_rng(seed: int, method: str) -> np.random.Generator:
    """Independent random sub-stream per optimizer: ``default_rng([seed, METHOD_IDS[method]])``."""
    return np.random.default_rng([int(seed), METHOD_IDS[method]])


def _record_summary(rec: Optional[EvaluationRecord], with_outputs: bool = True) -> Optional[Dict[str, Any]]:
    if rec is None:
        return None
    out: Dict[str, Any] = {
        "eval_index": rec.eval_index,
        "iteration": rec.iteration,
        "candidate_index": rec.candidate_index,
        "objective_value": rec.objective_value,
        "params": rec.params,
        "full_metrics": rec.full_metrics,
        "aggregate_metrics": rec.aggregate_metrics,
    }
    if with_outputs:
        out["output_dir"] = rec.output_dir
        out["csv_path"] = rec.csv_path
    return out


def run_optimization(evaluator: BaseEvaluator,
                     method: str,
                     base_params: ParameterSet,
                     *,
                     seed: int = 0,
                     gauge: str = DEFAULT_GAUGE_NUM,
                     n_candidates: int = 8,
                     failure_patient: int = 5,
                     pop_size: Optional[int] = None,
                     max_workers: int = 8,
                     space: Optional[SearchSpace] = None,
                     extra_config: Optional[Dict[str, Any]] = None,
                     fill_budget: bool = False,
                     eki_observation: str = "residual",
                     ensemble_size: Optional[int] = None,
                     sigma0: Optional[float] = None,
                     complexes: Optional[int] = None,
                     complex_size: Optional[int] = None,
                     subset_size: Optional[int] = None,
                     evolutions_per_complex: Optional[int] = None,
                     dream_likelihood: str = "ef5_sse",
                     lumped_budget: Optional[int] = None) -> Dict[str, Any]:
    """Run one optimizer to exhaustion of the budget and return the history payload."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'; expected one of {METHODS}")
    space = space or SearchSpace()
    if method in ("rule", "eki") and hasattr(evaluator, "need_signatures"):
        evaluator.need_signatures = True  # both consume numeric signatures
    if method == "eki" and eki_observation == "series" and hasattr(evaluator, "residual_series"):
        evaluator.residual_series = True
    rng = method_rng(seed, method)
    provenance = provenance_block(verbose=False)
    optimizer = build_optimizer(method, evaluator, space, base_params, rng,
                                n_candidates=n_candidates,
                                failure_patient=failure_patient,
                                pop_size=pop_size,
                                max_workers=max_workers,
                                fill_budget=fill_budget,
                                eki_observation=eki_observation,
                                ensemble_size=ensemble_size,
                                sigma0=sigma0,
                                complexes=complexes,
                                complex_size=complex_size,
                                subset_size=subset_size,
                                evolutions_per_complex=evolutions_per_complex,
                                dream_likelihood=dream_likelihood,
                                lumped_budget=lumped_budget)
    started = time.time()
    info = optimizer.run()
    elapsed = time.time() - started

    stopped_early = bool(info.pop("stopped_early", False))
    stop_reason = info.pop("stop_reason", "")
    n_eval_at_stop = info.pop("n_eval_at_stop", None)
    if not stopped_early and evaluator.n_evaluations < evaluator.budget:
        stopped_early = True
        stop_reason = stop_reason or "optimizer terminated before exhausting the budget"
    if stopped_early and n_eval_at_stop is None:
        n_eval_at_stop = evaluator.n_evaluations

    payload: Dict[str, Any] = {
        "method": method,
        "objective": evaluator.objective,
        "seed": seed,
        "rng_stream": {"seed": int(seed), "method_id": METHOD_IDS[method],
                       "scheme": "numpy.random.default_rng([seed, method_id])"},
        "budget": evaluator.budget,
        "gauge": gauge,
        "n_evaluations": evaluator.n_evaluations,
        "n_failed": sum(1 for r in evaluator.records if r.failed),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        # evaluations consumed when the optimizer's own stopping rule fired
        # (None = ran until the budget was exhausted)
        "n_eval_at_stop": n_eval_at_stop,
        "fill_budget": bool(fill_budget),
        "wall_time_s": elapsed,
        "param_names": list(space.names),
        "param_bounds": {n: list(PARAM_BOUNDS[base_name(n)]) for n in space.names},
        "frozen_parameters": sorted(FROZEN_PARAMETERS),
        "zoned": is_zoned(space.names),
        "dim": space.dim,
        "initial_params": dict(base_params.values),
        "composite_weights": {
            "lag_hours": COMPOSITE_LAG_WEIGHT,
            "log_peak_ratio": COMPOSITE_PEAK_WEIGHT,
            "event_volume_bias": COMPOSITE_VOLUME_WEIGHT,
        },
        "composite_tolerances": composite_tolerances(),
        "composite_penalty_cap": COMPOSITE_PENALTY_CAP,
        "composite_formula": ("NSE - sum_k w_k * clip(|dev_k| / tol_k, 0, cap); "
                              "peak lag averaged over events after |.|"),
        "provenance": provenance,
        "optimizer_config": info,
        "evaluations": [r.to_json() for r in evaluator.records],
        "running_best": list(evaluator.running_best),
        "running_best_units": evaluator.objective,
        "running_best_nse": list(evaluator.running_best_nse),
    }
    if extra_config:
        payload.update(extra_config)
    best = evaluator.best
    payload["best"] = _record_summary(best)
    if best is not None and best.signatures is not None:
        payload["best"]["signatures"] = best.signatures
    payload["best_by_nse"] = _record_summary(evaluator.best_by_nse)
    return _jsonable(payload)


def build_checkpoint(evaluator: BaseEvaluator,
                     method: str,
                     seed: int,
                     gauge: str) -> Dict[str, Any]:
    """Minimal in-progress history, for crash-safe checkpointing of long runs."""
    return _jsonable({
        "checkpoint": True,
        "method": method,
        "objective": evaluator.objective,
        "seed": seed,
        "rng_stream": {"seed": int(seed), "method_id": METHOD_IDS.get(method),
                       "scheme": "numpy.random.default_rng([seed, method_id])"},
        "budget": evaluator.budget,
        "gauge": gauge,
        "n_evaluations": evaluator.n_evaluations,
        "n_failed": sum(1 for r in evaluator.records if r.failed),
        "evaluations": [r.to_json() for r in evaluator.records],
        "running_best": list(evaluator.running_best),
        "running_best_units": evaluator.objective,
        "running_best_nse": list(evaluator.running_best_nse),
        "best": _record_summary(evaluator.best, with_outputs=False),
        "best_by_nse": _record_summary(evaluator.best_by_nse, with_outputs=False),
    })


def save_history(payload: Dict[str, Any], out_path: str | Path) -> Path:
    """Write the run history to ``out_path`` (parents created as needed)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2))
    return path


__all__ = [
    "METHODS",
    "METHOD_IDS",
    "OBJECTIVES",
    "COMPOSITE_LAG_WEIGHT",
    "COMPOSITE_PEAK_WEIGHT",
    "COMPOSITE_VOLUME_WEIGHT",
    "COMPOSITE_PENALTY_CAP",
    "RULE_STEP_GRID",
    "RULE_DEVIATION_SOFT_SCALE",
    "RULE_PROPOSER_SETTINGS",
    "LEGACY_RULE_PROPOSER",
    "GUIDE_SENTENCES",
    "DIAGNOSTICS",
    "RULE_TABLE",
    "SearchSpace",
    "EvaluationRecord",
    "BaseEvaluator",
    "EF5Evaluator",
    "SceUaOptimizer",
    "CmaEsOptimizer",
    "MaternGP",
    "BayesianOptimizer",
    "EkiOptimizer",
    "RuleProposer",
    "RandomLocalProposer",
    "composite_objective",
    "composite_penalties",
    "composite_tolerances",
    "diagnostic_deviations",
    "parameter_pushes",
    "adaptive_step",
    "eki_residuals",
    "latin_hypercube",
    "sceua_defaults",
    "method_rng",
    "build_optimizer",
    "run_optimization",
    "build_checkpoint",
    "save_history",
    "guide_text",
    "check_rule_provenance",
    "rule_table_sha256",
    "provenance_block",
]
