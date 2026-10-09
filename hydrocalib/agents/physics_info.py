"""Helpers for parameter naming and physics guidance in prompts."""

from __future__ import annotations

import re

from typing import Any, Dict, Optional, Sequence, Tuple

from ..config import FROZEN_PARAMETERS, PARAM_BOUNDS
from ..zoned import SEP, base_name, zone_of


PHYSICS_PARAMETER_GUIDE = (
    "EF5 Parameter Overview for Calibration\n"
    "WM controls the total soil water storage capacity; higher WM increases infiltration and reduces runoff. "
    "B defines the shape of the variable infiltration curve; larger B yields more surface runoff for a given soil moisture. "
    "IM is the impervious area fraction—higher values reduce infiltration and increase runoff. "
    "KE scales potential evapotranspiration (PET); larger KE increases evaporation and decreases runoff. "
    "FC is the saturated hydraulic conductivity; higher FC allows faster infiltration, reducing surface flow. "
    "IWU sets the initial soil moisture; too high a value can exaggerate early runoff. "
    "TH determines the drainage threshold for channel initiation; a larger TH produces fewer, coarser channels. "
    "UNDER controls interflow velocity—higher values accelerate subsurface flow. "
    "LEAKI defines the leakage rate from the interflow layer; higher LEAKI speeds lateral drainage. "
    "ISU is the initial interflow storage; nonzero values may create unrealistic early peaks. "
    "ALPHA and BETA are routing parameters in the discharge equation Q = αA^β; increasing either slows wave propagation and broadens flood peaks. "
    "ALPHA0 applies the same relationship for non-channel cells. "
    "Together, these parameters govern infiltration, storage, and routing. During calibration, adjust WM, B, IM, and FC to shape runoff volume; "
    "tune KE for evapotranspiration balance; and modify ALPHA, BETA, UNDER, and LEAKI to match hydrograph timing and attenuation."
)

# Definitions-only variant of the guide for the "names and meanings, no relations" ablation:
# every parameter is named and defined, but no sentence states how a change moves the
# hydrograph or which parameters to adjust for which deficiency.
PHYSICS_PARAMETER_DEFINITIONS = (
    "EF5 Parameter Overview for Calibration\n"
    "WM is the maximum soil water storage capacity of the soil column. "
    "B is the exponent of the variable infiltration capacity curve. "
    "IM is the impervious area fraction of a grid cell. "
    "KE is the multiplier applied to potential evapotranspiration (PET). "
    "FC is the saturated hydraulic conductivity of the soil. "
    "IWU is the initial soil water content. "
    "TH is the drainage-area threshold at which a cell is treated as a channel. "
    "UNDER is the interflow velocity multiplier. "
    "LEAKI is the leakage rate of the interflow reservoir. "
    "ISU is the initial interflow storage. "
    "ALPHA and BETA are the coefficients of the kinematic-wave relation A = \u03b1Q^\u03b2 between channel cross-section area A "
    "and discharge Q used for channel routing. "
    "ALPHA0 is the corresponding coefficient for non-channel (overland) cells."
)
# NOTE: EF5's documentation (and the guide used in the submitted study) write this relation as
# Q = alpha*A^beta; the implementation (EF5/src/KinematicRoute.cpp, rhs = A + alpha*Q^beta ...) is
# A = alpha*Q^beta, which is why a larger ALPHA slows and attenuates the wave, as the guide's
# directional statement says and the perturbation check confirms. The first definitions-only
# ablation (arm ND) used the documented, inverted form; arm ND2 uses the implemented form.

_GUIDE_MODE = {"mode": "full"}   # "full" (relations stated) or "definitions" (names and meanings only)


def set_guide_mode(mode: str) -> None:
    if mode not in ("full", "definitions"):
        raise ValueError(f"unknown guide mode {mode!r}")
    _GUIDE_MODE["mode"] = mode


def guide_mode() -> str:
    return _GUIDE_MODE["mode"]


def _parameter_aliases() -> Dict[str, str]:
    numbered = [f"x{idx + 1}" for idx, _ in enumerate(PARAM_BOUNDS)]
    return {name: alias for name, alias in zip(PARAM_BOUNDS.keys(), numbered)}


def build_display_name_map(use_physics_information: bool,
                           names: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Return mapping from real parameter names to prompt-facing labels.

    ``names`` overrides the default 13 base parameters; pass the zoned names
    (``wm@V1`` ...) for a multi-gauge run. Anonymised labels keep the zone suffix
    so the ablation still hides physics, not the partition.
    """

    aliases = _parameter_aliases()
    keys = list(names) if names is not None else list(PARAM_BOUNDS)
    if use_physics_information:
        return {name: name for name in keys}
    out: Dict[str, str] = {}
    for name in keys:
        zone = zone_of(name)
        alias = aliases[base_name(name)]
        out[name] = alias if zone is None else f"{alias}{SEP}{zone}"
    return out


def build_zone_note(zone_ids: Sequence[str],
                    areas: Optional[Dict[str, float]] = None,
                    allow_wildcard: bool = True,
                    observed: bool = False) -> str:
    """One paragraph describing the partition, appended to the parameter guide.

    Kept deliberately short: it is repeated in every proposal and evaluation call,
    and the per-zone numbers are the only part that carries information.

    ``observed`` says every zone's gauge has its own record (the multi-site setting):
    then the agent is told that the objective is the mean over all gauges and that each
    gauge has its own diagnostic block, which is what makes per-zone reasoning possible.
    ``allow_wildcard`` False removes the ``<param>@*`` shorthand so every update has to
    name a zone -- the arm that tests whether the agent really works in all dimensions.
    """
    n = len(zone_ids)
    if areas:
        listing = ", ".join(f"{z} ({areas[z]:.0f}%)" for z in zone_ids if z in areas)
    else:
        listing = ", ".join(zone_ids)
    head = (
        f"\nThis basin is split into {n} disjoint zones by interior gauges, so every parameter "
        f"exists once per zone and is written `<param>@<zone>`. Zones, in downstream order with "
        f"their share of basin area: {listing}. The first zone is the outlet's own remaining area; "
        f"the others drain through it. "
    )
    if observed:
        obs = (
            "Every zone's gauge has its own observed record and its own diagnostic block below, "
            "and the objective is the mean NSE over all gauges. A zone's parameters affect its own "
            "gauge directly and every gauge downstream of it. "
        )
    else:
        obs = (
            "Only the outlet discharge is observed, so a zone's effect on the hydrograph is "
            "roughly proportional to its area. "
        )
    if allow_wildcard:
        tail = (
            "Write `<param>@*` to set that parameter in every zone at once — use it for basin-wide "
            "moves, and name individual zones only when you have a reason to treat them differently."
        )
    else:
        tail = (
            "Every update must name one specific zone as `<param>@<zone>`; the shorthand "
            "`<param>@*` is NOT accepted and will be rejected. Treat the zones as independent: "
            "give each zone the value its own diagnostics call for, and cover every zone whose "
            "gauge shows a problem."
        )
    return head + obs + tail


def display_parameters(values: Dict[str, float], display_map: Dict[str, str]) -> Dict[str, float]:
    return {display_map[name]: val for name, val in values.items() if name in display_map}


def invert_display_map(display_map: Dict[str, str]) -> Dict[str, str]:
    inverted = {shown: real for real, shown in display_map.items()}
    # `<label>@*` (every zone) has no single real counterpart; map it back to the
    # real base label so coerce_updates can expand it.
    for real, shown in display_map.items():
        if SEP in shown:
            inverted.setdefault(f"{base_name(shown)}{SEP}*", f"{base_name(real)}{SEP}*")
    return inverted


def _unique_bases(names: Sequence[str]) -> list:
    seen, out = set(), []
    for n in names:
        b = base_name(n)
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


# Semantic-split ablation (reviewer 2, contribution of names vs guidance): when on, the
# anonymised arm keeps the neutral labels but receives the parameter guide with the
# labels substituted for the names. Process-level module state so that the manager, the
# proposal agent and the verification agent all render the same guide without a new
# argument threading through every constructor; set once from the CLI before any agent
# is built.
_ANON_GUIDE = {"on": False}


def set_anon_guide(flag: bool) -> None:
    _ANON_GUIDE["on"] = bool(flag)


def anon_guide_on() -> bool:
    """True when the anonymised arm receives the (label-substituted) parameter guide."""
    return bool(_ANON_GUIDE["on"])


def anonymised_guide(display_map: Dict[str, str]) -> str:
    """PHYSICS_PARAMETER_GUIDE with every parameter name replaced by its neutral label."""
    label_of = {base_name(real).upper(): base_name(shown) for real, shown in display_map.items()}
    text = PHYSICS_PARAMETER_GUIDE
    for name in sorted(label_of, key=len, reverse=True):      # ALPHA0 before ALPHA, ...
        text = re.sub(rf"\b{re.escape(name)}\b", label_of[name], text)
    return ("Parameter labels are neutral (" + ", ".join(sorted(set(label_of.values()), key=lambda s: (len(s), s)))
            + "); the descriptions below give their roles in the model.\n" + text)


def render_parameter_guide(use_physics_information: bool,
                           display_map: Dict[str, str],
                           zone_note: str = "") -> str:
    if use_physics_information:
        if _GUIDE_MODE["mode"] == "definitions":
            return PHYSICS_PARAMETER_DEFINITIONS + zone_note
        return PHYSICS_PARAMETER_GUIDE + zone_note
    if _ANON_GUIDE["on"]:
        return anonymised_guide(display_map) + zone_note

    # Zoned runs list the base labels once rather than all 10*(k+1) of them.
    display_names = _unique_bases([display_map[n] for n in display_map])
    frozen = _unique_bases([shown for real, shown in display_map.items()
                            if base_name(real) in FROZEN_PARAMETERS])
    frozen_text = ", ".join(frozen) if frozen else "none"
    return zone_note + (
        "Parameters are anonymized for ablation. Use neutral labels "
        f"{', '.join(display_names)} with no assumed physical meaning. "
        f"Treat {frozen_text} as fixed and leave them unchanged."
    )


def translate_updates(updates: Dict[str, Any], reverse_display_map: Dict[str, str]) -> Dict[str, Any]:
    """Map anonymized parameter names back to their real counterparts."""

    translated: Dict[str, Any] = {}
    for key, value in updates.items():
        real_name = reverse_display_map.get(key, key)
        translated[real_name] = value
    return translated


BOUNDS_RULE = (
    "Values are clipped to parameter_bounds after every update. A parameter listed in "
    "at_bound is already at that limit, so an update pushing it further in the same "
    "direction has NO effect at all. To change that zone's behaviour, move it the other "
    "way or use a different parameter."
)


def bounds_hint(values: Dict[str, float],
                display_map: Dict[str, str],
                rel_tol: float = 1e-6) -> Tuple[Dict[str, list], Dict[str, str]]:
    """The parameter box, and which current values are sitting on it.

    Every optimizer has this information for free; the agent did not, and in the
    multi-site 100-D run it kept proposing ke x1.15 / fc x1.1 in zones where both were
    already clipped at their upper bound, so the proposals did nothing. Bounds are
    keyed by base label (one entry per parameter, shared by all zones); at-bound flags
    are keyed by the full display name so a zone can be addressed directly.
    """
    bounds: Dict[str, list] = {}
    at: Dict[str, str] = {}
    for real, shown in display_map.items():
        b = base_name(real)
        if b in FROZEN_PARAMETERS or b not in PARAM_BOUNDS:
            continue
        lo, hi = PARAM_BOUNDS[b]
        bounds.setdefault(base_name(shown), [lo, hi])
        v = values.get(real)
        if v is None:
            continue
        span = (hi - lo) or 1.0
        if float(v) >= hi - rel_tol * span:
            at[shown] = "upper"
        elif float(v) <= lo + rel_tol * span:
            at[shown] = "lower"
    return bounds, at


def frozen_display_names(display_map: Dict[str, str]) -> Tuple[str, ...]:
    hits = [display_map[name] for name in FROZEN_PARAMETERS if name in display_map]
    if hits:
        return tuple(hits)
    # zoned map: name the frozen parameters once, not once per zone
    return tuple(_unique_bases([shown for real, shown in display_map.items()
                                if base_name(real) in FROZEN_PARAMETERS]))


__all__ = [
    "PHYSICS_PARAMETER_GUIDE",
    "build_display_name_map",
    "build_zone_note",
    "display_parameters",
    "invert_display_map",
    "render_parameter_guide",
    "anon_guide_on",
    "PHYSICS_PARAMETER_DEFINITIONS",
    "set_guide_mode",
    "guide_mode",
    "translate_updates",
    "frozen_display_names",
]
