"""Helpers for parameter naming and physics guidance in prompts."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..config import FROZEN_PARAMETERS, PARAM_BOUNDS


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


def _parameter_aliases() -> Dict[str, str]:
    numbered = [f"x{idx + 1}" for idx, _ in enumerate(PARAM_BOUNDS)]
    return {name: alias for name, alias in zip(PARAM_BOUNDS.keys(), numbered)}


def build_display_name_map(use_physics_information: bool) -> Dict[str, str]:
    """Return mapping from real parameter names to prompt-facing labels."""

    aliases = _parameter_aliases()
    return {name: (name if use_physics_information else aliases[name]) for name in PARAM_BOUNDS}


def display_parameters(values: Dict[str, float], display_map: Dict[str, str]) -> Dict[str, float]:
    return {display_map[name]: val for name, val in values.items() if name in display_map}


def invert_display_map(display_map: Dict[str, str]) -> Dict[str, str]:
    return {shown: real for real, shown in display_map.items()}


def render_parameter_guide(use_physics_information: bool, display_map: Dict[str, str]) -> str:
    if use_physics_information:
        return PHYSICS_PARAMETER_GUIDE

    display_names = [display_map[name] for name in PARAM_BOUNDS if name in display_map]
    frozen = [display_map[name] for name in FROZEN_PARAMETERS if name in display_map]
    frozen_text = ", ".join(frozen) if frozen else "none"
    return (
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


def frozen_display_names(display_map: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(display_map[name] for name in FROZEN_PARAMETERS if name in display_map)


__all__ = [
    "PHYSICS_PARAMETER_GUIDE",
    "build_display_name_map",
    "display_parameters",
    "invert_display_map",
    "render_parameter_guide",
    "translate_updates",
    "frozen_display_names",
]
