"""Zoned (per-gauge) parameters: the high-dimensional calibration configuration.

EF5 lets a parameter set carry one block per gauge (``gauge=<id>`` starts a block inside
``[CrestParamSet ...]`` / ``[kwparamset ...]``), and assigns every grid cell to the first
gauge met walking downstream (BasicGrids.cpp, the walkNodes loop). Listing the outlet plus
k interior "virtual" gauges therefore partitions the basin into k+1 zones, each with its
own copy of the 10 free multipliers: dim = 10 * (k + 1). Verified numerically on
02338660 (highdim/smoke_02338660): equal blocks reproduce the single-gauge run exactly,
and changing only the outlet block leaves every virtual gauge's series bit-identical.

Naming: a zoned parameter is ``<base>@<gauge id>``, e.g. ``wm@02338660`` (outlet) or
``wm@V1``. Bounds and the frozen set are looked up by the base name, so the rest of the
package needs no zone-specific tables. A parameter set is "zoned" when any key carries
the separator; plain names keep the legacy single-gauge behaviour untouched.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import FROZEN_PARAMETERS, PARAM_BOUNDS

SEP = "@"
CREST_KEYS: Tuple[str, ...] = ("wm", "b", "im", "ke", "fc", "iwu")
KW_KEYS: Tuple[str, ...] = ("under", "leaki", "th", "isu", "alpha", "beta", "alpha0")
DEFAULT_VALUES: Dict[str, float] = {
    "wm": 1.0, "b": 1.0, "im": 1.0, "ke": 1.0, "fc": 1.0, "iwu": 25.0,
    "under": 1.0, "leaki": 1.0, "th": 10.0, "isu": 0.0, "alpha": 1.0, "beta": 1.0, "alpha0": 0.0,
}
# which parameter set each key lives in
_SET_OF = {**{k: "crest" for k in CREST_KEYS}, **{k: "kw" for k in KW_KEYS}}
_SET_HEADER = {
    "crest": re.compile(r"(?im)^\[CrestParamSet\s+\w+\]\s*$"),
    "kw": re.compile(r"(?im)^\[kwparamset\s+\w+\]\s*$"),
}
_ANY_HEADER = re.compile(r"(?m)^\[[^\]]+\]\s*$")
_GAUGE_LINE = re.compile(r"(?im)^\s*gauge\s*=\s*(\S+)\s*$")


# --- names ---------------------------------------------------------------------------
def base_name(name: str) -> str:
    return name.split(SEP, 1)[0]


def zone_of(name: str) -> Optional[str]:
    return name.split(SEP, 1)[1] if SEP in name else None


def zoned_name(base: str, zone: str) -> str:
    return f"{base}{SEP}{zone}"


def is_zoned(names: Iterable[str]) -> bool:
    return any(SEP in n for n in names)


def zoned_free_names(zones: Sequence[str]) -> List[str]:
    """Free parameter names, zone-major: all of zone 0, then zone 1, ..."""
    return [zoned_name(b, z) for z in zones for b in PARAM_BOUNDS if b not in FROZEN_PARAMETERS]


def expand_base(values: Dict[str, float], zones: Sequence[str]) -> Dict[str, float]:
    """Replicate a single-gauge parameter dict into every zone (the zoned starting point)."""
    return {zoned_name(b, z): float(values.get(b, DEFAULT_VALUES[b])) for z in zones for b in PARAM_BOUNDS}


def collapse_to_base(values: Dict[str, float], zone: str) -> Dict[str, float]:
    """The plain 13-parameter dict of one zone (e.g. to compare with a single-gauge run)."""
    return {base_name(k): v for k, v in values.items() if zone_of(k) == zone}


# --- control-file surgery ---------------------------------------------------------------
def _section_span(text: str, which: str) -> Tuple[int, int]:
    m = _SET_HEADER[which].search(text)
    if not m:
        raise ValueError(f"control has no {which} parameter set section")
    nxt = _ANY_HEADER.search(text, m.end())
    return m.start(), (nxt.start() if nxt else len(text))


def _block_span(section: str, gauge_id: str) -> Tuple[int, int]:
    """Span of the ``gauge=<id>`` block inside one parameter-set section."""
    starts = [(mm.start(), mm.end(), mm.group(1)) for mm in _GAUGE_LINE.finditer(section)]
    for i, (s, e, gid) in enumerate(starts):
        if gid.lower() == gauge_id.lower():
            end = starts[i + 1][0] if i + 1 < len(starts) else len(section)
            return s, end
    raise ValueError(f"no 'gauge={gauge_id}' block in this parameter set")


def gauges_in_basin(text: str) -> List[str]:
    """Gauge ids listed under [Basin 0], in order (outlet first by our convention)."""
    m = re.search(r"(?ims)^\[Basin\s+\w+\]\s*$(.*?)(?=^\[|\Z)", text)
    if not m:
        raise ValueError("control has no [Basin] section")
    return re.findall(r"(?im)^\s*GAUGE\s*=\s*(\S+)\s*$", m.group(1))


def render_zoned(text: str, values: Dict[str, float]) -> str:
    """Write ``<base>@<gauge>`` values into their own gauge blocks, nothing else.

    Every zoned key must exist in its block already (the block generator writes all
    13 lines), so a typo in a zone or key fails loudly instead of silently steering a
    different zone. Plain (un-zoned) keys are rejected here: mixing schemes in one set
    is never what anyone wants.
    """
    plain = [k for k in values if SEP not in k]
    if plain:
        raise ValueError(f"render_zoned got un-zoned keys {plain[:5]}; use the legacy path")
    out = text
    for key, value in values.items():
        base, zone = base_name(key), zone_of(key)
        if base not in _SET_OF:
            raise KeyError(f"unknown parameter {base!r} in {key!r}")
        s0, s1 = _section_span(out, _SET_OF[base])
        section = out[s0:s1]
        b0, b1 = _block_span(section, zone)
        block = section[b0:b1]
        pat = re.compile(rf"(?im)^(\s*{re.escape(base)}\s*=\s*)[^\n]*$")
        new_block, n = pat.subn(lambda m: f"{m.group(1)}{value}", block, count=1)
        if n != 1:
            raise ValueError(f"block gauge={zone} has no '{base}=' line")
        out = out[:s0] + section[:b0] + new_block + section[b1:] + out[s1:]
    return out


def build_control(base_text: str,
                  candidates: Sequence[Dict[str, float]],
                  outlet: str,
                  k: int,
                  overrides: Optional[Dict[str, Dict[str, float]]] = None,
                  param_zones: Optional[Sequence[str]] = None,
                  simu_begin: Optional[str] = None,
                  simu_end: Optional[str] = None,
                  warmup_begin: Optional[str] = None,
                  output_dir: Optional[str] = None) -> Tuple[str, List[str]]:
    """Turn a single-gauge control into an outlet + k virtual-gauge control.

    ``candidates`` are tributary outlets from delineate.py, ranked by drainage area; the
    top-k become gauges V1..Vk pinned to their exact grid cell via CELLX/CELLY (global
    column/row of the CONUS grid; EF5 skips projection and, with no BASINAREA, skips the
    area search) with OUTPUTTS, so EF5 writes ts.<name>.crest.csv with a NaN Observed
    column. Candidates lacking cell indices fall back to LON/LAT + BASINAREA snapping.
    A gauge listed in ``param_zones`` gets its own full parameter block (EF5 errors on a
    block that omits a key); the others are still gauges -- they are simulated, written out
    and compared with their OBS -- but inherit their downstream neighbour's parameters
    (BasicGrids.cpp:990-1015). Passing ``param_zones=[outlet]`` is therefore the lumped
    control arm: one parameter set, judged against every record. The default gives every
    gauge its own block, which is the high-dimensional arm.

    Returns the control text and the ids that carry a parameter block, i.e. the zones,
    in [Basin] order (outlet first).
    """
    virt = list(candidates)[:k]
    # A candidate carrying ``site_no`` is a real gauge: it keeps its USGS id and, with
    # ``obs``, its own record, which is what turns the zone parameters from unidentifiable
    # into identifiable. Candidates without it stay the virtual V1..Vk of the earlier runs.
    ids = [outlet] + [str(c.get("site_no") or f"V{i + 1}") for i, c in enumerate(virt)]
    zone_ids = [g for g in ids if g in set(param_zones)] if param_zones is not None else list(ids)
    if not zone_ids or zone_ids[0] != outlet:
        raise ValueError(f"param_zones must include the outlet {outlet!r}; got {param_zones!r}")
    params = {g: dict(DEFAULT_VALUES) for g in zone_ids}
    for g, kv in (overrides or {}).items():
        params[g].update({kk: float(vv) for kk, vv in kv.items()})

    def section(txt: str, header_regex: str) -> re.Match:
        m = re.search(rf"(?ms)^\[{header_regex}\].*?(?=^\[|\Z)", txt)
        if not m:
            raise ValueError(f"section {header_regex} not found")
        return m

    text = base_text
    g = section(text, rf"Gauge {re.escape(outlet)}")

    def gauge_block(i: int, c: Dict[str, float]) -> str:
        # Pin the gauge to its exact grid cell. With LON/LAT + BASINAREA, EF5 searches up
        # to ~18 cells for the best area match and two tributaries of similar size can
        # collapse onto one cell (Raritan V2/V3 did). CELLX/CELLY alone skips the
        # projection, and *no* BASINAREA skips the search (BasicGrids.cpp: HasObsFlowAccum).
        # The snapping was already done, and audited, by nest_real_gauges.py.
        gid = ids[i + 1]
        obs = f"OBS={c['obs']}\n" if c.get("obs") else ""
        if "cellx" in c and "celly" in c:
            return (f"\n[Gauge {gid}]\nCELLX={int(c['cellx'])}\nCELLY={int(c['celly'])}\n"
                    f"OUTPUTTS=TRUE\n{obs}")
        return (f"\n[Gauge {gid}]\nLON={c['lon']}\nLAT={c['lat']}\nOUTPUTTS=TRUE\n"
                f"BASINAREA={c['area_km2']}\n{obs}")

    blocks = "".join(gauge_block(i, c) for i, c in enumerate(virt))
    text = text[:g.end()] + blocks + text[g.end():]
    b = section(text, r"Basin \w+")
    text = text[:b.start()] + "[Basin 0]\n" + "".join(f"GAUGE={i}\n" for i in ids) + "\n" + text[b.end():]

    def rebuild(txt: str, header_regex: str, keys: Sequence[str]) -> str:
        s = section(txt, header_regex)
        lines = s.group(0).splitlines()
        grids = [ln for ln in lines[1:] if re.match(r"(?i)^\s*\w+_grid\s*=", ln)]
        new = lines[0] + "\n" + "\n".join(grids) + "\n"
        for i in zone_ids:
            new += f"gauge={i}\n" + "".join(f"{p}={params[i][p]}\n" for p in keys)
        return txt[:s.start()] + new + "\n" + txt[s.end():]

    text = rebuild(text, r"CrestParamSet \w+", CREST_KEYS)
    text = rebuild(text, r"kwparamset \w+", KW_KEYS)

    def set_task(txt: str, task: str, field: str, value: Optional[str]) -> str:
        if value is None:
            return txt
        m = re.search(rf"(?ms)^\[Task {task}\].*?(?=^\[|\Z)", txt)
        if not m:
            return txt
        blk = re.sub(rf"(?m)^{field}=.*$", f"{field}={value}", m.group(0))
        return txt[:m.start()] + blk + txt[m.end():]

    text = set_task(text, "warmup", "TIME_BEGIN", warmup_begin)
    text = set_task(text, "Simu", "TIME_BEGIN", simu_begin)
    text = set_task(text, "Simu", "TIME_END", simu_end)
    if output_dir:
        text = re.sub(r"(?m)^(OUTPUT|STATES)=.*$", lambda m: f"{m.group(1)}={output_dir}/", text)
    return text, zone_ids


__all__ = ["SEP", "base_name", "zone_of", "zoned_name", "is_zoned", "zoned_free_names",
           "expand_base", "collapse_to_base", "gauges_in_basin", "render_zoned", "build_control"]
