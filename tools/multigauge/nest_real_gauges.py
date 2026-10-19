"""Which screened USGS gauges actually sit inside the calibration basin, and where.

screen_gauges.py filters on record quality but knows nothing about topology, and the HUC
it searches covers the river below the outlet as well as above it: Lawrence Brook and
Manalapan Brook drain the lower Raritan and would pass every data check while lying
outside the basin entirely. Only the D8 grid can settle that, so this walks downstream
from each candidate's snapped cell and keeps it only if the walk reaches the outlet cell.

For the survivors it also reports what EF5 will actually see:
  * the snapped cell, chosen by matching drainage area the way EF5 does with BASINAREA;
  * grid area vs the area USGS reports -- a big disagreement means the gauge is on the
    wrong branch, not that the gauge is wrong;
  * cell collisions, since two nearby gauges can snap to one cell (Raritan's V2/V3 did);
  * for each k, the resulting zone areas, so a set with a degenerate zone can be rejected
    before any model runs.

Writes nested_<outlet>.json, the real-gauge counterpart of delineate_<outlet>.json.
"""
import argparse
import json
import math
import os
from collections import deque

import numpy as np
import rasterio
from rasterio.windows import from_bounds

BASIC = os.environ.get(
    "INCA_BASIC", "data_cali/basic_data/basic")
D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def read_window(name, L, B, R, T):
    with rasterio.open(f"{BASIC}/{name}_usa.tif") as src:
        w = from_bounds(L, B, R, T, src.transform)
        a = src.read(1, window=w)
        tr = rasterio.windows.transform(w, src.transform)
        off = (int(round(w.row_off)), int(round(w.col_off)))
        return a, tr, off


def snap(facc, tr, lon, lat, area_km2, cell_km2, radius=3):
    """Cell within +-radius whose facc*cell_area best matches the reported area."""
    c0, r0 = ~tr * (lon, lat)
    r0, c0 = int(r0), int(c0)
    r_lo, c_lo = max(0, r0 - radius), max(0, c0 - radius)
    win = facc[r_lo:r0 + radius + 1, c_lo:c0 + radius + 1]
    err = np.abs(np.nan_to_num(win, nan=-1.0) * cell_km2 - area_km2)
    rr, cc = np.unravel_index(np.argmin(err), win.shape)
    return int(r_lo + rr), int(c_lo + cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauges", required=True, help="gauges_<outlet>.json from screen_gauges.py")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--area", type=float, required=True, help="outlet drainage area, km2")
    ap.add_argument("--ks", default="2,4,6,9")
    ap.add_argument("--min-zone-frac", type=float, default=0.02,
                    help="flag a gauge set whose smallest zone is below this share of the basin")
    ap.add_argument("--radius", type=int, default=3,
                    help="snap search half-width in cells (EF5 itself searches about 18 cells)")
    ap.add_argument("--exclude", default="",
                    help=("comma-separated site numbers to drop before choosing zones: "
                          "gauges a few km apart on one stem leave a degenerate zone between "
                          "them, and a gauge just below a reservoir measures operations rather "
                          "than rainfall-runoff"))
    ap.add_argument("--max-area-err", type=float, default=0.10,
                    help="exclude a gauge whose snapped grid area misses the reported one by more")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    g = json.load(open(a.gauges))
    outlet = g["outlet"]
    cands = g["usable"]

    hs = 2.0 * math.sqrt(a.area) / 100.0 + 0.05
    fdir, tr, off = read_window("fdir", a.lon - hs, a.lat - hs, a.lon + hs, a.lat + hs)
    facc, _, _ = read_window("facc", a.lon - hs, a.lat - hs, a.lon + hs, a.lat + hs)
    nrow, ncol = fdir.shape

    lat_mid = tr.f + tr.e * (nrow / 2)
    cell_km2 = (111.32 * abs(tr.e)) * (111.32 * abs(tr.a) * math.cos(math.radians(lat_mid)))

    down, up = {}, {}
    for r in range(nrow):
        for c in range(ncol):
            code = int(fdir[r, c]) if np.isfinite(fdir[r, c]) else 0
            if code in D8:
                dr, dc = D8[code]
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < nrow and 0 <= c2 < ncol:
                    down[(r, c)] = (r2, c2)
                    up.setdefault((r2, c2), []).append((r, c))

    ro, co = snap(facc, tr, a.lon, a.lat, a.area, cell_km2)
    basin = {(ro, co)}
    q = deque([(ro, co)])
    while q:
        cell = q.popleft()
        for u in up.get(cell, []):
            if u not in basin:
                basin.add(u)
                q.append(u)
    A_cells = float(facc[ro, co])
    print(f"outlet {outlet} snapped to row {ro} col {co}  facc {A_cells:.0f} cells  "
          f"basin {len(basin)} cells = {len(basin)*cell_km2:.0f} km2 (USGS {a.area:.0f})")

    rows = []
    outside_window = []
    for c in cands:
        # The raster window is sized from the outlet's drainage area, so it contains the
        # basin and little else. A candidate whose coordinate falls outside it is a
        # sibling basin in the same HUC, tens of km away -- not nested, and snapping it
        # would search an empty array.
        cc_, rr_ = ~tr * (c["lon"], c["lat"])
        if not (0 <= rr_ < nrow and 0 <= cc_ < ncol):
            outside_window.append(c)
            rows.append({**c, "row": None, "col": None, "snap_cells": None, "snap_km": None,
                         "cellx": None, "celly": None, "facc_cells": None, "grid_km2": None,
                         "area_err_frac": 1.0, "in_basin": False, "drains_to_outlet": False,
                         "steps_to_outlet": None, "outside_window": True})
            continue
        r_, c_ = snap(facc, tr, c["lon"], c["lat"], c["area_km2"], cell_km2, radius=a.radius)
        inside = (r_, c_) in basin
        # walk downstream: reaching the outlet cell is the definition of nested
        cur, steps, reached = (r_, c_), 0, False
        while cur in down and steps < 10000:
            cur = down[cur]
            steps += 1
            if cur == (ro, co):
                reached = True
                break
        grid_km2 = float(facc[r_, c_]) * cell_km2
        # how far the snap moved, in cells and km -- a large move can still be right
        # (EF5 searches about 18 cells itself) but it must be visible in the record
        c_rep, r_rep = ~tr * (c["lon"], c["lat"])
        dcell = math.hypot(r_ - r_rep, c_ - c_rep)
        rows.append({**c, "row": r_, "col": c_,
                     "snap_cells": round(dcell, 2),
                     "snap_km": round(dcell * math.sqrt(cell_km2), 2),
                     "cellx": int(off[1] + c_), "celly": int(off[0] + r_),
                     "facc_cells": int(facc[r_, c_]),
                     "grid_km2": round(grid_km2, 1),
                     "area_err_frac": round(abs(grid_km2 - c["area_km2"]) / c["area_km2"], 3),
                     "in_basin": bool(inside), "drains_to_outlet": bool(reached),
                     "steps_to_outlet": steps if reached else None})

    if outside_window:
        print(f"\n{len(outside_window)} candidates fall outside the basin's raster window "
              f"(sibling basins in the same HUC): " +
              ", ".join(f"{c['site_no']}({c['area_km2']:.0f} km2)" for c in outside_window))
    print("\n%-12s %9s %9s %7s %7s %6s %7s  %s" %
          ("site", "USGS km2", "grid km2", "err", "nested", "steps", "snap km", "name"))
    for r in sorted(rows, key=lambda r: -r["area_km2"]):
        if r.get("outside_window"):
            print("%-12s %9.1f %9s %7s %7s %6s %7s  %s" % (
                r["site_no"], r["area_km2"], "-", "-", "OUT", "-", "-", r["name"][:40]))
            continue
        print("%-12s %9.1f %9.1f %6.1f%% %7s %6s %7.1f  %s" % (
            r["site_no"], r["area_km2"], r["grid_km2"], 100 * r["area_err_frac"],
            "yes" if r["drains_to_outlet"] else "NO", r["steps_to_outlet"] or "-",
            r["snap_km"], r["name"][:40]))

    nested = [r for r in rows if r["drains_to_outlet"]]
    bad_area = [r for r in nested if r["area_err_frac"] > a.max_area_err]
    if bad_area:
        print("\nexcluded on area disagreement (> %.0f%%): the modelled catchment at the snapped"
              " cell is not the gauge's catchment, so its NSE would be biased by volume alone"
              % (100 * a.max_area_err))
        for r in bad_area:
            print("  %-12s USGS %.1f vs grid %.1f km2  (%.0f%%)  %s"
                  % (r["site_no"], r["area_km2"], r["grid_km2"], 100 * r["area_err_frac"],
                     r["name"][:40]))
    nested = [r for r in nested if r["area_err_frac"] <= a.max_area_err]
    drop = {s.strip() for s in a.exclude.split(",") if s.strip()}
    if drop:
        kept = [r for r in nested if r["site_no"] not in drop]
        print("\nexcluded by --exclude: " + ", ".join(
            f"{r['site_no']}({r['area_km2']:.0f} km2)" for r in nested if r["site_no"] in drop))
        nested = kept
    nested.sort(key=lambda r: -r["area_km2"])
    cells = {}
    for r in nested:
        cells.setdefault((r["row"], r["col"]), []).append(r["site_no"])
    dup = {k: v for k, v in cells.items() if len(v) > 1}
    print(f"\n{len(nested)} of {len(rows)} gauges are nested inside the basin")
    if dup:
        print("  !! cell collisions (EF5 would merge these):", dup)

    # zones for each k: the k largest nested gauges plus the outlet
    zones = {}
    for k in [int(x) for x in a.ks.split(",")]:
        chosen = nested[:k]
        if len(chosen) < k:
            print(f"  k={k}: only {len(chosen)} nested gauges available, skipped")
            continue
        gauge_cells = {(c["row"], c["col"]): i + 1 for i, c in enumerate(chosen)}
        gauge_cells[(ro, co)] = 0
        local = {i: 0 for i in range(len(chosen) + 1)}
        memo = {}
        for cell in basin:
            path, cur = [], cell
            while cur not in gauge_cells and cur not in memo:
                path.append(cur)
                cur = down.get(cur)
                if cur is None or cur not in basin:
                    cur = (ro, co)
                    break
            gz = gauge_cells[cur] if cur in gauge_cells else memo[cur]
            for p in path:
                memo[p] = gz
            local[gz] += 1
        fr = [round(local[i] / len(basin), 4) for i in range(len(chosen) + 1)]
        zones[k] = {"n_zones": len(chosen) + 1, "dim": 10 * (len(chosen) + 1),
                    "gauges": [outlet] + [c["site_no"] for c in chosen],
                    "local_area_frac": fr, "min_frac": min(fr), "max_frac": max(fr),
                    "degenerate": min(fr) < a.min_zone_frac}
        flag = "  <-- has a zone below %.0f%%" % (100 * a.min_zone_frac) if min(fr) < a.min_zone_frac else ""
        print("  k=%d  dim=%3d  zones %2d  local area %.1f%%..%.1f%%%s"
              % (k, 10 * (len(chosen) + 1), len(chosen) + 1, 100 * min(fr), 100 * max(fr), flag))
        print("        %s" % ", ".join("%s:%.1f%%" % (gid, 100 * fr[i])
                                        for i, gid in enumerate(zones[k]["gauges"])))

    rec = {"outlet": outlet, "outlet_area_km2": a.area,
           "outlet_snapped": {"row": ro, "col": co, "cellx": int(off[1] + co),
                              "celly": int(off[0] + ro), "facc_cells": A_cells},
           "n_cells": len(basin), "cell_km2": round(cell_km2, 4),
           "delineated_km2": round(len(basin) * cell_km2, 1),
           "gauges": rows, "nested": nested, "collisions": {str(k): v for k, v in dup.items()},
           "zones": zones}
    out = a.out or f"nested_{outlet}.json"
    json.dump(rec, open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
