"""Find and quality-screen the real USGS gauges nested inside a calibration basin.

The high-dimensional experiment so far used *virtual* interior gauges: EF5 partitions the
basin at them and each zone gets its own parameter block, but nothing is observed there,
so the zone parameters are not identifiable from the single outlet hydrograph. Replacing
them with real gauges that have their own records turns 10*(k+1) parameters into a genuinely
identifiable problem -- provided the gauges are actually usable, which is what this checks.

Screening, in order (a site must pass all of them):
  1. it is a stream site in the basin's HUC with a reported drainage area smaller than the
     outlet's, and at least MIN_AREA_FRAC of it -- a zone too small to change the outlet
     hydrograph adds parameters without adding information;
  2. its instantaneous-value record spans the calibration and validation years;
  3. daily-value coverage over those years is at least MIN_COVERAGE -- gaps are what make a
     nominally available gauge useless;
  4. sub-daily data really exists (the IV catalogue can list a site whose record is daily),
     tested on one week of the calibration year.

Nesting and grid snapping are NOT decided here: they need the D8 flow-direction raster, so
they happen in nest_real_gauges.py. What comes out of this is the candidate list
plus the numbers that justify keeping or dropping each one.

Usage: python screen_gauges.py --outlet 01403060 --huc 02030105 --out gauges_<outlet>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
MI2_TO_KM2 = 2.58999
DISCHARGE = "00060"


def fetch(url: str, params: Dict[str, str], tries: int = 3, timeout: int = 90) -> str:
    q = f"{url}?{urllib.parse.urlencode(params)}"
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(q, timeout=timeout) as fh:
                return fh.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:          # NWIS says "no sites/data" with a 404
                return ""
            last = exc
        except Exception as exc:         # noqa: BLE001
            last = exc
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"NWIS request failed after {tries} tries: {q}\n{last!r}")


def rdb_rows(text: str) -> List[Dict[str, str]]:
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 2:
        return []
    hdr = lines[0].split("\t")
    # line 1 is the RDB format ruler ("5s", "15s", ...)
    return [dict(zip(hdr, l.split("\t"))) for l in lines[2:] if l.strip()]


def f(x: Optional[str]) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def catalogue(huc: str) -> List[Dict[str, str]]:
    """Stream sites in the HUC that advertise instantaneous discharge."""
    rows = rdb_rows(fetch(SITE_URL, {
        "format": "rdb", "huc": huc, "siteType": "ST",
        "siteOutput": "expanded", "hasDataTypeCd": "iv", "parameterCd": DISCHARGE,
        "siteStatus": "all",
    }))
    periods = rdb_rows(fetch(SITE_URL, {
        "format": "rdb", "huc": huc, "siteType": "ST",
        "outputDataTypeCd": "iv", "parameterCd": DISCHARGE, "siteStatus": "all",
    }))
    span = {}
    for p in periods:
        if p.get("parm_cd") != DISCHARGE:
            continue
        # a site can list several IV series; keep the widest span
        b, e = p.get("begin_date", ""), p.get("end_date", "")
        cur = span.get(p["site_no"])
        span[p["site_no"]] = (min(b, cur[0]) if cur else b, max(e, cur[1]) if cur else e)
    for r in rows:
        r["iv_begin"], r["iv_end"] = span.get(r["site_no"], ("", ""))
    return rows


def dv_coverage(sites: List[str], start: str, end: str, expect_days: int) -> Dict[str, int]:
    """Days with a daily-mean discharge value, per site (batched)."""
    got: Dict[str, int] = {s: 0 for s in sites}
    for i in range(0, len(sites), 40):
        chunk = sites[i:i + 40]
        txt = fetch(DV_URL, {"format": "json", "sites": ",".join(chunk),
                             "startDT": start, "endDT": end,
                             "parameterCd": DISCHARGE, "statCd": "00003"})
        if not txt:
            continue
        for ts in json.loads(txt).get("value", {}).get("timeSeries", []):
            sid = ts["sourceInfo"]["siteCode"][0]["value"]
            n = sum(1 for v in ts["values"][0]["value"] if f(v.get("value")) is not None
                    and f(v.get("value")) >= 0)
            got[sid] = max(got.get(sid, 0), n)
    return got


def has_subdaily(site: str, day: str) -> Optional[int]:
    """Number of instantaneous values in one week -- proves the record is not daily."""
    txt = fetch(IV_URL, {"format": "json", "sites": site, "startDT": day,
                         "endDT": day[:8] + "07", "parameterCd": DISCHARGE})
    if not txt:
        return 0
    series = json.loads(txt).get("value", {}).get("timeSeries", [])
    return max((len(s["values"][0]["value"]) for s in series), default=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlet", required=True)
    ap.add_argument("--huc", required=True, help="8-digit HUC of the basin")
    ap.add_argument("--cal-year", default="2018")
    ap.add_argument("--val-year", default="2019")
    ap.add_argument("--min-area-frac", type=float, default=0.03,
                    help="drop interior gauges smaller than this share of the outlet area")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="minimum share of days with a daily value, over both years")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    start, end = f"{a.cal_year}-01-01", f"{a.val_year}-12-31"
    expect_days = 365 * 2 + (1 if int(a.cal_year) % 4 == 0 or int(a.val_year) % 4 == 0 else 0)

    print(f"[1] catalogue: stream sites with IV discharge in HUC {a.huc}")
    rows = catalogue(a.huc)
    by_site = {r["site_no"]: r for r in rows}
    print(f"    {len(by_site)} sites")

    out_row = by_site.get(a.outlet)
    if out_row is None:
        out_row = (rdb_rows(fetch(SITE_URL, {"format": "rdb", "sites": a.outlet,
                                             "siteOutput": "expanded"})) or [None])[0]
        if out_row is None:
            print(f"outlet {a.outlet} not found", file=sys.stderr)
            return 1
    out_area_mi2 = f(out_row.get("drain_area_va"))
    if not out_area_mi2:
        print("outlet has no reported drainage area", file=sys.stderr)
        return 1
    out_area = out_area_mi2 * MI2_TO_KM2
    print(f"    outlet {a.outlet} {out_row.get('station_nm','').strip()} -> {out_area:.0f} km2")

    print(f"[2] filter by area and IV span covering {a.cal_year}-{a.val_year}")
    keep, dropped = [], []
    for sid, r in by_site.items():
        if sid == a.outlet:
            continue
        area_mi2 = f(r.get("drain_area_va"))
        area = area_mi2 * MI2_TO_KM2 if area_mi2 else None
        why = None
        if area is None:
            why = "no reported drainage area"
        elif area >= out_area:
            why = f"area {area:.0f} km2 >= outlet"
        elif area < a.min_area_frac * out_area:
            why = f"area {area:.0f} km2 < {100*a.min_area_frac:.0f}% of outlet"
        elif not (r.get("iv_begin", "9999") <= start and r.get("iv_end", "") >= end):
            why = f"IV span {r.get('iv_begin','?')}..{r.get('iv_end','?')} misses the window"
        if why:
            dropped.append((sid, r.get("station_nm", "").strip(), why))
        else:
            keep.append({"site_no": sid, "name": r.get("station_nm", "").strip(),
                         "lon": f(r.get("dec_long_va")), "lat": f(r.get("dec_lat_va")),
                         "area_km2": round(area, 1), "area_frac": round(area / out_area, 4),
                         "huc": r.get("huc_cd", ""),
                         "iv_begin": r.get("iv_begin", ""), "iv_end": r.get("iv_end", "")})
    print(f"    {len(keep)} pass, {len(dropped)} dropped")

    print(f"[3] daily-value coverage over {start}..{end} ({expect_days} days expected)")
    cov = dv_coverage([k["site_no"] for k in keep], start, end, expect_days)
    for k in keep:
        k["days_with_data"] = cov.get(k["site_no"], 0)
        k["coverage"] = round(cov.get(k["site_no"], 0) / expect_days, 4)

    print("[4] sub-daily check (one week of the calibration year)")
    for k in keep:
        n = has_subdaily(k["site_no"], f"{a.cal_year}-03-01")
        k["iv_values_in_week"] = n
        k["subdaily"] = bool(n and n > 7 * 2)

    good = [k for k in keep if k["coverage"] >= a.min_coverage and k["subdaily"]]
    good.sort(key=lambda k: -k["area_km2"])
    marginal = [k for k in keep if k not in good]
    marginal.sort(key=lambda k: -k["area_km2"])

    print(f"\n=== {len(good)} usable interior gauges (area desc) ===")
    print("%-12s %9s %7s %9s %8s  %s" % ("site", "km2", "frac", "coverage", "iv/week", "name"))
    for k in good:
        print("%-12s %9.1f %6.1f%% %8.1f%% %8d  %s" % (
            k["site_no"], k["area_km2"], 100 * k["area_frac"], 100 * k["coverage"],
            k["iv_values_in_week"], k["name"][:52]))
    if marginal:
        print(f"\n=== {len(marginal)} rejected on data quality ===")
        for k in marginal:
            why = []
            if k["coverage"] < a.min_coverage:
                why.append(f"coverage {100*k['coverage']:.0f}%")
            if not k["subdaily"]:
                why.append(f"only {k['iv_values_in_week']} IV values/week")
            print("%-12s %9.1f km2  %-28s %s" % (k["site_no"], k["area_km2"],
                                                 ", ".join(why), k["name"][:44]))
    if dropped:
        print(f"\n=== {len(dropped)} dropped before the data check ===")
        for sid, nm, why in sorted(dropped, key=lambda t: t[2]):
            print("%-12s %-42s %s" % (sid, nm[:42], why))

    payload = {"outlet": a.outlet, "outlet_name": out_row.get("station_nm", "").strip(),
               "outlet_area_km2": round(out_area, 1), "huc": a.huc,
               "window": [start, end], "min_area_frac": a.min_area_frac,
               "min_coverage": a.min_coverage,
               "usable": good, "rejected_quality": marginal,
               "dropped": [{"site_no": s, "name": n, "reason": w} for s, n, w in dropped]}
    out = a.out or f"gauges_{a.outlet}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
