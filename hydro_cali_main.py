#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hydro_cali_main.py (simplified)

Key changes:
- Replace individual DEM/DDM/FAM flags with --basic_data_path
  expecting: dem_usa.tif, fdir_usa.tif, facc_usa.tif inside it.
- Replace individual parameter grid flags with --default_param_dir
  expecting subfolders and files:
    <default_param_dir>/crest_params/{wm_usa.tif, im_usa.tif, ksat_usa.tif, b_usa.tif}
    <default_param_dir>/kw_params/{leaki_usa.tif, alpha_usa.tif, beta_usa.tif, alpha0_usa.tif}
- Still: downloads USGS gauge CSV, writes control.txt, and runs calibration.
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Any, Union

import requests

# ----------------------------- constants ---------------------------------
MI2_TO_KM2 = 2.58999


# ------------------------ USGS helpers (robust) --------------------------
@dataclass
class UsgsSiteInfo:
    site_no: str
    latitude: float
    longitude: float
    drainage_area_km2: Optional[float]  # may be None


def _parse_rdb_site(text: str, site_no: str):
    """Parse NWIS RDB (siteOutput=expanded) and return (lat, lon, area_km2 or None)."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 3:
        raise ValueError("RDB response is incomplete.")
    header = lines[0].strip().split("\t")
    rows = [ln.strip().split("\t") for ln in lines[2:] if ln.strip()]
    records = [dict(zip(header, row)) for row in rows]
    rec = next((r for r in records if r.get("site_no") == site_no),
               (records[0] if records else None))
    if not rec:
        raise ValueError(f"Site {site_no} not found in RDB.")

    lat = float(rec["dec_lat_va"])
    lon = float(rec["dec_long_va"])
    da_km2 = None
    mi2 = (rec.get("drain_area_va") or "").strip()
    if mi2 and mi2.upper() != "NA":
        try:
            da_km2 = float(mi2) * MI2_TO_KM2
        except ValueError:
            da_km2 = None
    return lat, lon, da_km2


def _extract_drnarea_from_streamstats(obj: Union[dict, list]) -> Optional[float]:
    """Search any nested 'parameters' for DRNAREA and return km^2."""
    if isinstance(obj, dict):
        params = obj.get("parameters")
        if isinstance(params, list):
            for p in params:
                code = str(p.get("code", "")).upper()
                if code == "DRNAREA":
                    val = p.get("value")
                    units = (p.get("unit") or p.get("units") or "").lower()
                    if val is None:
                        continue
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        continue
                    if "square kilometer" in units or "sq km" in units or "km" in units:
                        return val
                    return val * MI2_TO_KM2
        for v in obj.values():
            got = _extract_drnarea_from_streamstats(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for item in obj:
            got = _extract_drnarea_from_streamstats(item)
            if got is not None:
                return got
    return None


def _streamstats_drainage_area_km2(lat: float, lon: float, timeout: int = 30) -> Optional[float]:
    """Call StreamStats watershed service (no explicit rcode) and read DRNAREA."""
    url = "https://streamstats.usgs.gov/streamstatsservices/watershed.geojson"
    params = {
        "x": lon, "y": lat, "crs": 4326,
        "includeparameters": "true",
        "includefeatures": "false",
        "simplify": "true",
    }
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return _extract_drnarea_from_streamstats(data)


def get_usgs_site_info(site_no: str, timeout: int = 20) -> UsgsSiteInfo:
    """Fetch site lon/lat and drainage area (km^2)."""
    base = "https://waterservices.usgs.gov/nwis/site/"
    headers = {"User-Agent": "python-requests/usgs-helper"}
    params_rdb = {"sites": site_no, "format": "rdb", "siteOutput": "expanded", "siteStatus": "all"}
    resp = requests.get(base, params=params_rdb, headers=headers, timeout=timeout)
    resp.raise_for_status()
    lat, lon, da_km2 = _parse_rdb_site(resp.text, site_no)
    if da_km2 is None:
        try:
            da_km2 = _streamstats_drainage_area_km2(lat, lon, timeout=max(30, timeout))
        except Exception:
            pass
    return UsgsSiteInfo(site_no=site_no, latitude=lat, longitude=lon, drainage_area_km2=da_km2)


# --------------------- control.txt templating utilities -------------------
DEFAULT_TEMPLATE = """[Basic]
DEM={DEM_PATH}
DDM={DDM_PATH}
FAM={FAM_PATH}
PROJ=geographic
ESRIDDM=true
SelfFAM=false

[PrecipForcing MRMS]
TYPE=TIF
UNIT=mm/h
FREQ=1h
LOC={PRECIP_PATH}
NAME={PRECIP_NAME}

[PETForcing PET]
TYPE=TIF
UNIT=mm/100d
FREQ=d
LOC={PET_PATH}
NAME={PET_NAME}

[Gauge {SITE_NO}]
LON={LON}
LAT={LAT}
OBS={OBS_PATH}
OUTPUTTS=TRUE
WANTCO=TRUE
BASINAREA={BASINAREA}

[Basin 0]
GAUGE={SITE_NO}

[CrestParamSet CrestParam]
gauge={SITE_NO}
WM_GRID={WM_GRID}
IM_GRID={IM_GRID}
FC_GRID={FC_GRID}
B_GRID={B_GRID}
wm={WM}
b={B}
im={IM}
ke={KE}
fc={FC}
iwu={IWU}

[kwparamset KWParam]
gauge={SITE_NO}
leaki_grid={LEAKI_GRID}
alpha_grid={ALPHA_GRID}
beta_grid={BETA_GRID}
alpha0_grid={ALPHA0_GRID}
under={UNDER}
leaki={LEAKI}
th={TH}
isu={ISU}
alpha={ALPHA}
beta={BETA}
alpha0={ALPHA0}

[Task Simu]
STYLE=SIMU
MODEL={MODEL}
ROUTING={ROUTING}
BASIN=0
PRECIP=MRMS
PET=PET
OUTPUT={RESULTS_OUTDIR}
PARAM_SET=CrestParam
ROUTING_PARAM_Set=KWParam
TIMESTEP={TIME_STEP}
TIME_BEGIN={TIME_BEGIN}
TIME_END={TIME_END}

[Execute]
TASK=Simu
""".rstrip() + "\n"


def _format_lat(lat: float) -> str:
    return f"{lat:.4f}"


def _format_lon(lon: float) -> str:
    return f"{lon:.4f}"


def _format_area_km2(area: float) -> str:
    return f"{area:.2f}"


def build_control_text(template: str,
                       site_no: str,
                       lon: float,
                       lat: float,
                       basin_km2: Optional[float],
                       **kwargs) -> str:
    """Fill the control template with all values."""
    return template.format(
        SITE_NO=site_no,
        LON=_format_lon(lon),
        LAT=_format_lat(lat),
        BASINAREA=_format_area_km2(basin_km2) if basin_km2 is not None else "NA",
        **kwargs
    )


# --------------------------- calibration runner --------------------------
def try_import_manager():
    """Import TwoStageCalibrationManager from common locations."""
    try:
        from hydrocalib.agents.manager import TwoStageCalibrationManager
        return TwoStageCalibrationManager
    except Exception:
        try:
            from aquah_cali.hydrocalib.agents.manager import TwoStageCalibrationManager
            return TwoStageCalibrationManager
        except Exception as e:
            raise ImportError(
                "Cannot import TwoStageCalibrationManager. "
                "Ensure 'hydrocalib' (or 'aquah_cali.hydrocalib') is importable."
            ) from e


# ------------------------------ CLI main ---------------------------------
def parse_args() -> argparse.Namespace:
    # fromfile_prefix_chars='@' means: args can be loaded from a text file
    # Example: python hydro_cali_main.py @args.txt
    p = argparse.ArgumentParser(
        description="Hydro calibration end-to-end driver (simplified)",
        fromfile_prefix_chars='@'
    )

    # Required core
    p.add_argument("--site_num", required=True, help="USGS gauge id, e.g., 08069000")

    # Root folders (your simplification)
    p.add_argument("--basic_data_path", required=True,
                   help="Folder containing dem_usa.tif, fdir_usa.tif, facc_usa.tif")
    p.add_argument("--default_param_dir", required=True,
                   help="Folder containing crest_params/ and kw_params/ subfolders")

    # Folder layout
    p.add_argument("--cali_set_dir", default="./cali_set",
                   help="Base folder to hold <site>_<tag>/control.txt")
    p.add_argument("--cali_tag", default="2018", help="Suffix tag for the calibration folder name")

    # Forcings
    p.add_argument("--precip_path", required=True, help="Folder of precipitation rasters")
    p.add_argument("--precip_name", required=True, help="File name pattern, e.g. GaugeCorr_QPE_....tif")
    p.add_argument("--pet_path", required=True, help="Folder of PET rasters")
    p.add_argument("--pet_name", required=True, help="File name pattern, e.g. etYYYYMMDD.tif")

    # Gauge CSV output and run results
    p.add_argument("--gauge_outdir", required=True, help="Folder to save USGS hourly CSV")
    p.add_argument("--results_outdir", required=True, help="CREST outputs folder")

    # Time config
    p.add_argument("--time_begin", required=True, help="YYYYMMDDhhmm, e.g., 201801010000")
    p.add_argument("--time_end", required=True, help="YYYYMMDDhhmm, e.g., 201812312300")
    p.add_argument("--time_step", default="1h", help="CREST time step, e.g., 1h")

    # Model/routing
    p.add_argument("--model", default="CREST")
    p.add_argument("--routing", default="KW")

    # Scalar parameter defaults
    p.add_argument("--wm", type=float, default=1.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--im", type=float, default=1.0)
    p.add_argument("--ke", type=float, default=1.0)
    p.add_argument("--fc", type=float, default=1.0)
    p.add_argument("--iwu", type=float, default=25.0)
    p.add_argument("--under", type=float, default=1.0)
    p.add_argument("--leaki", type=float, default=1.0)
    p.add_argument("--th", type=float, default=10.0)
    p.add_argument("--isu", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha0", type=float, default=0.0)

    # Downloader invocation
    p.add_argument("--python_exec", default=sys.executable,
                   help="Python executable used to call usgs_gauge_download.py")
    p.add_argument("--usgs_script_path", default="./usgs_gauge_download.py",
                   help="Path to usgs_gauge_download.py")
    p.add_argument("--skip_download", action="store_true",
                   help="If set, do not call usgs_gauge_download.py")

    # Calibration manager knobs
    p.add_argument("--n_candidates", type=int, default=8)
    p.add_argument("--n_peaks", type=int, default=3)
    p.add_argument("--max_rounds", type=int, default=20)

    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_obs_csv_path(gauge_outdir: str, site_no: str) -> str:
    """OBS file naming follows your convention: USGS_<id>_1h_UTC.csv"""
    ensure_dir(gauge_outdir)
    return os.path.join(gauge_outdir, f"USGS_{site_no}_1h_UTC.csv")


def run_usgs_downloader(python_exec: str,
                        script_path: str,
                        site_no: str,
                        time_begin: str,
                        time_end: str,
                        time_step: str,
                        outdir: str) -> None:
    """Call your existing usgs_gauge_download.py via subprocess."""
    cmd = [
        python_exec, script_path,
        "--site_num", site_no,
        "--time_start", time_begin[:10],  # expects YYYYMMDDhh
        "--time_end", time_end[:10],
        "--time_step", time_step,
        "--output", outdir
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_control_file(control_folder: str, control_text: str) -> str:
    ensure_dir(control_folder)
    out_path = os.path.join(control_folder, "control.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(control_text)
    return out_path


def main():
    args = parse_args()
    site = args.site_num

    # Resolve fixed filenames from the two root folders
    dem_path = os.path.join(args.basic_data_path, "dem_usa.tif")
    ddm_path = os.path.join(args.basic_data_path, "fdir_usa.tif")
    fam_path = os.path.join(args.basic_data_path, "facc_usa.tif")

    crest_dir = os.path.join(args.default_param_dir, "crest_params")
    kw_dir = os.path.join(args.default_param_dir, "kw_params")

    wm_grid = os.path.join(crest_dir, "wm_usa.tif")
    im_grid = os.path.join(crest_dir, "im_usa.tif")
    fc_grid = os.path.join(crest_dir, "ksat_usa.tif")
    b_grid  = os.path.join(crest_dir, "b_usa.tif")

    leaki_grid = os.path.join(kw_dir, "leaki_usa.tif")
    alpha_grid = os.path.join(kw_dir, "alpha_usa.tif")
    beta_grid  = os.path.join(kw_dir, "beta_usa.tif")
    alpha0_grid= os.path.join(kw_dir, "alpha0_usa.tif")

    # Optional path validation (warn if any expected file is missing)
    for pth in [dem_path, ddm_path, fam_path,
                wm_grid, im_grid, fc_grid, b_grid,
                leaki_grid, alpha_grid, beta_grid, alpha0_grid]:
        if not os.path.isfile(pth):
            print(f"[WARN] Expected file not found: {pth}")

    # 1) Fetch lon/lat/area
    print(f"[INFO] Fetching site info for {site} ...")
    info = get_usgs_site_info(site)
    print(f"[INFO]  lat={info.latitude:.6f}, lon={info.longitude:.6f}, "
          f"area_km2={'NA' if info.drainage_area_km2 is None else f'{info.drainage_area_km2:.2f}'}")

    # 2) Resolve folders and OBS CSV path
    control_folder = os.path.join(args.cali_set_dir, f"{site}_{args.cali_tag}")
    obs_csv_path = build_obs_csv_path(args.gauge_outdir, site)

    # 3) Optionally download the hourly CSV
    if not args.skip_download:
        run_usgs_downloader(
            python_exec=args.python_exec,
            script_path=args.usgs_script_path,
            site_no=site,
            time_begin=args.time_begin,
            time_end=args.time_end,
            time_step=args.time_step,
            outdir=args.gauge_outdir
        )
    else:
        print("[INFO] Skipping gauge download as requested.")

    # 4) Build control.txt content
    control_text = build_control_text(
        template=DEFAULT_TEMPLATE,
        site_no=site,
        lon=info.longitude,
        lat=info.latitude,
        basin_km2=info.drainage_area_km2,
        DEM_PATH=dem_path,
        DDM_PATH=ddm_path,
        FAM_PATH=fam_path,
        PRECIP_PATH=args.precip_path,
        PRECIP_NAME=args.precip_name,
        PET_PATH=args.pet_path,
        PET_NAME=args.pet_name,
        OBS_PATH=obs_csv_path,
        WM_GRID=wm_grid,
        IM_GRID=im_grid,
        FC_GRID=fc_grid,
        B_GRID=b_grid,
        WM=args.wm, B=args.b, IM=args.im, KE=args.ke, FC=args.fc, IWU=args.iwu,
        LEAKI_GRID=leaki_grid, ALPHA_GRID=alpha_grid, BETA_GRID=beta_grid, ALPHA0_GRID=alpha0_grid,
        UNDER=args.under, LEAKI=args.leaki, TH=args.th, ISU=args.isu,
        ALPHA=args.alpha, BETA=args.beta, ALPHA0=args.alpha0,
        MODEL=args.model, ROUTING=args.routing,
        RESULTS_OUTDIR=args.results_outdir,
        TIME_STEP=args.time_step,
        TIME_BEGIN=args.time_begin,
        TIME_END=args.time_end,
    )

    # 5) Write control.txt
    control_path = write_control_file(control_folder, control_text)
    print(f"[INFO] control.txt written to: {control_path}")

    # 6) Run calibration
    TwoStageCalibrationManager = try_import_manager()
    calib = TwoStageCalibrationManager(
        args,
        simu_folder=os.path.relpath(control_folder, start=os.getcwd()),
        gauge_num=site,
        n_candidates=args.n_candidates,
        n_peaks=args.n_peaks,
    )
    print(f"[INFO] Starting calibration (max_rounds={args.max_rounds}) ...")
    calib.run(max_rounds=args.max_rounds)
    print("[INFO] Calibration finished.")


if __name__ == "__main__":
    main()
