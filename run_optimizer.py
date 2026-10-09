#!/usr/bin/env python
"""Run a non-LLM optimizer baseline against the same EF5 setup as INCA.

Every method searches the ten free entries of ``PARAM_BOUNDS`` (the 13 CREST/KW
parameters minus ``FROZEN_PARAMETERS``), maximises the selected objective, and
stops as soon as ``--budget`` EF5 forward evaluations have been consumed.  The
complete evaluation trace - including the ``running_best`` array used for the
convergence figures - is written to ``optimizer_history.json``.

Examples
--------
    python run_optimizer.py --method sceua --site_num 02338660 \
        --simu_folder cali_set/02338660_2018 --budget 1000 --seed 0

    python run_optimizer.py --method rule --objective composite \
        --site_num 02338660 --simu_folder cali_set/02338660_2018 \
        --n_candidates 8 --failure_patient 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

# The optimizers only do small dense linear algebra (GP factorisations, CMA-ES
# eigendecompositions); on a many-core node the default BLAS thread pool costs
# far more in synchronisation than it saves.  The real work happens in the EF5
# subprocesses, so cap the pool here (before numpy is imported) unless the user
# has already chosen a value.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "2")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ``hydrocalib.__init__`` imports the LLM agents, which instantiate an OpenAI
# client at import time.  Loading the repository .env first keeps this driver
# usable from any working directory even though no LLM call is ever made.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

from hydrocalib.config import (  # noqa: E402
    DEFAULT_PEAK_PICK_KWARGS,
    FROZEN_PARAMETERS,
    PARAM_BOUNDS,
)
from hydrocalib.optimizers import (  # noqa: E402
    METHODS,
    OBJECTIVES,
    EF5Evaluator,
    SearchSpace,
    build_checkpoint,
    run_optimization,
    save_history,
)
from hydrocalib.parameters import ParameterSet, safe_clip  # noqa: E402
from hydrocalib.simulation import SimulationRunner  # noqa: E402


# Fallback starting point: the CLI defaults of hydro_cali_main.py.
DEFAULT_PARAMS: Dict[str, float] = {
    "wm": 1.0, "b": 1.0, "im": 1.0, "ke": 1.0, "fc": 1.0, "iwu": 25.0,
    "under": 1.0, "leaki": 1.0, "th": 10.0, "isu": 0.0,
    "alpha": 1.0, "beta": 1.0, "alpha0": 0.0,
}


def read_params_from_control(control_path: Path) -> Dict[str, float]:
    """Extract the scalar parameter values written in a control.txt."""
    if not control_path.exists():
        return {}
    text = control_path.read_text()
    found: Dict[str, float] = {}
    for name in PARAM_BOUNDS:
        match = re.search(rf"^\s*{name}\s*=\s*([0-9eE.+-]+)\s*$", text,
                          flags=re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                found[name] = float(match.group(1))
            except ValueError:
                continue
    return found


def build_base_params(simu_folder: Path) -> ParameterSet:
    """Starting parameter set: control.txt values, falling back to the defaults.

    Free parameters are clipped into ``PARAM_BOUNDS`` (the optimizers search that
    box); frozen parameters are passed through untouched so the model
    configuration is exactly the one the control file describes.
    """
    values = dict(DEFAULT_PARAMS)
    from_control = read_params_from_control(simu_folder / "control.txt")
    values.update(from_control)
    resolved: Dict[str, float] = {}
    for name in PARAM_BOUNDS:
        raw = float(values.get(name, DEFAULT_PARAMS[name]))
        if name in FROZEN_PARAMETERS:
            resolved[name] = raw
            continue
        resolved[name] = safe_clip(name, raw)
        if abs(resolved[name] - raw) > 1e-12:
            print(f"[WARN] {name}={raw} lies outside {PARAM_BOUNDS[name]}; "
                  f"clipped to {resolved[name]}")
    if from_control:
        print(f"[INFO] Initial parameters read from {simu_folder / 'control.txt'}")
    else:
        print("[WARN] control.txt has no scalar parameters; using hydro_cali_main defaults")
    return ParameterSet(resolved)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Non-LLM optimizer baselines (rule / SCE-UA / CMA-ES / BO / EKI) for EF5",
        fromfile_prefix_chars="@",
    )
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--objective", choices=list(OBJECTIVES), default="nse",
                   help="nse = full-period NSE (identical to INCA's objective); "
                        "composite = NSE penalised by peak timing, peak ratio and event volume")
    p.add_argument("--budget", type=int, default=1000,
                   help="Maximum number of EF5 forward evaluations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--site_num", required=True, help="USGS gauge id, e.g. 02338660")
    p.add_argument("--simu_folder", required=True,
                   help="Experiment folder containing control.txt")
    p.add_argument("--ef5_executable", default="./EF5/bin/ef5")
    p.add_argument("--max_workers", type=int, default=8,
                   help="Number of EF5 runs executed in parallel per batch")
    p.add_argument("--n_candidates", type=int, default=8,
                   help="Candidates per round (rule / random_local)")
    p.add_argument("--failure_patient", type=int, default=5,
                   help="Early stop after this many non-improving rounds (rule / random_local)")
    p.add_argument("--fill_budget", action="store_true",
                   help="rule / random_local: after the early stop, spend the remaining budget "
                        "on random local restarts from the incumbent so that an equal-budget "
                        "reading exists (payload records n_eval_at_stop). Default: off")
    p.add_argument("--n_peaks", type=int, default=3,
                   help="Number of peak events used for the event metrics")
    p.add_argument("--pop_size", type=int, default=None,
                   help="Population/ensemble size (cmaes, eki; also the SCE-UA complex size "
                        "via pop_size/complexes and the BO batch size when given). Default: automatic")
    # SCE-UA (Duan et al. 1994 defaults: p=2, m=2n+1, q=n+1, beta=2n+1)
    p.add_argument("--complexes", type=int, default=None, help="SCE-UA: number of complexes p")
    p.add_argument("--complex_size", type=int, default=None, help="SCE-UA: points per complex m")
    p.add_argument("--subset_size", type=int, default=None, help="SCE-UA: sub-complex size q")
    p.add_argument("--evolutions_per_complex", type=int, default=None,
                   help="SCE-UA: CCE evolution steps per complex before shuffling (beta)")
    # EKI / CMA-ES hyper-parameters (for sensitivity sweeps)
    p.add_argument("--eki_observation", choices=["residual", "scalar", "series"], default="residual",
                   help="EKI observation operator: standardised diagnostics + (1-NSE) [residual], "
                        "the single term 1-NSE [scalar], or the full daily-mean residual series "
                        "normalised so that its squared norm is 1 - mean NSE over gauges [series]")
    p.add_argument("--lumped_budget", type=int, default=None,
                   help="blk_lump_eki: evaluations spent on the lumped stage before refining by zone (default 200)")
    p.add_argument("--ensemble_size", type=int, default=None,
                   help="EKI ensemble size J (overrides --pop_size for eki)")
    p.add_argument("--sigma0", type=float, default=None,
                   help="CMA-ES initial step size in the unit cube (default 0.30)")
    p.add_argument("--out", default=None,
                   help="History JSON path (default: <simu_folder>/results/optimizer_history.json)")
    # zoned (high-dimensional) configuration: outlet + k virtual interior gauges, one
    # parameter block each -> dim = 10 * (k + 1). See hydrocalib/zoned.py.
    p.add_argument("--dream_likelihood", choices=["ef5_sse", "nse_half"],
                   default="ef5_sse",
                   help="DREAM log-density. ef5_sse reproduces EF5's own acceptance rule "
                        "log L = -(MaxT/2)*log(1-NSE) (dream_functions.cpp:616-631, Gamma=0). "
                        "nse_half is the flat NSE/2 form EF5 only stores as a diagnostic column; "
                        "kept as a contrast arm.")
    p.add_argument("--zones", type=int, default=0,
                   help="k virtual interior gauges (0 = legacy single-gauge run)")
    p.add_argument("--delin", default=None,
                   help="delineate_<gauge>.json with the ranked tributary outlets (required with --zones > 0)")
    p.add_argument("--real_gauges", default=None,
                   help=("nested_<gauge>.json from nest_real_gauges.py: use screened real USGS "
                         "gauges, with their own records, instead of virtual ones"))
    p.add_argument("--obs_dir", default="data_cali/gauge_18_19",
                   help="where USGS_<id>_1h_UTC.csv live, for --real_gauges")
    p.add_argument("--lumped", action="store_true",
                   help=("keep every gauge (so every record is scored) but give a parameter "
                         "block only to the outlet: the 10-parameter control for the "
                         "multi-site objective. Same as --param_zones 1."))
    p.add_argument("--param_zones", type=int, default=None,
                   help=("how many gauges get their own parameter block, outlet first; the "
                         "rest are still simulated and scored but inherit their downstream "
                         "neighbour's parameters. Use this, not a smaller --zones, to vary "
                         "the parameter dimension at a FIXED set of observations -- "
                         "--zones k drops the other gauges from the control entirely, so the "
                         "objective would be averaged over a different set of records."))
    p.add_argument("--multisite", default=None,
                   help=("comma-separated gauge ids to score with --objective multisite_nse; "
                         "defaults to the zoned control's own gauges. Give it explicitly with "
                         "--zones 0 to score a lumped parameter set against every record."))
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    simu_folder = Path(args.simu_folder).resolve()
    if not simu_folder.exists():
        raise FileNotFoundError(f"simu_folder not found: {simu_folder}")

    out_path = Path(args.out) if args.out else simu_folder / "results" / "optimizer_history.json"
    base_params = build_base_params(simu_folder)
    space = SearchSpace()
    zone_ids: list = []
    all_gauges: list = []
    if args.zones > 0:
        # Rewrite <simu_folder>/control.txt as outlet + k virtual gauges with one full
        # parameter block each (the single-gauge original is kept as control.single.txt),
        # replicate the starting point into every zone, and search the zoned names.
        if not (args.delin or args.real_gauges):
            raise SystemExit("--zones > 0 needs --delin delineate_<gauge>.json or --real_gauges nested_<gauge>.json")
        from hydrocalib.zoned import build_control, expand_base, zoned_free_names  # noqa: E402
        ctrl = simu_folder / "control.txt"
        single = simu_folder / "control.single.txt"
        if not single.exists():
            single.write_text(ctrl.read_text())
        if args.real_gauges:
            # Real interior gauges: each keeps its USGS id and its own record, so the
            # zone parameters become identifiable instead of being constrained only
            # through the outlet. nest_real_gauges.py already screened and snapped them.
            nested = json.loads(Path(args.real_gauges).read_text())
            picks = list(nested["nested"])[:args.zones]
            obs_dir = Path(args.obs_dir).resolve()
            for c in picks:
                obs = obs_dir / f"USGS_{c['site_no']}_1h_UTC.csv"
                if not obs.exists():
                    raise SystemExit(f"missing observations for interior gauge {c['site_no']}: {obs}")
                c["obs"] = str(obs)
            candidates = picks
        else:
            candidates = json.loads(Path(args.delin).read_text())["candidates"]
        all_gauges = [args.site_num] + [str(c.get("site_no") or f"V{i + 1}")
                                        for i, c in enumerate(candidates[:args.zones])]
        if len(all_gauges) != args.zones + 1:
            raise SystemExit(f"only {len(all_gauges) - 1} interior gauges available, asked for {args.zones}")
        # --lumped keeps every gauge (so every record is still scored) but gives a
        # parameter block to the outlet alone, so the search stays 10-dimensional. That
        # is the control the high-dimensional arms have to beat: if 10 parameters already
        # fit all k+1 records, the extra 10k are not earning their keep.
        n_param_zones = 1 if args.lumped else args.param_zones
        if n_param_zones is not None and not 1 <= n_param_zones <= len(all_gauges):
            raise SystemExit(f"--param_zones must be between 1 and {len(all_gauges)}")
        text, zone_ids = build_control(single.read_text(), candidates, args.site_num, args.zones,
                                       param_zones=(all_gauges[:n_param_zones]
                                                    if n_param_zones else None))
        ctrl.write_text(text)
        base_params = ParameterSet(expand_base(base_params.values, zone_ids))
        space = SearchSpace(names=zoned_free_names(zone_ids))
        print(f"[zoned] {len(zone_ids)} zones {zone_ids} -> dim {space.dim}; control rewritten at {ctrl}")

    runner = SimulationRunner(
        simu_folder=str(simu_folder),
        ef5_executable=args.ef5_executable,
        gauge_num=args.site_num,
    )
    evaluator = EF5Evaluator(
        runner=runner,
        method=args.method,
        objective=args.objective,
        budget=args.budget,
        n_peaks=args.n_peaks,
        max_workers=args.max_workers,
        peak_pick_kwargs=DEFAULT_PEAK_PICK_KWARGS,
        need_signatures=(args.method in ("rule", "eki") or args.objective == "composite"),
        # Record NSE at every gauge whenever the control has more than one, whatever the
        # objective is. It costs one CSV read per gauge per evaluation (~3% of an EF5 run)
        # and it is what lets an outlet-only solution be asked how it does at the interior
        # gauges it never saw -- the diagnostic, not just the score.
        multisite_gauges=(args.multisite.split(",") if args.multisite else (all_gauges or None)),
    )
    if args.objective == "multisite_nse" and not (args.multisite or all_gauges):
        raise SystemExit("--objective multisite_nse needs --zones > 0 or an explicit --multisite list")

    print("=== Optimizer baseline ===")
    print(f"  method        : {args.method}")
    print(f"  objective     : {args.objective}")
    print(f"  budget        : {args.budget} EF5 evaluations")
    print(f"  seed          : {args.seed}")
    print(f"  gauge         : {args.site_num}")
    print(f"  simu_folder   : {simu_folder}")
    print(f"  ef5_executable: {runner.ef5_executable}")
    print(f"  max_workers   : {args.max_workers}")
    print(f"  rng stream    : default_rng([{args.seed}, METHOD_IDS['{args.method}']])")
    if args.method in ("rule", "random_local"):
        print(f"  fill_budget   : {args.fill_budget}")
    if args.method == "eki":
        print(f"  eki_observation: {args.eki_observation}  ensemble_size: "
              f"{args.ensemble_size or args.pop_size or 'default'}")
    if args.method == "dream":
        print(f"  likelihood    : {args.dream_likelihood}  chains: {args.ensemble_size or args.pop_size or 'default(3)'}")
    if args.method == "cmaes":
        print(f"  sigma0        : {args.sigma0 or 'default'}")
    if args.method == "sceua":
        print(f"  sceua p/m/q/beta: {args.complexes or 'auto'}/{args.complex_size or 'auto'}/"
              f"{args.subset_size or 'auto'}/{args.evolutions_per_complex or 'auto'}")
    print(f"  free params   : {', '.join(space.names)}")
    print(f"  frozen params : {', '.join(sorted(FROZEN_PARAMETERS))}")
    print(f"  start params  : {json.dumps(base_params.values)}")
    print(f"  history out   : {out_path}")

    # a multi-hour run should survive a crash: dump a partial history after
    # every batch, and delete it once the final history has been written
    partial_path = out_path.with_name(f"{out_path.stem}.partial{out_path.suffix or '.json'}")
    evaluator.on_batch = lambda ev: save_history(
        build_checkpoint(ev, args.method, args.seed, args.site_num), partial_path)

    payload = run_optimization(
        evaluator,
        method=args.method,
        base_params=base_params,
        seed=args.seed,
        gauge=args.site_num,
        n_candidates=args.n_candidates,
        failure_patient=args.failure_patient,
        pop_size=args.pop_size,
        max_workers=args.max_workers,
        space=space,
        extra_config={"simu_folder": str(simu_folder), "n_peaks": args.n_peaks,
                      "max_workers": args.max_workers,
                      "zones": args.zones, "zone_ids": zone_ids, "delin": args.delin},
        fill_budget=args.fill_budget,
        eki_observation=args.eki_observation,
        dream_likelihood=args.dream_likelihood,
        ensemble_size=args.ensemble_size,
        sigma0=args.sigma0,
        complexes=args.complexes,
        complex_size=args.complex_size,
        subset_size=args.subset_size,
        evolutions_per_complex=args.evolutions_per_complex,
        lumped_budget=args.lumped_budget,
    )
    save_history(payload, out_path)
    try:
        partial_path.unlink(missing_ok=True)
    except OSError:
        pass

    best = payload.get("best")
    print("=== Finished ===")
    print(f"  evaluations : {payload['n_evaluations']} (failed: {payload['n_failed']})")
    print(f"  stopped_early: {payload['stopped_early']} {payload['stop_reason']}")
    if payload.get("n_eval_at_stop") is not None:
        print(f"  n_eval_at_stop: {payload['n_eval_at_stop']}")
    if best:
        value = best.get("objective_value")
        shown = f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else "n/a"
        print(f"  best objective ({args.objective}) = {shown} "
              f"at eval_index {best['eval_index']}")
        print(f"  best params : {json.dumps(best['params'])}")
        print(f"  best metrics: {json.dumps(best['full_metrics'])}")
    best_nse = payload.get("best_by_nse")
    if best_nse and args.objective != "nse":
        nse = (best_nse.get("full_metrics") or {}).get("NSE")
        print(f"  best by NSE  = {nse} at eval_index {best_nse['eval_index']}")
    prov = payload.get("provenance") or {}
    print(f"  provenance  : git={prov.get('git_head')} optimizers.py="
          f"{(prov.get('optimizers_py_sha256') or '')[:12]} rule_table="
          f"{(prov.get('rule_table_sha256') or '')[:12]} guide_ok={prov.get('guide_provenance_ok')}")
    print(f"  history     : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
