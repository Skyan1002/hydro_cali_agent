# INCA: IntelliCal Agent

INCA is a large-language-model agent for diagnosis-guided parameter specification of the spatially distributed CREST hydrologic model (run with EF5). In each round it combines parameter semantics with a diagnosis of simulation–observation mismatches, proposes several hypothesis-driven candidate parameter sets, checks them against a memory of earlier adjustments, and evaluates every candidate with a forward simulation. A candidate is accepted only if it improves the calibration NSE.

This repository contains the code used in *Towards a Diagnostic Reasoning Paradigm for Parameter Specification in Spatially Distributed Hydrologic Models* (Yan et al., under review at *Science Advances*).

**Results page:** https://huggingface.co/spaces/skyan1002/INCA_demo

---

## Repository contents

| Path | Description |
| --- | --- |
| `hydro_cali_main.py` | INCA driver: data preparation, EF5 control files, and the calibration loop. |
| `hydrocalib/agents/` | Staged agent workflow: behaviour perception (`image_reader.py`), candidate proposal (`proposal.py`), verification (`evaluation.py`), orchestration and memory (`manager.py`), parameter semantics (`physics_info.py`), and the LLM interface for OpenAI, Anthropic and Google models (`llm.py`). |
| `hydrocalib/signatures.py` | Numerical hydrologic signatures computed from the simulated and observed streamflow. |
| `hydrocalib/optimizers.py`, `run_optimizer.py` | Calibration baselines on the same EF5 setup: EKI, CMA-ES, Bayesian optimization, SCE-UA, DREAM(ZS) and the rule-based proposer. |
| `hydrocalib/zoned.py` | Zone-wise (multi-gauge) parameterization for the 50- and 100-parameter experiments. |
| `tools/multigauge/` | Screening and grid snapping of the nested USGS gauges for the multi-gauge experiments, and the nested-gauge files used for the Raritan River and Alameda Creek. |
| `hydro_test_2019.py` | Evaluation-period (2019) simulations for INCA and baseline runs. |
| `download_data.sh`, `usgs_gauge_download.py` | Forcing, parameter-grid and streamflow downloads. |
| `cali_args.txt` | Argument file with the settings used in the manuscript. |
| `Dockerfile` | Container with EF5 and the Python dependencies. |

---

## Installation

```bash
git clone https://github.com/Skyan1002/hydro_cali_agent.git
cd hydro_cali_agent/

# Forcing, parameter grids and observations (2018–2019)
chmod +x download_data.sh
./download_data.sh

# EF5 (fork and commit used in the manuscript)
git clone https://github.com/Skyan1002/EF5.git
cd EF5/
git checkout 95dd7032f3fc51df0e600ad5f1061cdadbb1e223
autoreconf --force --install
./configure
make
cd ..

# Python environment
python -m venv .venv && source .venv/bin/activate   # or a conda environment with Python 3.10
pip install -r requirements.txt

# API keys (only the providers you use)
cat > .env <<EOF
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
EOF
```

Alternatively build the container: `docker build -t hydro-cali .` and mount `data_cali/`, `.env` and `cali_args.txt` when running it. The executable path can also be set with the `EF5_EXECUTABLE` environment variable.

---

## Running INCA

```bash
python3 hydro_cali_main.py @cali_args.txt --site_num 02338660
```

`cali_args.txt` holds the settings used in the manuscript: calibration year 2018, k = 8 candidates per round, at most 20 accepted updates, and stopping after five consecutive non-improving rounds. The default language model is `gpt-5-2025-08-07`.

The calibrated values are multipliers applied to the a priori spatially distributed parameter fields, as described in the manuscript Methods; their search bounds are set in `hydrocalib/config.py` (`PARAM_BOUNDS`). TH, IWU and ISU are held fixed.

Frequently used options:

| Option | Purpose |
| --- | --- |
| `--perception image` / `numeric` / `hybrid` | Behaviour perception from hydrograph and flow-duration-curve images (default), from numerical hydrologic signatures computed from the same series, or from both (the signatures plus a reading of the plots made with the signatures in view; staged workflow only). |
| `--image_input_off`, `--physics_information_off` | Ablations: withhold the diagnostic images (together with the event-level metrics), or replace parameter names by neutral labels and remove the parameter guide. |
| `--workflow full` / `direct_matched` | Staged INCA workflow (default) or single-call direct prompting with the same information; add `--direct_keep_failures` to keep the failure notes. A one-shot run uses `--max_rounds 1 --failure_patient 1`. |
| `--llm_model`, `--llm_provider`, `--thinking_budget` | Other foundation models, e.g. `--llm_model claude-sonnet-4-5-20250929 --thinking_budget 6000`. |
| `--memory-cutoff N` | Number of past rounds kept in the prompts. |
| `--n_candidates`, `--max_rounds`, `--failure_patient` | Candidates per round, maximum accepted updates, and the early-stopping patience. |

### Multi-gauge (zoned) calibration

The 50- and 100-parameter experiments give each nested gauge zone its own copy of the ten parameter multipliers and calibrate the mean NSE over all gauges:

```bash
python3 hydro_cali_main.py @cali_args.txt --site_num <outlet id> \
  --site_lon <lon> --site_lat <lat> --site_area_km2 <area> \
  --zones <k> --real_gauges <nested gauges json> --obs_dir <gauge records> \
  --objective multisite_nse --no_zone_wildcard --bounds_hint --zone_assembly \
  --perception numeric --memory-cutoff 3 --failure_patient 10
```

The nested-gauge file is prepared in two steps: `tools/multigauge/screen_gauges.py` finds and quality-screens the USGS gauges inside the basin, and `tools/multigauge/nest_real_gauges.py` snaps them to the D8 grid and keeps those that drain to the outlet. The files used in the manuscript are `tools/multigauge/nested_01403060.json` (Raritan River, nine interior gauges) and `tools/multigauge/nested_11179000_clean.json` (Alameda Creek, four interior gauges).

`--bounds_hint` tells the agent the parameter bounds and which parameters sit on a bound; `--zone_assembly` adds, in every round, one candidate assembled zone by zone from the best-scoring sources at each zone's own gauge, which is then simulated and competes with the other candidates.

---

## Calibration baselines

```bash
python3 run_optimizer.py --method eki --site_num 02338660 \
  --simu_folder data_cali/cali_set/<experiment folder> --budget 2000 --seed 0
```

`--method` accepts `eki`, `cmaes`, `bo`, `sceua`, `dream`, `rule` (rule-based proposer), `random_local`, and the zone-wise variants used in the multi-gauge experiments. All methods search the same parameter multipliers within the same bounds and maximise the calibration NSE (mean NSE over gauges with `--objective multisite_nse`). The complete evaluation trace is written to `optimizer_history.json`.

INCA simulates the 2019 evaluation period for each accepted update during calibration (switch off with `--disable_test_run`). For baseline runs, the evaluation-period simulation of the best parameter set is run afterwards:

```bash
python3 hydro_test_2019.py --from_optimizer_history <optimizer_history.json>
```

---

## Outputs

Each INCA run writes its results to `<cali_set_dir>/[<exp_prefix>_]<site>_<tag>_<folder_label>/results/`:

* `calibration_history.json`: every round, its proposals, refined candidates, simulated parameter sets and metrics, and the accepted updates;
* `detail_logs/`: the prompts and responses of each agent stage;
* a usage ledger with LLM calls, tokens, cost and timing;
* EF5 outputs of every candidate.

The LLM calls do not set a sampling temperature or seed, so repeated runs differ; the manuscript reports medians across repeated runs.

---

## Citation

Yan, S., Gupta, H. V., et al. Towards a Diagnostic Reasoning Paradigm for Parameter Specification in Spatially Distributed Hydrologic Models. *Science Advances* (under review).
