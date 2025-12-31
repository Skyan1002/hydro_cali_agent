# hydro_cali_agent

> **AI-powered EF5 calibration companion** – automate data prep, candidate generation, EF5 runs, and diagnostics for rapid watershed experimentation.

## ✨ What you get
- Reproducible scripts to fetch forcing/observation data (`download_data.sh`, `usgs_gauge_download.py`).
- A Python driver (`hydro_cali_main.py`) that assembles EF5 control files, manages results, and coordinates iterative calibration logic.
- Agent utilities (`hydrocalib/agents/*`) that leverage OpenAI models via a `.env` file for proposal generation.

---

## 🚀 Quick start
```bash
# 1) Grab the calibration agent
git clone https://github.com/Skyan1002/hydro_cali_agent.git
cd hydro_cali_agent/

# 2) Download example forcing/obs bundles (2018 to 2019)
chmod +x download_data.sh

./download_data.sh

# OR use docker interactively:
# 1) Build the image
docker build -t hydro-cali .

# 2) Run container
docker run -it --rm -v "$(pwd)/data_cali":/app/data_cali -v "$(pwd)/cali_set":/app/cali_set -v "$(pwd)/.env":/app/.env -v "$(pwd)/cali_args.txt":/app/cali_args.txt hydro-cali /bin/bash

#If you use docker, skip to #5

# 3) Fetch and build EF5 next to this repo
git clone https://github.com/Skyan1002/EF5.git
cd EF5/
autoreconf --force --install
./configure
make
cd ..

# 4) Create your Python environment and install dependencies
python -m venv .venv  # or conda create -n hydro-cali python=3.10
source .venv/bin/activate # or conda activate hydro-cali
pip install -r requirements.txt # or just conda install -c conda-forge pandas numpy matplotlib requests dataretrieval openai python-dotenv

# 5) Provide API keys to the agents
echo "OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>" > .env
```

---

## 🐳 Singularity (HPC)

If you are running on an HPC cluster where Docker is not available, you can use Singularity.

### Option 1: Build Locally & Push to Docker Hub (Recommended)
1. **Local Machine**: Build and push the image to Docker Hub.
   ```bash
   docker login
   # Replace chrimerss with your Docker Hub username
   # Use --platform linux/amd64 to ensure compatibility with most HPC clusters
   docker build --platform linux/amd64 -t chrimerss/hydro-cali:latest .
   docker push chrimerss/hydro-cali:latest
   ```

2. **HPC Cluster**: Pull directly from Docker Hub.
   ```bash
   singularity build hydro-cali.sif docker://chrimerss/hydro-cali:latest
   ```

### Option 2: Build & Convert Locally
If you have Singularity installed on your local machine:
```bash
docker build -t hydro-cali .
singularity build hydro-cali.sif docker-daemon://hydro-cali:latest
# Then transfer hydro-cali.sif to your HPC
```

### Running on HPC
Once you have `hydro-cali.sif` on the cluster:
```bash
singularity run \
  --bind $(pwd)/data_cali:/app/data_cali \
  --bind $(pwd)/.env:/app/.env \
  --bind $(pwd)/cali_args.txt:/app/cali_args.txt \
  hydro-cali.sif python3 hydro_cali_main.py @cali_args.txt --site_num 03284230
```

---

## 🧠 Running the calibration agent
1. Configure `cali_args.txt` with the CLI flags you reuse often (each line contains `--flag value`).
2. Launch the pipeline for a specific gauge:
   ```bash
   python3 hydro_cali_main.py @cali_args.txt --site_num 03284230
   ```
3. Check `cali_set/<site>_<tag>/results/cali_*/cand_*/` for EF5 outputs and `history_round_*.json` for candidate evolution.

### CLI reference
| Flag | Purpose |
| --- | --- |
| `--site_num` | USGS gauge identifier that anchors the calibration and data download workflow. |
| `--basic_data_path` | Directory that must contain `dem_usa.tif`, `fdir_usa.tif`, `facc_usa.tif` for terrain/flow routing inputs. |
| `--default_param_dir` | Folder with `crest_params/` & `kw_params/` grids (e.g., `wm_usa.tif`, `alpha_usa.tif`) used as spatial priors. |
| `--cali_set_dir` | Root folder where site-specific subdirectories (control files, histories, plots) are created. Defaults to `./cali_set`. |
| `--cali_tag` | Text appended to the site folder (`<site>_<tag>`) to distinguish multiple experiments (e.g., `2018`, `stormA`). |
| `--folder_label` | Optional extra suffix after `<site>_<tag>` when creating the calibration folder. Defaults to the creation timestamp `YYYYMMDDHHmm`. |
| `--precip_path`, `--precip_name` | Location and filename pattern for precipitation rasters passed to EF5’s `[PrecipForcing]`. |
| `--pet_path`, `--pet_name` | PET raster location/pattern for `[PETForcing]`. |
| `--gauge_outdir` | Where the script stores downloaded hourly USGS CSV files (`USGS_<id>_1h_UTC.csv`). |
| `--results_outdir` | Target directory recorded in the template control file; EF5 run artifacts end up under `<cali_set>/<site>/<results/...` when managed by the runner. |
| `--time_begin`, `--time_end` | Simulation window in `YYYYMMDDhhmm`. Controls both data download span and EF5 control file. |
| `--warmup_time_begin`, `--warmup_time_end` | Warmup run window (defaults `201710010000` → `201801010000`) written to the `[Task warmup]` block. |
| `--time_step` | EF5 timestep (default `1h`), also forwarded to the USGS downloader. |
| `--model`, `--routing` | EF5 model and routing scheme names written into `[Task Simu]`. Defaults are `CREST` and `KW`. |
| `--wm`, `--b`, `--im`, `--ke`, `--fc`, `--iwu` | Scalar Crest parameter seeds used as the starting point for candidate generation (override raster-derived defaults). |
| `--under`, `--leaki`, `--th`, `--isu`, `--alpha`, `--beta`, `--alpha0` | KW routing parameter seeds manipulated by the agent per round. |
| `--python_exec`, `--usgs_script_path` | Let you specify which Python binary/script should execute `usgs_gauge_download.py`. |
| `--skip_download` | When present, assumes gauge observations already exist and bypasses the download subprocess. |
| `--n_candidates`, `--n_peaks`, `--max_rounds` | Control the calibration loop breadth, number of hydrograph peaks used for scoring, and max rounds of EF5 runs, respectively. |
| `--memory-cutoff` | Limit the number of prior rounds included in LLM prompts to reduce context size. |

> 💡 Use `@cali_args.txt` to keep long flag sets tidy; the argparse configuration already enables `fromfile_prefix_chars='@'`.

---

## 🧩 System design at a glance
```text
hydro_cali_main.py
├─ fetches USGS metadata & observations
├─ materializes EF5 control.txt from templates
└─ orchestrates hydrocalib.simulation.TwoStageCalibrationManager
    ├─ hydrocalib.parameters / config → define tunable parameter spaces & guards
    ├─ hydrocalib.peak_events / metrics → score EF5 hydrographs vs. observations
    ├─ hydrocalib.ef5_runner → renders control files, launches EF5 binaries
    ├─ hydrocalib.agents.* → OpenAI-driven candidate proposal helpers
    └─ hydrocalib.plotting → produce round-by-round diagnostics
```

* `download_data.sh` bundles frequently used forcing/PET datasets for quick experiments.
* `usgs_gauge_download.py` is a standalone script you can reuse outside the agent.
* `.env` secrets are loaded via `python-dotenv` (`hydrocalib/agents/utils.py`) before contacting OpenAI’s APIs.

With these pieces, you can trace every calibration round: from `hydro_cali_main.py` resolving assets ➜ `TwoStageCalibrationManager` generating candidate parameters ➜ `ef5_runner` executing EF5 ➜ `history_round_*.json` documenting results.

---

## 📂 Repository map
| Path | Description |
| --- | --- |
| `hydro_cali_main.py` | CLI entry point tying data download, control templating, and calibration orchestration together. |
| `hydrocalib/` | Library package for calibration logic (agents, EF5 runner, metrics, plotting, simulation managers). |
| `download_data.sh` | Shell helper that fetches sample MRMS/PET/parameter grids into the expected folder layout. |
| `cali_args.txt` | Example argument file demonstrating how to store shared CLI options. |
| `usgs_gauge_download.py` | Utility to pull hourly discharge series from USGS NWIS. |
| `requirements.txt` | Python dependency lock-in for both CLI and agent subsystems. |

Happy calibrating!
