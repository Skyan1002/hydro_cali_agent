# hydro_cali_agent

> **AI-powered EF5 calibration companion** – automate data prep, candidate generation, EF5 runs, and diagnostics for rapid watershed experimentation.

## ✨ What you get
- Reproducible scripts to fetch forcing/observation data (`download_data.sh`, `usgs_gauge_download.py`).
- A Python driver (`hydro_cali_main.py`) that assembles EF5 control files, manages results, and coordinates iterative calibration logic.
- Agent utilities (`hydrocalib/agents/*`) that leverage LLM models (Gemini-3-Pro default) via a `.env` file for proposal generation.

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
echo "GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>" >> .env # Required for Gemini models
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

On docker: 

```bash
docker run --rm \
  -v "$(pwd)/data_cali:/app/data_cali" \
  -v "$(pwd)/cali_set:/app/cali_set" \
  -v "$(pwd)/.env:/app/.env" \
  -v "$(pwd)/cali_args.txt:/app/cali_args.txt" \
  hydro-cali \
  python3 hydro_cali_main.py @cali_args.txt
```

---

## 🧠 Running the calibration agent
1. Configure defaults in `conf/config.yaml`.
2. Launch the pipeline for a specific gauge (overriding defaults via CLI):
   ```bash
   python3 hydro_cali_main.py site_num=03284230
   ```
3. Check `cali_set/<site>_<tag>/results/cali_*/cand_*/` for EF5 outputs and `history_round_*.json` for candidate evolution.

### CLI reference (Hydra)
Use `key=value` syntax to override `conf/config.yaml`.
Examples:
- `site_num="03284230"`: Target USGS gauge.
- `basic_data_path="/path/to/data"`: Override basic data path.
- `llm.model="gpt-4o"`: Switch LLM model.
- `llm.api_key="sk-..."`: Provide API key directly (or set `GOOGLE_API_KEY`/`OPENAI_API_KEY` env var).

See `conf/config.yaml` for all available options.

> 💡 Hydra supports robust configuration composition. See [Hydra documentation](https://hydra.cc/docs/intro/) for more.

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
    ├─ hydrocalib.agents.* → LLM-driven candidate proposal helpers
    └─ hydrocalib.plotting → produce round-by-round diagnostics
```

* `download_data.sh` bundles frequently used forcing/PET datasets for quick experiments.
* `usgs_gauge_download.py` is a standalone script you can reuse outside the agent.
* `.env` secrets are loaded via `python-dotenv` (`hydrocalib/agents/utils.py`) before contacting LLM APIs.

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
