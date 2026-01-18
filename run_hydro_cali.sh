#!/bin/bash
set -euo pipefail

# List of gauge IDs
GAUGE_IDS=(
  02338660
  02472000
)

# Output file for all runs
OUTPUT_FILE="hydro_cali_all_gauges.txt"

# Experiment prefix (applies to all runs)
EXP_PREFIX="exp001"

# Single run configuration per gauge
MODEL_TYPE="noboth"
IMAGE_INPUT_OFF=true
PHYSICS_INFORMATION_OFF=true

# Clear previous output if the file exists
> "$OUTPUT_FILE"

run_case() {
  local gauge_id="$1"
  local case_label="$2"
  shift 2

  echo "[RUN] ${case_label}" >> "$OUTPUT_FILE"
  python hydro_cali_main.py @cali_args.txt \
    --site_num "${gauge_id}" \
    --exp_prefix "${EXP_PREFIX}" \
    "$@" >> "$OUTPUT_FILE" 2>&1
}

for GAUGE_ID in "${GAUGE_IDS[@]}"; do
  echo "==================================================" >> "$OUTPUT_FILE"
  echo "Running hydro calibration for gauge_id=${GAUGE_ID}" >> "$OUTPUT_FILE"
  echo "==================================================" >> "$OUTPUT_FILE"

  if [[ -z "${EXP_PREFIX}" ]]; then
    echo "EXP_PREFIX must be non-empty." >> "$OUTPUT_FILE"
    exit 1
  fi

  case_label="model_type=${MODEL_TYPE}"
  if [[ "${IMAGE_INPUT_OFF}" == true ]]; then
    case_label="${case_label}, image_input_off"
  fi
  if [[ "${PHYSICS_INFORMATION_OFF}" == true ]]; then
    case_label="${case_label}, physics_information_off"
  fi

  args=()
  if [[ "${IMAGE_INPUT_OFF}" == true ]]; then
    args+=(--image_input_off)
  fi
  if [[ "${PHYSICS_INFORMATION_OFF}" == true ]]; then
    args+=(--physics_information_off)
  fi
  args+=(--model_type "${MODEL_TYPE}" --detail_output)

  run_case "${GAUGE_ID}" "${case_label}" "${args[@]}"
  echo -e "\nFinished gauge_id=${GAUGE_ID}\n" >> "$OUTPUT_FILE"
done

echo "All gauge runs finished." >> "$OUTPUT_FILE"
