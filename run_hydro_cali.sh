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
EXP_PREFIX="batch_exp_0116"

# Clear previous output if the file exists
> "$OUTPUT_FILE"

run_case() {
  local gauge_id="$1"
  local case_id="$2"
  local case_label="$3"
  shift 3

  echo "[${case_id}] ${case_label}" >> "$OUTPUT_FILE"
  python hydro_cali_main.py @cali_args.txt \
    --site_num "${gauge_id}" \
    --exp_prefix "${EXP_PREFIX}" \
    "$@" >> "$OUTPUT_FILE" 2>&1
}

for GAUGE_ID in "${GAUGE_IDS[@]}"; do
  echo "==================================================" >> "$OUTPUT_FILE"
  echo "Running hydro calibration for gauge_id=${GAUGE_ID}" >> "$OUTPUT_FILE"
  echo "==================================================" >> "$OUTPUT_FILE"

  run_case "${GAUGE_ID}" 1 "image_input_off + physics_information_off" \
    --image_input_off \
    --physics_information_off \
    --model_type noboth \
    --detail_output &

  run_case "${GAUGE_ID}" 2 "image_input_off" \
    --image_input_off \
    --model_type noimage \
    --detail_output &

  run_case "${GAUGE_ID}" 3 "physics_information_off" \
    --physics_information_off \
    --model_type nophysics \
    --detail_output &

  run_case "${GAUGE_ID}" 4 "baseline (all on)" \
    --model_type base \
    --detail_output &

  wait
  echo -e "\nFinished gauge_id=${GAUGE_ID}\n" >> "$OUTPUT_FILE"
done

echo "All gauge runs finished." >> "$OUTPUT_FILE"
