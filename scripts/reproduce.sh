#!/usr/bin/env bash
# Four datasets, three independent seeds; plan is the default and performs no work.
set -euo pipefail

mode="${1:-plan}"
if [[ $# -gt 1 || ( "$mode" != plan && "$mode" != run ) ]]; then
  echo 'Usage: scripts/reproduce.sh [plan|run]' >&2
  exit 2
fi

: "${CUB_ROOT:?Set CUB_ROOT to the official CUB dataset root}"
: "${CARS_ROOT:?Set CARS_ROOT to the official Cars196 dataset root}"
: "${AIRCRAFT_ROOT:?Set AIRCRAFT_ROOT to the official Aircraft dataset root}"
: "${INSHOP_ROOT:?Set INSHOP_ROOT to the official In-Shop dataset root}"
: "${VISION_CHECKPOINT:?Set VISION_CHECKPOINT to local MambaVision weights}"
: "${MAMBA_MODEL:?Set MAMBA_MODEL to the local pretrained decoder directory}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="${PYTHON:-python}"
config="${CONFIG:-configs/paper.json}"
output_runs="${RUN_ROOT:-runs/reproduction-main}"
device="${DEVICE:-cuda:0}"
datasets=(cub cars aircraft inshop)
roots=("$CUB_ROOT" "$CARS_ROOT" "$AIRCRAFT_ROOT" "$INSHOP_ROOT")
seeds=(3407 3408 3409)

command_or_plan() {
  if [[ "$mode" == plan ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$mode" == run && -e "$output_runs" ]]; then
  echo "RUN_ROOT already exists: $output_runs; choose a new output directory." >&2
  exit 2
fi
command_or_plan mkdir -p "$output_runs/manifests"

for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  manifest="$output_runs/manifests/$dataset.json"
  command_or_plan "$python" -m mambagir prepare-data --dataset "$dataset" \
    --root "${roots[$index]}" --output "$manifest"
  for seed in "${seeds[@]}"; do
    run_dir="$output_runs/$dataset/seed$seed"
    command_or_plan "$python" -m mambagir train-tokenizer --config "$config" \
      --manifest "$manifest" --vision-checkpoint "$VISION_CHECKPOINT" \
      --output "$run_dir/tokenizer" --seed "$seed" --device "$device"
    command_or_plan "$python" -m mambagir.reproduction export --manifest "$manifest" \
      --tokenizer "$run_dir/tokenizer/tokenizer.pt" --dataset-root "${roots[$index]}" \
      --output "$run_dir/reproduction"
    command_or_plan "$python" -m mambagir train --config "$config" --manifest "$manifest" \
      --tokenizer "$run_dir/tokenizer/tokenizer.pt" --mamba-model "$MAMBA_MODEL" \
      --output "$run_dir/generator" --seed "$seed" --device "$device"
    command_or_plan "$python" -m mambagir evaluate --checkpoint "$run_dir/generator/last.pt" \
      --manifest "$manifest" --output "$run_dir/evaluation" \
      --beam-size 8 --ks 1 4 8 --device "$device"
  done
  command_or_plan "$python" -m mambagir summarize \
    "$output_runs/$dataset/seed3407/evaluation/metrics.json" \
    "$output_runs/$dataset/seed3408/evaluation/metrics.json" \
    "$output_runs/$dataset/seed3409/evaluation/metrics.json" \
    --output "$output_runs/$dataset/summary.json"
done

for seed in "${seeds[@]}"; do
  command_or_plan "$python" -m mambagir average \
    --cub "$output_runs/cub/seed$seed/evaluation/metrics.json" \
    --cars "$output_runs/cars/seed$seed/evaluation/metrics.json" \
    --aircraft "$output_runs/aircraft/seed$seed/evaluation/metrics.json" \
    --inshop "$output_runs/inshop/seed$seed/evaluation/metrics.json" \
    --output "$output_runs/average_seed$seed.json"
done
command_or_plan "$python" -m mambagir summarize \
  "$output_runs/average_seed3407.json" "$output_runs/average_seed3408.json" \
  "$output_runs/average_seed3409.json" --output "$output_runs/average_summary.json"
