#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {WISDM|FD|Boiler} {SHOT|AaD|NRC} SOURCE_CHECKPOINT_ROOT [SAVE_DIR] [DEVICE] [NUM_RUNS]" >&2
}

if [[ $# -lt 3 || $# -gt 6 ]]; then
  usage
  exit 2
fi

dataset=$1
method=$2
source_root=$3
save_dir=${4:-MAPU_SFDA_TS/runs_fal}
device=${5:-cuda}
num_runs=${6:-3}

case "$dataset" in WISDM|FD|Boiler) ;; *) usage; exit 2 ;; esac
case "$method" in SHOT|AaD|NRC) ;; *) usage; exit 2 ;; esac
if [[ ! -d "$source_root" ]]; then
  echo "Source checkpoint directory not found: $source_root" >&2
  exit 2
fi

run_description="${method}_FAL"
python trainers/train.py \
  --dataset "$dataset" \
  --da_method "$method" \
  --num_runs "$num_runs" \
  --device "$device" \
  --save_dir "$save_dir" \
  --run_description "$run_description" \
  --source_checkpoint_root "$source_root" \
  --add_fal

python trainers/test.py \
  --dataset "$dataset" \
  --da_method "$method" \
  --num_runs "$num_runs" \
  --device "$device" \
  --save_dir "$save_dir" \
  --run_description "$run_description" \
  --add_fal
