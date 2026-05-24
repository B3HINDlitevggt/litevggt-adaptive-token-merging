#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found at $VENV_DIR"
  echo "Run scripts/setup_ec2.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"

: "${MODEL_PATH:?Set MODEL_PATH=/path/to/te_dict.pt}"
: "${DTU_DIR:?Set DTU_DIR=/path/to/eval_data/dtu}"

python "$ROOT_DIR/eval/eval_DTU.py" \
  --model_path "$MODEL_PATH" \
  --dtu_dir "$DTU_DIR" \
  --seed "${SEED:-0}" \
  ${GA_DEPTH_DIR:+--ga_depth_dir "$GA_DEPTH_DIR"} \
  --ga_edge_weight "${GA_EDGE_WEIGHT:-0.7}" \
  --ga_variance_weight "${GA_VARIANCE_WEIGHT:-0.3}" \
  --ga_depth_boundary_weight "${GA_DEPTH_BOUNDARY_WEIGHT:-0.0}" \
  ${GA_DEPTH_MAP_IS_BOUNDARY:+--ga_depth_map_is_boundary} \
  --ga_interaction_weight "${GA_INTERACTION_WEIGHT:-0.0}" \
  --ga_interaction_mode "${GA_INTERACTION_MODE:-sqrt}" \
  --ga_laplacian_weight "${GA_LAPLACIAN_WEIGHT:-0.0}" \
  ${GA_ADAPTIVE_WEIGHTS:+--ga_adaptive_weights} \
  ${GA_ADAPTIVE_PROTECT_RATIO:+--ga_adaptive_protect_ratio} \
  --ga_protect_base_ratio "${GA_PROTECT_BASE_RATIO:-0.1}" \
  --ga_protect_complexity_lambda "${GA_PROTECT_COMPLEXITY_LAMBDA:-0.0}" \
  --ga_protect_min_ratio "${GA_PROTECT_MIN_RATIO:-0.05}" \
  --ga_protect_max_ratio "${GA_PROTECT_MAX_RATIO:-0.2}" \
  ${GA_PROTECT_NMS:+--ga_protect_nms} \
  --ga_depth_protect_ratio "${GA_DEPTH_PROTECT_RATIO:-0.0}"
