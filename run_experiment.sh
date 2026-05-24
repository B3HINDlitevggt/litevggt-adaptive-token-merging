#!/usr/bin/env bash
# run_experiment.sh
# Usage:
#   ./run_experiment.sh --aggregator sobel --mode baseline --img_dir ./data/test
#   ./run_experiment.sh --aggregator cls   --mode sttm    --img_dir ./data/test --gt_path ./gt.ply
#
# VGGT_AGGREGATOR is set automatically from --aggregator; no need to export it manually.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
AGGREGATOR="sobel"
MODE="baseline"
IMG_DIR=""
OUTPUT_BASE="./output"
GT_PATH=""
BASELINE_DIR=""
CKPT_PATH="./te_dict.pt"
DEVICE="cuda:0"
KEEP_RATIO="0.42"
STTM_SPATIAL="0.8"
STTM_TEMPORAL="0.6"
MAX_FRAMES=""
NO_VERBOSE=""

# ── Argument parsing ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --aggregator)       AGGREGATOR="$2";      shift 2 ;;
    --mode)             MODE="$2";            shift 2 ;;
    --img_dir)          IMG_DIR="$2";         shift 2 ;;
    --output_base)      OUTPUT_BASE="$2";     shift 2 ;;
    --gt_path)          GT_PATH="$2";         shift 2 ;;
    --baseline_dir)     BASELINE_DIR="$2";    shift 2 ;;
    --ckpt_path)        CKPT_PATH="$2";       shift 2 ;;
    --device)           DEVICE="$2";          shift 2 ;;
    --keep_ratio)       KEEP_RATIO="$2";      shift 2 ;;
    --sttm_spatial)     STTM_SPATIAL="$2";    shift 2 ;;
    --sttm_temporal)    STTM_TEMPORAL="$2";   shift 2 ;;
    --max_frames)       MAX_FRAMES="$2";      shift 2 ;;
    --no_verbose)       NO_VERBOSE="--no_verbose"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Validate ───────────────────────────────────────────────────────────
if [[ -z "$IMG_DIR" ]]; then
  echo "Error: --img_dir is required."
  exit 1
fi

if [[ "$AGGREGATOR" != "sobel" && "$AGGREGATOR" != "cls" ]]; then
  echo "Error: --aggregator must be 'sobel' or 'cls'."
  exit 1
fi

VALID_MODES="baseline dynamic_protect dynamic_grid dynamic_all sttm"
if [[ ! " $VALID_MODES " =~ " $MODE " ]]; then
  echo "Error: --mode must be one of: $VALID_MODES"
  exit 1
fi

# ── Setup ──────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_BASE}/${AGGREGATOR}/${MODE}"
export VGGT_AGGREGATOR="$AGGREGATOR"

echo ""
echo "============================================================"
echo "  Aggregator : $AGGREGATOR"
echo "  Mode       : $MODE"
echo "  Output     : $OUTPUT_DIR"
echo "============================================================"
echo ""

# ── Build python args ──────────────────────────────────────────────────
PY_ARGS=(
  --img_dir      "$IMG_DIR"
  --output_dir   "$OUTPUT_DIR"
  --ckpt_path    "$CKPT_PATH"
  --device       "$DEVICE"
  --keep_ratio   "$KEEP_RATIO"
  --mode         "$MODE"
  --sttm_spatial_thresh  "$STTM_SPATIAL"
  --sttm_temporal_thresh "$STTM_TEMPORAL"
)

[[ -n "$GT_PATH"      ]] && PY_ARGS+=(--gt_path      "$GT_PATH")
[[ -n "$BASELINE_DIR" ]] && PY_ARGS+=(--baseline_dir "$BASELINE_DIR")
[[ -n "$MAX_FRAMES"   ]] && PY_ARGS+=(--max_frames   "$MAX_FRAMES")
[[ -n "$NO_VERBOSE"   ]] && PY_ARGS+=("$NO_VERBOSE")

# ── Run ────────────────────────────────────────────────────────────────
python3 run_experiment.py "${PY_ARGS[@]}"
