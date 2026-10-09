#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/full_pre_roi_experiment.yaml}"
DATASETS="${2:-$DATASETS_ALL}"
run_logged "$CONFIG" pre_roi_gamma "pre_roi_summary" \
  python -m gamma_rppg.analysis.summary --config "$CONFIG" --datasets "$DATASETS"
