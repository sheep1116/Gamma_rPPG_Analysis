#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/sensitivity_analysis.yaml}"
DATASETS="${2:-all}"
run_logged "$CONFIG" pre_roi_gamma "pre_roi_sensitivity" \
  python -m gamma_rppg.analysis.sensitivity --config "$CONFIG" --datasets "$DATASETS"
