#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/full_pre_roi_experiment.yaml}"
MODE="${2:---resume}"
run_logged "$CONFIG" pre_roi_gamma "all_pre_roi_${MODE#--}" \
  python -m gamma_rppg.experiments.pre_roi --config "$CONFIG" --datasets "$DATASETS_ALL" --methods "$METHODS_ALL" "$MODE"
run_logged "$CONFIG" pre_roi_gamma "all_pre_roi_summary" \
  python -m gamma_rppg.analysis.summary --config "$CONFIG" --datasets "$DATASETS_ALL"
