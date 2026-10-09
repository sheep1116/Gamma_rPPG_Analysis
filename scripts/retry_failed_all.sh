#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/full_pre_roi_experiment.yaml}"
run_logged "$CONFIG" pre_roi_gamma "pre_roi_retry_errors" \
  python -m gamma_rppg.experiments.pre_roi --config "$CONFIG" --datasets "$DATASETS_ALL" --methods "$METHODS_ALL" --retry-errors
run_logged "$CONFIG" pre_roi_gamma "pre_roi_summary_after_retry" \
  python -m gamma_rppg.analysis.summary --config "$CONFIG" --datasets "$DATASETS_ALL"
