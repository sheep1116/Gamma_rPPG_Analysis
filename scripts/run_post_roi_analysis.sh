#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/post_roi_analysis.yaml}"
STAGE="${2:-all}"
MODE="${3:---resume}"
run_logged "$CONFIG" post_roi_analyse "post_roi_${STAGE}_${MODE#--}" \
  python -m gamma_rppg.experiments.post_roi --config "$CONFIG" --stage "$STAGE" \
  --datasets "$DATASETS_ALL" --methods "$METHODS_ALL" "$MODE"
