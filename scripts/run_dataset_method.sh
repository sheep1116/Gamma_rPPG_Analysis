#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
DATASET="${1:?dataset required}"
METHODS="${2:?comma-separated methods required}"
CONFIG="${3:-configs/full_pre_roi_experiment.yaml}"
MODE="${4:---resume}"
run_logged "$CONFIG" pre_roi_gamma "${DATASET}_${METHODS//,/_}_${MODE#--}" \
  python -m gamma_rppg.experiments.pre_roi --config "$CONFIG" --datasets "$DATASET" --methods "$METHODS" "$MODE"
