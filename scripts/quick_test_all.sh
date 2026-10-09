#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/quick_pre_roi_test.yaml}"
run_logged "$CONFIG" pre_roi_gamma "quick_pre_roi" \
  python -m gamma_rppg.experiments.quick_test --config "$CONFIG" --datasets PURE --methods "CHROM,POS"
