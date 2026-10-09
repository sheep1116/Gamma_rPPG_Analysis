#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
python -m gamma_rppg.diagnostics.pure_structure --config "${1:-configs/full_pre_roi_experiment.yaml}"
