#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/case_mechanism_figures.yaml}"
PRE_ROI_ROOT="${2:-}"
OUTPUT_ROOT="${3:-}"
DATASETS="${4:-all}"

CMD=(python -m gamma_rppg.figures.cases --config "$CONFIG" --datasets "$DATASETS")
if [[ -n "$PRE_ROI_ROOT" ]]; then
  CMD+=(--pre-roi-root "$PRE_ROI_ROOT")
fi
if [[ -n "$OUTPUT_ROOT" ]]; then
  CMD+=(--output-root "$OUTPUT_ROOT")
fi

"${CMD[@]}"
