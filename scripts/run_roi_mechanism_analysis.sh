#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
RESULT_ROOT="${1:-}"
OUTPUT_DIR="${2:-}"
CONFIG="${3:-configs/roi_mechanism.yaml}"
if [[ -z "$RESULT_ROOT" ]]; then
  RESULT_ROOT="$(python -m gamma_rppg.analysis.roi_mechanism.cli --config "$CONFIG" --print-result-root)"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(python -m gamma_rppg.analysis.roi_mechanism.cli --config "$CONFIG" --result-root "$RESULT_ROOT" --print-output)"
fi
mkdir -p "$OUTPUT_DIR/logs"
LOG_FILE="$OUTPUT_DIR/logs/roi_gamma_analysis_$(date +%Y%m%d_%H%M%S).log"
PYTHONUNBUFFERED=1 python -m gamma_rppg.analysis.roi_mechanism.cli \
  --config "$CONFIG" --result-root "$RESULT_ROOT" --output "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"
