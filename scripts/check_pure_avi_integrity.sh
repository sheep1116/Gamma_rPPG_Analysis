#!/usr/bin/env bash
source "$(dirname "$0")/_env.sh"
CONFIG="${1:-configs/pure_avi_integrity.yaml}"
run_logged "$CONFIG" pure_avi_check "PURE_avi_check" \
  python -m gamma_rppg.diagnostics.avi_integrity --config "$CONFIG"
