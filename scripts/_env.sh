#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
METHODS_ALL="CHROM,POS,GREEN,LGI,PBV,OMIT"
DATASETS_ALL="UBFC2,LGI_PPGI,PURE"

run_logged() {
  local config="$1"
  local mode="$2"
  local label="$3"
  shift 3
  local result_dir
  result_dir="$(python -m gamma_rppg.common.paths --config "$config" --mode "$mode")"
  local log_dir="$result_dir/logs"
  mkdir -p "$log_dir"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="$log_dir/${label}_${stamp}.log"
  echo "[INFO] Log file: $log_file"
  set -o pipefail
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "$log_file"
}
