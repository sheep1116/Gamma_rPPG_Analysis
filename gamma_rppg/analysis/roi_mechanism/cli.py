#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from .io import resolve_result_root
from gamma_rppg.common.config import load_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "roi_mechanism.yaml"
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = Path(__file__).with_name("default.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse why ROI pixel summaries are associated with positive Gamma responses in gamma_rppg_project_v5."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--result-root", default=None, help="Path to GammaBrightness_rPPG_v5 result root.")
    parser.add_argument("--output", default=None, help="Optional output directory. Default: <result-root>/roi_gamma_mechanism_analysis")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    paths = parser.add_mutually_exclusive_group()
    paths.add_argument("--print-result-root", action="store_true", help="Resolve input path without running analysis.")
    paths.add_argument("--print-output", action="store_true", help="Resolve output path without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.print_result_root or args.print_output:
            root = resolve_result_root(args.result_root or config["result_root"])
            output = Path(args.output).expanduser().resolve() if args.output else root / config.get("output_subdir", "roi_gamma_mechanism_analysis")
            print(root if args.print_result_root else output)
            return 0
        from .analysis import run_analysis
        output = run_analysis(config, args.result_root, args.output)
        print(f"Analysis output: {output}")
        return 0
    except Exception:
        logging.exception("ROI-Gamma mechanism analysis failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
