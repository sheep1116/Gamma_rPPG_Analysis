from __future__ import annotations

import argparse

from gamma_rppg.common.config import load_config, resolve_mode_root, resolve_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a configured v5 result path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--project", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    path = resolve_project_root(config) if args.project else resolve_mode_root(config, args.mode)
    print(path)


if __name__ == "__main__":
    main()
