#!/usr/bin/env python3
"""CLI wrapper for running the SCE-UA benchmark calibration."""
from hydrocalib.sce_benchmark import build_parser, run_cli


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    history_path = run_cli(args)
    print(f"SCE-UA benchmark complete. History saved to {history_path}")


if __name__ == "__main__":
    main()
