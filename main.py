"""Top-level CLI for environment checks and simple model demos."""

from __future__ import annotations

import argparse

from scripts.check_env import main as check_env_main
from scripts.run_yoloe_demo import main as run_yoloe_main
from scripts.run_yolo_demo import main as run_yolo_main


def main(argv: list[str] | None = None) -> int:
    """Dispatch package subcommands without duplicating parser logic."""
    parser = argparse.ArgumentParser(description="Simple demo CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["check-env", "run-yolo", "run-yoloe"]:
        subparsers.add_parser(command)

    args, remaining = parser.parse_known_args(argv)

    if args.command == "check-env":
        return check_env_main(remaining)
    if args.command == "run-yolo":
        return run_yolo_main(remaining)
    return run_yoloe_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
