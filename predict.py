"""Unified CLI for environment setup, preprocessing, YOLOE demos, and Siamese training."""

from __future__ import annotations

import argparse
import sys


COMMANDS = [
    "check-env",
    "install-yoloe",
    "preprocess-yoloe",
    "run-yolo",
    "run-yoloe",
    "train-siamese",
    "eval-siamese",
    "compare-siamese",
    "run-siamese",
]


def main(argv: list[str] | None = None) -> int:
    """Dispatch top-level subcommands without duplicating parser logic."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Drone few-shot pipeline CLI.")
    parser.add_argument("command", choices=COMMANDS, nargs="?")

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = argv[0]
    if command not in COMMANDS:
        parser.error(f"Unknown command: {command}")
    remaining = argv[1:]

    if command == "check-env":
        from check_env import main as check_env_main

        return check_env_main(remaining)
    if command == "install-yoloe":
        from yoloe.install_yoloe import main as install_yoloe_main

        return install_yoloe_main(remaining)
    if command == "preprocess-yoloe":
        from preprocessing import main as preprocess_main

        return preprocess_main(remaining)
    if command == "run-yolo":
        from yoloe.run_yolo_demo import main as run_yolo_main

        return run_yolo_main(remaining)
    if command == "run-yoloe":
        from yoloe.run_yoloe_demo import main as run_yoloe_main

        return run_yoloe_main(remaining)
    if command == "train-siamese":
        from siamese.train import main as train_siamese_main

        return train_siamese_main(remaining)
    if command == "eval-siamese":
        from siamese.evaluate import main as evaluate_siamese_main

        return evaluate_siamese_main(remaining)
    from siamese.run_siamese_demo import main as compare_siamese_main

    return compare_siamese_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
