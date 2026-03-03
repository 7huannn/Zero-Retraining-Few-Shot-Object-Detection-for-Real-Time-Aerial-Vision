"""Run a simple YOLOE smoke test from the YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fsod_drone.models import YOLOEDemoRunner
from fsod_drone.utils import load_yaml, resolve_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run YOLOE demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fsod_drone/configs/yoloe_config.yaml"),
        help="Path to the YOLOE demo config.",
    )
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    runner = YOLOEDemoRunner(weights=config.get("weights", "yoloe-11s-seg.pt"))
    outputs = runner.predict(
        source=resolve_path(config["source"]),
        conf=float(config.get("conf", 0.25)),
        device=config.get("device"),
        imgsz=int(config.get("imgsz", 960)),
        output_dir=resolve_path(config.get("output_dir", "outputs/yoloe_demo")),
    )

    print("YOLOE demo completed.")
    for item in outputs:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
