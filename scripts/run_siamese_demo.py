"""Run a simple Siamese-style similarity smoke test from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fsod_drone.models import SiameseDemoRunner
from fsod_drone.utils import load_yaml, resolve_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run Siamese demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fsod_drone/configs/siamese_config.yaml"),
        help="Path to the Siamese demo config.",
    )
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    runner = SiameseDemoRunner(
        device=str(config.get("device", "auto")),
        pretrained_backbone=bool(config.get("pretrained_backbone", True)),
    )

    pairs = []
    for pair in config.get("pairs", []):
        pairs.append(
            {
                "name": pair.get("name"),
                "image_a": str(resolve_path(pair["image_a"])),
                "image_b": str(resolve_path(pair["image_b"])),
            }
        )

    if not pairs:
        raise ValueError("Siamese config must define at least one image pair.")

    results, report_path = runner.compare_many(
        pairs=pairs,
        output_dir=resolve_path(config.get("output_dir", "outputs/siamese_demo")),
    )

    print("Siamese demo completed.")
    print(f"Report: {report_path}")
    for item in results:
        print(
            f"{item['pair_name']}: cosine_similarity={item['cosine_similarity']}, "
            f"l2_distance={item['l2_distance']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
