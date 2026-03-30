"""Check whether the basic demo dependencies are available."""

from __future__ import annotations

import argparse
import platform
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fsod_drone.utils import detect_optional_package


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check demo environment.")
    parser.parse_args(argv)

    print(f"Python: {platform.python_version()}")
    print(f"yaml: {'ok' if detect_optional_package('yaml') else 'missing'}")
    print(f"ultralytics: {'ok' if detect_optional_package('ultralytics') else 'missing'}")
    print(f"torch: {'ok' if detect_optional_package('torch') else 'missing'}")
    print(f"torchvision: {'ok' if detect_optional_package('torchvision') else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
