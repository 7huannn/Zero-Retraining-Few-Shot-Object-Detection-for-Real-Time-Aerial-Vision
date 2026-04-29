"""Check whether the project dependencies are available."""

from __future__ import annotations

import argparse
import platform

from tracker_adapter import available_tracker_backends
from utils import detect_optional_package


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Check demo environment.")
    parser.parse_args(argv)

    print(f"Python: {platform.python_version()}")
    print(f"yaml: {'ok' if detect_optional_package('yaml') else 'missing'}")
    print(f"ultralytics: {'ok' if detect_optional_package('ultralytics') else 'missing'}")
    print(f"torch: {'ok' if detect_optional_package('torch') else 'missing'}")
    print(f"torchvision: {'ok' if detect_optional_package('torchvision') else 'missing'}")
    print(f"opencv-python: {'ok' if detect_optional_package('cv2') else 'missing'}")
    print(f"scikit-learn: {'ok' if detect_optional_package('sklearn') else 'missing'}")
    print(f"matplotlib: {'ok' if detect_optional_package('matplotlib') else 'missing'}")

    if detect_optional_package("ultralytics"):
        try:
            from ultralytics import YOLOE  # noqa: F401

            print("YOLOE: ok")
        except Exception as exc:
            print(f"YOLOE: missing ({exc})")

    tracker_backends = available_tracker_backends()
    print(f"tracker_backends_kcf_policy: {', '.join(tracker_backends) if tracker_backends else 'none'}")
    print(f"kcf_available: {'yes' if any(name in {'kcf', 'legacy_kcf'} for name in tracker_backends) else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
