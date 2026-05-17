"""Compatibility wrapper for the unified inference pipeline."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Forward old `python predict.py ...` commands to `inference.py`."""
    from src.apps.inference import main as inference_main

    raw_args = sys.argv[1:] if argv is None else argv
    print("[predict.py] Forwarding to inference.py (unified matcher pipeline).")
    return inference_main(raw_args)


if __name__ == "__main__":
    raise SystemExit(main())
