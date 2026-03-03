"""Simple Ultralytics YOLOE runner for installation smoke tests."""

from __future__ import annotations

from pathlib import Path

from .yolo_model import YOLODemoRunner


class YOLOEDemoRunner(YOLODemoRunner):
    """Load a pretrained YOLOE model and run a basic prediction demo."""

    def __init__(self, weights: str | Path = "yoloe-11s-seg.pt") -> None:
        super().__init__(weights=weights)
