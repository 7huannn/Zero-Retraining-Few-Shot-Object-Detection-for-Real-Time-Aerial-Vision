"""Simple Ultralytics YOLOE runner for installation smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from utils import ensure_dir


class YOLOEDemoRunner:
    """Load a pretrained YOLOE model and run a basic prediction demo."""

    def __init__(self, weights: str | Path = "models/yoloe-11s-seg.pt") -> None:
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install it with `pip install ultralytics`."
            ) from exc

        self.weights = str(weights)
        self._api = YOLOE(self.weights)

    def predict(
        self,
        source: str | Path,
        classes: Sequence[str],
        conf: float = 0.25,
        device: str | int | None = None,
        imgsz: int = 640,
        output_dir: str | Path = "result/yoloe_demo",
        save: bool = True,
    ) -> list[str]:
        """Run YOLOE prediction after defining the text prompts to detect."""
        if not classes:
            raise ValueError("YOLOE requires at least one class prompt before prediction.")

        ensure_dir(output_dir)
        try:
            self._api.set_classes(list(classes))
        except Exception as exc:
            raise RuntimeError(
                "YOLOE needs text-prompt support to run. Ensure the CLIP dependency is installed "
                "and allow the first run to download the MobileCLIP weights."
            ) from exc

        results = self._api.predict(
            source=source,
            conf=conf,
            device=device,
            imgsz=imgsz,
            save=save,
            project=str(output_dir),
            name="predict",
            exist_ok=True,
            verbose=False,
        )
        output_paths: list[str] = []
        for result in results:
            output_paths.append(str(result.path))
        return output_paths
