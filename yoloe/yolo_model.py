"""Simple Ultralytics YOLOv8 runner for installation smoke tests."""

from __future__ import annotations

from pathlib import Path

from utils import ensure_dir


class YOLODemoRunner:
    """Load a pretrained YOLOv8 model and run a basic prediction demo."""

    def __init__(self, weights: str | Path = "models/yolov8n.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install it with `pip install ultralytics`."
            ) from exc

        self.weights = str(weights)
        self._api = YOLO(self.weights)

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        device: str | int | None = None,
        imgsz: int = 640,
        output_dir: str | Path = "result/yolo_demo",
        save: bool = True,
    ) -> list[str]:
        """Run prediction on an image, folder, or video path."""
        ensure_dir(output_dir)
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
