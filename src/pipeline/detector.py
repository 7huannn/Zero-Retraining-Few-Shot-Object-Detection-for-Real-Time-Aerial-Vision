"""YOLOE detector with visual prompt engineering for few-shot detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLOE


class YOLOEDetector:
    """YOLOE detection module with visual prompt support."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.001,
        top_k_proposals: int = 24,
        device: str | None = None,
    ) -> None:
        """
        Initialize YOLOE detector.

        Args:
            model_path: Path to YOLOE model weights
            conf_threshold: Detection confidence threshold
            top_k_proposals: Maximum number of candidate proposals per frame
            device: torch device (default: auto-detect)
        """
        self.conf_threshold = conf_threshold
        self.top_k_proposals = top_k_proposals
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLOE(model_path)
        self.model.predictor = None
        self.class_name: str | None = None
        self.initial_vpe: torch.Tensor | None = None

    def set_visual_prompt(self, class_name: str, initial_vpe: torch.Tensor) -> None:
        """
        Configure detector with visual prompt for a specific class.

        Args:
            class_name: Name of the target object class
            initial_vpe: Visual prompt embedding tensor for the class
        """
        self.class_name = class_name
        self.initial_vpe = initial_vpe
        self.model.set_classes([class_name], initial_vpe)
        self.model.predictor = None

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """
        Run detection on a single frame.

        Args:
            frame: Input image (BGR, numpy array)

        Returns:
            List of detections with keys: xyxy, xywh, conf, mask
        """
        if self.initial_vpe is None:
            raise RuntimeError("Visual prompt not set. Call set_visual_prompt() first.")

        results = self.model(
            frame,
            save=False,
            conf=self.conf_threshold,
            verbose=False,
            retina_masks=True,
        )

        result = results[0]
        detections: list[dict[str, Any]] = []

        if not result.boxes:
            return detections

        has_masks = result.masks is not None and result.masks.data is not None
        for idx in range(len(result.boxes)):
            xyxy = tuple(map(int, result.boxes.xyxy[idx].cpu().numpy()))
            x1, y1, x2, y2 = xyxy
            detections.append(
                {
                    "xyxy": xyxy,
                    "xywh": (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
                    "conf": float(result.boxes.conf[idx].cpu().numpy()),
                    "mask": result.masks.data[idx].cpu().numpy() if has_masks else None,
                }
            )

        detections.sort(key=lambda item: item["conf"], reverse=True)
        return detections[: self.top_k_proposals]
