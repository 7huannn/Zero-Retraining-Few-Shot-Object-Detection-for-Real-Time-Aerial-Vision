"""Tracker adapter with KCF-first policy and optional debug fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


TrackerFactory = Callable[[], Any]


@dataclass(frozen=True)
class TrackerBackend:
    name: str
    factory: TrackerFactory


class OpenCVTrackerAdapter:
    """Thin wrapper around OpenCV tracker APIs."""

    def __init__(self, factory: TrackerFactory, backend_name: str) -> None:
        self._factory = factory
        self.backend_name = backend_name
        self._tracker: Any | None = None

    def init(self, frame: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> bool:
        self._tracker = self._factory()
        x, y, w, h = bbox_xywh
        result = self._tracker.init(frame, (int(x), int(y), int(w), int(h)))
        return True if result is None else bool(result)

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        if self._tracker is None:
            return False, (0.0, 0.0, 0.0, 0.0)
        ok, box = self._tracker.update(frame)
        return bool(ok), tuple(float(value) for value in box)


class LKOpticalFlowTracker:
    """Translation+scale tracker using sparse Lucas-Kanade optical flow."""

    backend_name = "lk_optical_flow"

    def __init__(self, max_corners: int = 120, min_points: int = 6) -> None:
        self.max_corners = max_corners
        self.min_points = min_points
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.bbox_xywh = np.zeros(4, dtype=np.float32)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _grid_points(x: int, y: int, w: int, h: int) -> np.ndarray | None:
        if w < 8 or h < 8:
            return None
        xs = np.linspace(x + 1, x + w - 2, 4, dtype=np.float32)
        ys = np.linspace(y + 1, y + h - 2, 4, dtype=np.float32)
        points = np.array([(px, py) for py in ys for px in xs], dtype=np.float32)
        return points.reshape(-1, 1, 2)

    def _detect_points(self, gray: np.ndarray, bbox_xywh: np.ndarray) -> np.ndarray | None:
        x, y, w, h = bbox_xywh.astype(np.int32).tolist()
        frame_h, frame_w = gray.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(2, min(w, frame_w - x))
        h = max(2, min(h, frame_h - y))
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return None

        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=7,
        )
        if points is None:
            return self._grid_points(x, y, w, h)

        points[:, 0, 0] += x
        points[:, 0, 1] += y
        return points.astype(np.float32)

    @staticmethod
    def _points_inside_bbox(points: np.ndarray, bbox_xywh: np.ndarray) -> np.ndarray:
        x, y, w, h = bbox_xywh.tolist()
        x2 = x + w
        y2 = y + h
        pts = points.reshape(-1, 2)
        mask = (
            (pts[:, 0] >= x)
            & (pts[:, 0] <= x2)
            & (pts[:, 1] >= y)
            & (pts[:, 1] <= y2)
        )
        inside = pts[mask]
        return inside.reshape(-1, 1, 2).astype(np.float32)

    def init(self, frame: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            return False
        self.bbox_xywh = np.array([x, y, w, h], dtype=np.float32)
        self.prev_gray = self._to_gray(frame)
        self.prev_points = self._detect_points(self.prev_gray, self.bbox_xywh)
        if self.prev_points is None or len(self.prev_points) < self.min_points:
            self.prev_points = self._grid_points(int(x), int(y), int(w), int(h))
        return self.prev_points is not None and len(self.prev_points) > 0

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        if self.prev_gray is None:
            return False, tuple(float(v) for v in self.bbox_xywh.tolist())

        gray = self._to_gray(frame)
        if self.prev_points is None or len(self.prev_points) < self.min_points:
            self.prev_points = self._detect_points(self.prev_gray, self.bbox_xywh)
            if self.prev_points is None or len(self.prev_points) < self.min_points:
                self.prev_gray = gray
                return False, tuple(float(v) for v in self.bbox_xywh.tolist())

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            self.prev_gray = gray
            return False, tuple(float(v) for v in self.bbox_xywh.tolist())

        valid = status.reshape(-1) == 1
        good_old = self.prev_points[valid].reshape(-1, 2)
        good_new = next_points[valid].reshape(-1, 2)
        if len(good_new) < self.min_points:
            self.prev_gray = gray
            self.prev_points = self._detect_points(gray, self.bbox_xywh)
            return False, tuple(float(v) for v in self.bbox_xywh.tolist())

        delta = np.median(good_new - good_old, axis=0)
        scale = 1.0
        if len(good_new) >= 8:
            old_center = good_old.mean(axis=0)
            new_center = good_new.mean(axis=0)
            old_dist = np.linalg.norm(good_old - old_center, axis=1)
            new_dist = np.linalg.norm(good_new - new_center, axis=1)
            valid_dist = old_dist > 1e-3
            if np.any(valid_dist):
                ratio = np.median(new_dist[valid_dist] / old_dist[valid_dist])
                if np.isfinite(ratio):
                    scale = float(np.clip(ratio, 0.85, 1.15))

        x, y, w, h = self.bbox_xywh.tolist()
        new_w = max(8.0, w * scale)
        new_h = max(8.0, h * scale)
        new_x = x + float(delta[0]) - (new_w - w) * 0.5
        new_y = y + float(delta[1]) - (new_h - h) * 0.5

        frame_h, frame_w = gray.shape[:2]
        new_x = float(np.clip(new_x, 0.0, max(0.0, frame_w - new_w)))
        new_y = float(np.clip(new_y, 0.0, max(0.0, frame_h - new_h)))
        self.bbox_xywh = np.array([new_x, new_y, new_w, new_h], dtype=np.float32)

        inside_points = self._points_inside_bbox(good_new.reshape(-1, 1, 2), self.bbox_xywh)
        self.prev_points = (
            inside_points
            if inside_points is not None and len(inside_points) >= self.min_points
            else self._detect_points(gray, self.bbox_xywh)
        )
        self.prev_gray = gray
        return True, tuple(float(v) for v in self.bbox_xywh.tolist())


def _opencv_tracker_backends() -> list[TrackerBackend]:
    backends: list[TrackerBackend] = []
    if hasattr(cv2, "TrackerKCF_create"):
        backends.append(TrackerBackend("kcf", cv2.TrackerKCF_create))
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        backends.append(TrackerBackend("legacy_kcf", cv2.legacy.TrackerKCF_create))
    if hasattr(cv2, "TrackerCSRT_create"):
        backends.append(TrackerBackend("csrt", cv2.TrackerCSRT_create))
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        backends.append(TrackerBackend("legacy_csrt", cv2.legacy.TrackerCSRT_create))
    if hasattr(cv2, "TrackerMOSSE_create"):
        backends.append(TrackerBackend("mosse", cv2.TrackerMOSSE_create))
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        backends.append(TrackerBackend("legacy_mosse", cv2.legacy.TrackerMOSSE_create))
    return backends


def available_tracker_backends(include_debug_backends: bool = False) -> list[str]:
    """List tracker backends available in the current environment."""
    names = [backend.name for backend in _opencv_tracker_backends()]
    if include_debug_backends:
        names.append(LKOpticalFlowTracker.backend_name)
    return names


def create_tracker(
    preferred_order: list[str] | None = None,
    allow_debug_fallback: bool = False,
) -> tuple[Any, str]:
    """Create a tracker instance.

    Default behavior is strict KCF-only. Non-KCF backends are available only for debug runs.
    """
    order = preferred_order or ["kcf", "legacy_kcf"]
    if allow_debug_fallback:
        order = order + ["csrt", "legacy_csrt", "mosse", "legacy_mosse", LKOpticalFlowTracker.backend_name]

    backends = {backend.name: backend for backend in _opencv_tracker_backends()}
    for name in order:
        if name == LKOpticalFlowTracker.backend_name and allow_debug_fallback:
            return LKOpticalFlowTracker(), LKOpticalFlowTracker.backend_name
        backend = backends.get(name)
        if backend is None:
            continue
        return OpenCVTrackerAdapter(backend.factory, backend.name), backend.name

    raise RuntimeError(
        "KCF tracker backend is unavailable. Install/enable OpenCV contrib "
        "(opencv-contrib-python) so cv2.TrackerKCF_create or cv2.legacy.TrackerKCF_create exists."
    )
