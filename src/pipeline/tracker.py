"""KCF tracker wrapper for temporal tracking."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def available_tracker_backends() -> list[str]:
    """Get list of available tracker backends."""
    backends = []
    if hasattr(cv2, "TrackerKCF_create"):
        backends.append("kcf")
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        backends.append("legacy_kcf")
    return backends


def create_tracker(allow_debug_fallback: bool = False) -> tuple[Any, str]:
    """
    Create a KCF tracker instance.

    Args:
        allow_debug_fallback: If True, use legacy_kcf as fallback

    Returns:
        Tuple of (tracker_instance, backend_name)

    Raises:
        RuntimeError: If no suitable tracker backend is available
    """
    try:
        if hasattr(cv2, "TrackerKCF_create"):
            tracker = cv2.TrackerKCF_create()
            return tracker, "kcf"
    except Exception as e:
        print(f"[Tracker] Failed to create cv2.TrackerKCF: {e}")

    try:
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            tracker = cv2.legacy.TrackerKCF_create()
            return tracker, "legacy_kcf"
    except Exception as e:
        print(f"[Tracker] Failed to create legacy.TrackerKCF: {e}")

    backends = available_tracker_backends()
    raise RuntimeError(f"No KCF tracker backend available. Backends: {backends}")


class KCFTracker:
    """Wrapper around KCF tracker for consistent interface."""

    def __init__(self) -> None:
        """Initialize tracker (uninitialized until init() is called)."""
        self.tracker = None
        self.backend_name = None
        self.is_initialized = False

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with first frame and bounding box.

        Args:
            frame: First frame (BGR)
            bbox: Initial bbox as (x, y, width, height)

        Returns:
            True if initialization successful
        """
        try:
            self.tracker, self.backend_name = create_tracker(allow_debug_fallback=False)
            self.is_initialized = self.tracker.init(frame, bbox)
            if self.is_initialized:
                print(f"[KCF] Initialized with backend: {self.backend_name}")
            return self.is_initialized
        except Exception as e:
            print(f"[KCF] Init failed: {e}")
            self.is_initialized = False
            return False

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        """
        Update tracker position on new frame.

        Args:
            frame: Next frame (BGR)

        Returns:
            Tuple of (success, bbox) where bbox is (x, y, w, h) or None
        """
        if not self.is_initialized or self.tracker is None:
            return False, None

        try:
            ok, box = self.tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                return True, (x, y, w, h)
            return False, None
        except Exception as e:
            # Don't print on every update failure, as it's common
            return False, None

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker = None
        self.backend_name = None
        self.is_initialized = False
