"""Score normalization and fusion helpers for the detector pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def cosine_to_similarity(value: float) -> float:
    """Map cosine score from [-1, 1] to [0, 1]."""
    return float(np.clip((value + 1.0) * 0.5, 0.0, 1.0))


def normalize_detector_scores(confidences: Iterable[float]) -> np.ndarray:
    """Normalize detector confidences for frame-level re-ranking."""
    scores = np.asarray(list(confidences), dtype=np.float32)
    if scores.size == 0:
        return scores
    scores = np.clip(scores, 0.0, 1.0)
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score - min_score < 1e-6:
        return scores
    return (scores - min_score) / (max_score - min_score)


def aggregate_support_scores(scores: np.ndarray, mode: str = "mean") -> float:
    """Aggregate candidate-vs-support similarities."""
    if scores.size == 0:
        return 0.0
    if mode == "max":
        return float(scores.max())
    return float(scores.mean())


def fuse_scores(
    det_score: float,
    match_score: float,
    w_det: float,
    w_match: float,
    bonus: float = 0.0,
    penalty: float = 0.0,
) -> float:
    """Weighted fusion for detector + selected matcher score."""
    value = w_det * det_score + w_match * match_score + bonus - penalty
    return float(np.clip(value, 0.0, 1.0))
