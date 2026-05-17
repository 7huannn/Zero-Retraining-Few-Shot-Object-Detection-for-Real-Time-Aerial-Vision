#!/usr/bin/env python3
"""Inspect per-support cosine similarities for one candidate crop.

This utility helps reproduce the slide-level view:
- cosine(candidate, support_1)
- cosine(candidate, support_2)
- ...
plus aggregated raw score and normalized matcher score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusion import aggregate_support_scores
from matcher import build_matcher


def _load_video_path(preprocessed_dir: Path, sample_id: str) -> str:
    metadata_path = preprocessed_dir / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for item in metadata:
        item_id = str(item.get("video_id", item.get("sample_id", "")))
        if item_id == sample_id:
            return str(item["video_path"])

    raise KeyError(f"Sample '{sample_id}' not found in {metadata_path}")


def _extract_crop_from_video(
    video_path: str,
    frame_index: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_index} from: {video_path}")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {(x1, y1, x2, y2)}")

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise RuntimeError("Empty crop extracted from frame")
    return crop


def _load_reference_crops(preprocessed_dir: Path, sample_id: str) -> list[np.ndarray]:
    sample_dir = preprocessed_dir / sample_id
    path = sample_dir / "reference_crops.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference crops: {path}")

    arr = np.load(path, allow_pickle=True)
    crops: list[np.ndarray] = []
    for item in arr.tolist():
        if isinstance(item, np.ndarray) and item.size > 0:
            crops.append(item)

    if not crops:
        raise ValueError("No valid reference crops loaded")

    return crops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect per-support cosine similarities")

    parser.add_argument("--preprocessed-dir", default="preprocessed_data/public_test")
    parser.add_argument("--sample-id", required=True)

    parser.add_argument("--matcher", choices=["mobileclip2", "siamese"], default="mobileclip2")
    parser.add_argument("--support-aggregation", choices=["mean", "max"], default="mean")

    parser.add_argument("--clip-model-name", default="MobileCLIP2-S0")
    parser.add_argument("--clip-encoder-path", default="models/mobileclip2_image_encoder_fp16.pt")
    parser.add_argument("--clip-pretrained", default="dfndr2b")
    parser.add_argument("--siamese-checkpoint", default="result/siamese/best.pt")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--candidate-image", default="")
    parser.add_argument("--frame-index", type=int, default=-1)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Used with --frame-index to crop from video frame",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    preprocessed_dir = Path(args.preprocessed_dir)
    reference_crops = _load_reference_crops(preprocessed_dir, args.sample_id)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    matcher = build_matcher(
        args.matcher,
        clip_model_name=args.clip_model_name,
        clip_encoder_path=args.clip_encoder_path,
        clip_pretrained=args.clip_pretrained,
        siamese_checkpoint=args.siamese_checkpoint,
        device=device,
    )
    matcher.set_support_embeddings(reference_crops)

    if args.candidate_image:
        candidate = cv2.imread(args.candidate_image, cv2.IMREAD_UNCHANGED)
        if candidate is None:
            raise FileNotFoundError(f"Cannot read candidate image: {args.candidate_image}")
    else:
        if args.frame_index < 0 or args.bbox is None:
            raise ValueError("Provide either --candidate-image OR (--frame-index + --bbox)")
        video_path = _load_video_path(preprocessed_dir, args.sample_id)
        candidate = _extract_crop_from_video(video_path, args.frame_index, tuple(args.bbox))

    emb = matcher.encode_single(candidate)
    if emb is None or matcher.support_embeddings is None:
        raise RuntimeError("Failed to encode candidate or support embeddings are missing")

    per_support = F.cosine_similarity(
        emb.expand(matcher.support_embeddings.shape[0], -1),
        matcher.support_embeddings,
        dim=1,
    ).detach().cpu().numpy()

    raw = aggregate_support_scores(per_support, args.support_aggregation)
    norm = matcher.raw_to_similarity(raw)

    print("=" * 72)
    print(f"sample_id           : {args.sample_id}")
    print(f"matcher             : {matcher.name}")
    print(f"support_count       : {len(per_support)}")
    print(f"support_aggregation : {args.support_aggregation}")
    if args.candidate_image:
        print(f"candidate_image     : {args.candidate_image}")
    else:
        print(f"frame_index         : {args.frame_index}")
        print(f"bbox_xyxy           : {tuple(args.bbox)}")
    print("-" * 72)
    print("Per-support cosine:")
    for i, value in enumerate(per_support, start=1):
        print(f"  support_{i:02d}: {float(value):.6f}")
    print("-" * 72)
    print(f"raw_match_score     : {float(raw):.6f}")
    print(f"match_score         : {float(norm):.6f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
