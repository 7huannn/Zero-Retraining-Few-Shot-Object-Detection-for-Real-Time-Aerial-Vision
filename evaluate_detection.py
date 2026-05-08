"""Evaluate prediction JSON against frame-level ground truth annotations.

The annotations format matches the train split in data/train/annotations/annotations.json.
The prediction format matches result/submission.json produced by predict.py.

This script can also export a filtered ground-truth subset for selected video IDs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import load_json, resolve_path, save_json


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BoxMatchResult:
    tp: int
    fp: int
    fn: int
    matched_ious: list[float]


def _bbox_from_mapping(payload: dict[str, Any]) -> BBox:
    return (
        int(payload["x1"]),
        int(payload["y1"]),
        int(payload["x2"]),
        int(payload["y2"]),
    )


def _collect_frames(entries: list[dict[str, Any]], kind: str) -> dict[str, dict[int, list[BBox]]]:
    video_frames: dict[str, dict[int, list[BBox]]] = {}
    for entry in entries:
        video_id = str(entry.get("video_id", ""))
        if not video_id:
            continue
        frames = video_frames.setdefault(video_id, {})
        groups = entry.get(kind, [])
        for group in groups:
            for bbox in group.get("bboxes", []):
                frame_index = int(bbox["frame"])
                frames.setdefault(frame_index, []).append(_bbox_from_mapping(bbox))
    return video_frames


def _iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _match_boxes(gt_boxes: list[BBox], pred_boxes: list[BBox], iou_threshold: float) -> BoxMatchResult:
    if not gt_boxes and not pred_boxes:
        return BoxMatchResult(tp=0, fp=0, fn=0, matched_ious=[])

    scored_pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            overlap = _iou(gt_box, pred_box)
            if overlap >= iou_threshold:
                scored_pairs.append((overlap, gt_index, pred_index))

    scored_pairs.sort(reverse=True, key=lambda item: item[0])
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_ious: list[float] = []
    for overlap, gt_index, pred_index in scored_pairs:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matched_ious.append(overlap)

    tp = len(matched_ious)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return BoxMatchResult(tp=tp, fp=fp, fn=fn, matched_ious=matched_ious)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _metrics_from_counts(tp: int, fp: int, fn: int, matched_ious: list[float]) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    mean_iou = _safe_divide(sum(matched_ious), len(matched_ious))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_iou": round(mean_iou, 6),
    }


def _filter_entries(entries: list[dict[str, Any]], video_ids: list[str]) -> list[dict[str, Any]]:
    if not video_ids:
        return entries
    wanted = set(video_ids)
    filtered = [entry for entry in entries if str(entry.get("video_id", "")) in wanted]
    missing = sorted(wanted.difference({str(entry.get("video_id", "")) for entry in filtered}))
    if missing:
        raise KeyError(f"Missing video IDs in ground truth: {', '.join(missing)}")
    return filtered


def _frame_window(frame_indices: list[int], pad_before: int, pad_after: int) -> tuple[int, int] | None:
    if not frame_indices:
        return None
    start_frame = max(0, min(frame_indices) - max(0, pad_before))
    end_frame = max(frame_indices) + max(0, pad_after) + 1
    return start_frame, end_frame


def _build_window_manifest(
    gt_entries: list[dict[str, Any]],
    video_ids: list[str],
    pad_before: int,
    pad_after: int,
) -> list[dict[str, Any]]:
    filtered_gt = _filter_entries(gt_entries, video_ids)
    gt_by_video = _collect_frames(filtered_gt, "annotations")
    manifest: list[dict[str, Any]] = []

    for video_id in sorted(gt_by_video):
        gt_frames = sorted(gt_by_video[video_id])
        window = _frame_window(gt_frames, pad_before, pad_after)
        if window is None:
            continue
        frame_start, frame_end = window
        manifest.append(
            {
                "video_id": video_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "pad_before": max(0, pad_before),
                "pad_after": max(0, pad_after),
                "gt_frame_start": min(gt_frames),
                "gt_frame_end": max(gt_frames),
                "gt_frame_count": len(gt_frames),
                "gt_box_count": sum(len(items) for items in gt_by_video[video_id].values()),
            }
        )
    return manifest


def evaluate(
    gt_entries: list[dict[str, Any]],
    pred_entries: list[dict[str, Any]],
    iou_threshold: float,
    video_ids: list[str],
    window_before: int = 0,
    window_after: int = 0,
) -> dict[str, Any]:
    gt_entries = _filter_entries(gt_entries, video_ids)
    if video_ids:
        selected_ids = set(video_ids)
        pred_entries = [entry for entry in pred_entries if str(entry.get("video_id", "")) in selected_ids]

    gt_by_video = _collect_frames(gt_entries, "annotations")
    pred_by_video = _collect_frames(pred_entries, "detections")
    window_map: dict[str, tuple[int, int]] = {}
    if window_before > 0 or window_after > 0:
        for video_id, frames in gt_by_video.items():
            window = _frame_window(sorted(frames), window_before, window_after)
            if window is not None:
                window_map[video_id] = window

    selected_video_ids = sorted(set(gt_by_video) | set(pred_by_video))
    video_reports: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    matched_ious: list[float] = []

    for video_id in selected_video_ids:
        gt_frames = gt_by_video.get(video_id, {})
        pred_frames = pred_by_video.get(video_id, {})
        frame_indices = sorted(set(gt_frames) | set(pred_frames))
        if video_id in window_map:
            window_start, window_end = window_map[video_id]
            frame_indices = [frame_index for frame_index in frame_indices if window_start <= frame_index < window_end]

        video_tp = video_fp = video_fn = 0
        video_ious: list[float] = []
        gt_box_count = sum(len(items) for items in gt_frames.values())
        pred_box_count = sum(len(items) for items in pred_frames.values())

        if video_id in window_map:
            window_start, window_end = window_map[video_id]
            gt_frames = {
                frame_index: items
                for frame_index, items in gt_frames.items()
                if window_start <= frame_index < window_end
            }
            pred_frames = {
                frame_index: items
                for frame_index, items in pred_frames.items()
                if window_start <= frame_index < window_end
            }
        gt_box_count = sum(len(items) for items in gt_frames.values())
        pred_box_count = sum(len(items) for items in pred_frames.values())

        for frame_index in frame_indices:
            match = _match_boxes(gt_frames.get(frame_index, []), pred_frames.get(frame_index, []), iou_threshold)
            video_tp += match.tp
            video_fp += match.fp
            video_fn += match.fn
            video_ious.extend(match.matched_ious)

        report = _metrics_from_counts(video_tp, video_fp, video_fn, video_ious)
        report.update(
            {
                "video_id": video_id,
                "frames": len(frame_indices),
                "gt_boxes": gt_box_count,
                "pred_boxes": pred_box_count,
            }
        )
        if video_id in window_map:
            report["frame_start"] = window_map[video_id][0]
            report["frame_end"] = window_map[video_id][1]
        video_reports.append(report)

        total_tp += video_tp
        total_fp += video_fp
        total_fn += video_fn
        matched_ious.extend(video_ious)

    overall = _metrics_from_counts(total_tp, total_fp, total_fn, matched_ious)
    overall.update(
        {
            "videos": len(selected_video_ids),
            "gt_boxes": sum(sum(len(items) for items in frames.values()) for frames in gt_by_video.values()),
            "pred_boxes": sum(sum(len(items) for items in frames.values()) for frames in pred_by_video.values()),
        }
    )
    return {
        "iou_threshold": iou_threshold,
        "videos": video_reports,
        "overall": overall,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detection predictions against train annotations.")
    parser.add_argument("--gt-annotations", default="data/train/annotations/annotations.json")
    parser.add_argument("--predictions", default="result/submission.json")
    parser.add_argument("--output", default="result/detection_metrics.json")
    parser.add_argument("--export-gt", default="")
    parser.add_argument("--export-window-manifest", default="")
    parser.add_argument("--window-before", type=int, default=0)
    parser.add_argument("--window-after", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--video-id", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gt_path = resolve_path(args.gt_annotations)
    pred_path = resolve_path(args.predictions)
    gt_entries = load_json(gt_path)
    pred_entries = load_json(pred_path)

    if args.export_gt:
        filtered_gt = _filter_entries(gt_entries, args.video_id)
        export_path = resolve_path(args.export_gt)
        save_json(export_path, filtered_gt)
        print(f"Exported ground truth subset: {export_path}")

    if args.export_window_manifest:
        manifest = _build_window_manifest(gt_entries, list(args.video_id), int(args.window_before), int(args.window_after))
        export_path = resolve_path(args.export_window_manifest)
        save_json(export_path, manifest)
        print(f"Exported window manifest: {export_path}")

    report = evaluate(
        gt_entries,
        pred_entries,
        float(args.iou_threshold),
        list(args.video_id),
        window_before=int(args.window_before),
        window_after=int(args.window_after),
    )
    output_path = resolve_path(args.output)
    save_json(output_path, report)
    print(f"Saved detection metrics: {output_path}")
    print(report["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())