#!/usr/bin/env python3
"""Run reproducible benchmark on train_val_mini_7cls and compute IoU/PR/RC/F1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_detection import evaluate


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_predict_for_video(
    repo_root: Path,
    video_id: str,
    frame_start: int,
    frame_end: int,
    preprocessed_runs_root: Path,
    output_path: Path,
    yoloe_conf: float,
    top_k: int,
    fused_threshold: float,
) -> None:
    pre_dir = preprocessed_runs_root / video_id
    cmd = [
        sys.executable,
        "predict.py",
        "--preprocessed-dir",
        str(pre_dir),
        "--output-json",
        str(output_path),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
        "--yoloe-conf",
        str(yoloe_conf),
        "--top-k",
        str(top_k),
        "--fused-threshold",
        str(fused_threshold),
        "--seed",
        "42",
    ]
    print(f"[run] {video_id}: frames[{frame_start}, {frame_end})")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def filter_submission_by_score(entries: list[dict[str, Any]], score_threshold: float) -> list[dict[str, Any]]:
    if score_threshold <= 0.0:
        return entries

    filtered: list[dict[str, Any]] = []
    for entry in entries:
        groups: list[dict[str, Any]] = []
        for group in entry.get("detections", []):
            bboxes = [
                bbox for bbox in group.get("bboxes", []) if float(bbox.get("score", 0.0)) >= score_threshold
            ]
            if bboxes:
                groups.append({"bboxes": bboxes})
        filtered.append({"video_id": entry.get("video_id"), "detections": groups})
    return filtered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark predict.py on train_val_mini_7cls with fixed GT.")
    parser.add_argument("--tag", required=True, help="Short run name used for output directory.")
    parser.add_argument("--yoloe-conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--fused-threshold", type=float, default=0.52)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--schedule", default="result/train_val_mini_7cls_schedule.json")
    parser.add_argument("--gt", default="result/train_val_mini_7cls_gt.json")
    parser.add_argument("--preprocessed-runs-root", default="preprocessed_data/train_val_mini_7cls_runs")
    parser.add_argument("--output-root", default="result/train_val_mini_7cls_tune")
    args = parser.parse_args(argv)

    repo_root = REPO_ROOT
    schedule_path = repo_root / args.schedule
    gt_path = repo_root / args.gt
    runs_root = repo_root / args.preprocessed_runs_root
    output_root = repo_root / args.output_root / args.tag
    preds_dir = output_root / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    schedule = load_json(schedule_path)
    gt_entries = load_json(gt_path)

    merged_submission: list[dict[str, Any]] = []
    for item in schedule:
        video_id = str(item["video_id"])
        frame_start = int(item["frame_start"])
        frame_end = int(item["frame_end"])
        pred_path = preds_dir / f"{video_id}.json"
        run_predict_for_video(
            repo_root=repo_root,
            video_id=video_id,
            frame_start=frame_start,
            frame_end=frame_end,
            preprocessed_runs_root=runs_root,
            output_path=pred_path,
            yoloe_conf=float(args.yoloe_conf),
            top_k=int(args.top_k),
            fused_threshold=float(args.fused_threshold),
        )
        entries = load_json(pred_path)
        if not isinstance(entries, list):
            raise TypeError(f"Invalid submission format in {pred_path}")
        merged_submission.extend(entries)

    merged_path = output_root / "submission_raw.json"
    save_json(merged_path, merged_submission)

    filtered_submission = filter_submission_by_score(merged_submission, float(args.score_threshold))
    filtered_path = output_root / "submission_filtered.json"
    save_json(filtered_path, filtered_submission)

    report = evaluate(
        gt_entries=gt_entries,
        pred_entries=filtered_submission,
        iou_threshold=float(args.iou_threshold),
        video_ids=[],
    )
    report["config"] = {
        "tag": args.tag,
        "yoloe_conf": float(args.yoloe_conf),
        "top_k": int(args.top_k),
        "fused_threshold": float(args.fused_threshold),
        "score_threshold": float(args.score_threshold),
        "iou_threshold": float(args.iou_threshold),
    }
    metrics_path = output_root / "metrics.json"
    save_json(metrics_path, report)

    print("\n[done] benchmark completed")
    print(f"output_dir={output_root}")
    print(f"overall={report['overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
