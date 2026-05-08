#!/usr/bin/env python3
"""Render bounding boxes from a JSON file onto a video."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from JSON onto a video and save annotated output."
    )
    parser.add_argument("--json", required=True, help="Path to JSON file with bboxes.")
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--output", required=True, help="Path for output annotated video.")
    parser.add_argument(
        "--video-id",
        default="",
        help=(
            "Video id to select from a multi-video annotations/submission JSON. "
            "Defaults to the parent folder name of --video when possible."
        ),
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Offset applied to bbox frame id (effective_frame = frame + frame_offset).",
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="Bounding box line thickness."
    )
    parser.add_argument(
        "--font-scale", type=float, default=0.5, help="Text label font scale."
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional fixed text to draw above every bbox, for example 'GT'.",
    )
    parser.add_argument(
        "--zoom-inset",
        action="store_true",
        help="Draw a magnified inset of the first bbox in the top-right corner.",
    )
    parser.add_argument(
        "--inset-size",
        type=int,
        default=220,
        help="Square inset size in pixels when --zoom-inset is enabled.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="Output video FPS. Defaults to source FPS when omitted or <= 0.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index to render from (ignored when --demo-frames > 0).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help="Number of frames to render (0 means until end of video).",
    )
    parser.add_argument(
        "--demo-frames",
        type=int,
        default=0,
        help=(
            "If > 0, render a short clip starting at the first frame that has a valid bbox "
            "and write this many frames."
        ),
    )
    return parser.parse_args()


def collect_boxes(obj, boxes, track_prefix=""):
    """Recursively collect bbox objects with frame/x1/y1/x2/y2 keys."""
    if isinstance(obj, dict):
        if {"frame", "x1", "y1", "x2", "y2"}.issubset(obj.keys()):
            boxes.append(
                {
                    "frame": int(obj["frame"]),
                    "x1": int(obj["x1"]),
                    "y1": int(obj["y1"]),
                    "x2": int(obj["x2"]),
                    "y2": int(obj["y2"]),
                    "score": float(obj.get("score", 0.0)),
                    "track": track_prefix,
                }
            )
            return

        if "annotations" in obj and isinstance(obj["annotations"], list):
            for idx, ann in enumerate(obj["annotations"]):
                collect_boxes(ann, boxes, f"ann_{idx}")
            return

        if "bboxes" in obj and isinstance(obj["bboxes"], list):
            for bbox in obj["bboxes"]:
                collect_boxes(bbox, boxes, track_prefix)
            return

        for value in obj.values():
            collect_boxes(value, boxes, track_prefix)
        return

    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            local_prefix = track_prefix if track_prefix else f"item_{idx}"
            collect_boxes(item, boxes, local_prefix)


def select_video_data(data, video_path: Path, video_id: str):
    if not isinstance(data, list):
        return data, video_id

    selected_video_id = video_id or video_path.parent.name
    matching = [
        item
        for item in data
        if isinstance(item, dict) and item.get("video_id") == selected_video_id
    ]
    if matching:
        return matching, selected_video_id

    if video_id:
        available = [
            str(item.get("video_id"))
            for item in data
            if isinstance(item, dict) and item.get("video_id")
        ]
        raise ValueError(
            f"Video id '{video_id}' not found in JSON. Available ids: {', '.join(available)}"
        )

    return data, selected_video_id


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_zoom_inset(frame, box, inset_size, color):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    crop_side = max(80, int(max(box_w, box_h) * 3.0))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    crop_x1 = max(0, cx - crop_side // 2)
    crop_y1 = max(0, cy - crop_side // 2)
    crop_x2 = min(width, crop_x1 + crop_side)
    crop_y2 = min(height, crop_y1 + crop_side)
    crop_x1 = max(0, crop_x2 - crop_side)
    crop_y1 = max(0, crop_y2 - crop_side)

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.size == 0:
        return

    rel_x1 = x1 - crop_x1
    rel_y1 = y1 - crop_y1
    rel_x2 = x2 - crop_x1
    rel_y2 = y2 - crop_y1
    cv2.rectangle(crop, (rel_x1, rel_y1), (rel_x2, rel_y2), color, 2)

    inset = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_CUBIC)
    margin = 16
    top = margin
    left = width - inset_size - margin
    frame[top : top + inset_size, left : left + inset_size] = inset
    cv2.rectangle(frame, (left, top), (left + inset_size, top + inset_size), color, 3)
    cv2.putText(
        frame,
        "zoom",
        (left + 8, top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    json_path = Path(args.json)
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data, selected_video_id = select_video_data(data, video_path, args.video_id)

    raw_boxes = []
    collect_boxes(data, raw_boxes)
    if not raw_boxes:
        raise ValueError(f"No bounding boxes found in {json_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    output_fps = args.output_fps if args.output_fps > 0 else fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_to_boxes = defaultdict(list)
    out_of_range_boxes = 0
    invalid_boxes = 0

    for box in raw_boxes:
        frame_idx = box["frame"] + args.frame_offset
        if frame_idx < 0 or frame_idx >= total_frames:
            out_of_range_boxes += 1
            continue

        clamped = clamp_box(box["x1"], box["y1"], box["x2"], box["y2"], width, height)
        if clamped is None:
            invalid_boxes += 1
            continue

        frame_to_boxes[frame_idx].append((clamped, box["track"], box["score"]))

    render_start = max(0, min(total_frames - 1, args.start_frame))
    render_end = total_frames - 1

    if args.num_frames > 0:
        render_end = min(total_frames - 1, render_start + args.num_frames - 1)

    if args.demo_frames > 0:
        if not frame_to_boxes:
            cap.release()
            writer.release()
            raise ValueError(
                "No valid in-range boxes found. Cannot use --demo-frames on this video/json pair."
            )
        render_start = min(frame_to_boxes.keys())
        render_end = min(total_frames - 1, render_start + args.demo_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, render_start)
    frame_idx = render_start
    frames_written = 0
    while frame_idx <= render_end:
        ok, frame = cap.read()
        if not ok:
            break

        current_boxes = frame_to_boxes.get(frame_idx, [])
        for (x1, y1, x2, y2), track_id, score in current_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)
            label = args.label or track_id or "bbox"
            if score > 0:
                label = f"{label} {score:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                args.font_scale,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if args.zoom_inset and current_boxes:
            best_box, _, _ = max(
                current_boxes,
                key=lambda item: (item[2], (item[0][2] - item[0][0]) * (item[0][3] - item[0][1])),
            )
            draw_zoom_inset(frame, best_box, args.inset_size, (0, 255, 0))

        writer.write(frame)
        frame_idx += 1
        frames_written += 1

    cap.release()
    writer.release()

    used = sum(
        len(v) for frame_no, v in frame_to_boxes.items() if render_start <= frame_no <= render_end
    )
    print(f"Saved annotated video: {output_path}")
    print(f"Selected video id: {selected_video_id}")
    print(f"Video frames: {total_frames}")
    print(f"Output FPS: {output_fps}")
    print(f"Rendered frame range: [{render_start}, {render_end}]")
    print(f"Rendered frames written: {frames_written}")
    print(f"Boxes loaded: {len(raw_boxes)}")
    print(f"Boxes drawn: {used}")
    print(f"Boxes skipped (out-of-range frames): {out_of_range_boxes}")
    print(f"Boxes skipped (invalid/clamped): {invalid_boxes}")


if __name__ == "__main__":
    main()
