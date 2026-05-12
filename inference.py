"""Main inference pipeline orchestration for few-shot detection and tracking."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import DataConfig, InferenceConfig, ModelConfig, PipelineConfig
from detector import YOLOEDetector
from fusion import aggregate_support_scores, cosine_to_similarity, fuse_scores, normalize_detector_scores
from matcher import MobileCLIP2Matcher
from tracker import KCFTracker
from utils import seed_everything


class FewShotDetectionPipeline:
    """End-to-end few-shot detection and tracking pipeline."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
    ) -> None:
        """
        Initialize pipeline with configuration.

        Args:
            inference_config: Inference hyperparameters
            model_config: Model paths and initialization
            data_config: Data paths
        """
        self.inference_config = inference_config
        self.model_config = model_config
        self.data_config = data_config
        self.device = model_config.get_device()

        print(f"[Pipeline] Device: {self.device}")

        # Initialize components
        self.detector = YOLOEDetector(
            model_path=model_config.yoloe_weights,
            conf_threshold=inference_config.yoloe_conf,
            top_k_proposals=inference_config.top_k_proposals,
            device=self.device,
        )

        self.matcher = MobileCLIP2Matcher(
            model_name=model_config.clip_model_name,
            encoder_path=model_config.clip_encoder_path,
            pretrained=model_config.clip_pretrained,
            device=self.device,
        )

        # State tracking
        self.tracker: KCFTracker | None = None
        self.frame_w = 0
        self.frame_h = 0
        self.frames_since_detection = 0
        self.last_fused_score = inference_config.fused_accept_threshold
        self.best_fused_seen = inference_config.fused_accept_threshold

        # Stats
        self.tracker_init_count = 0
        self.tracker_update_frames = 0
        self.tracker_successful_updates = 0
        self.tracker_bbox_return_frames = 0
        self.accepted_score_components: list[dict[str, float]] = []
        self.reference_crops: list[np.ndarray] = []

    def initialize_for_video(self, preprocessed_data: dict[str, Any], frame_width: int, frame_height: int) -> None:
        """
        Initialize pipeline state for a new video.

        Args:
            preprocessed_data: Preprocessed support set (initial_vpe, reference_crops, class_name)
            frame_width: Video frame width
            frame_height: Video frame height
        """
        initial_vpe = preprocessed_data["initial_vpe"]
        class_name = preprocessed_data["class_name"]
        reference_crops = preprocessed_data["reference_crops"]

        # Filter valid crops
        self.reference_crops = [item for item in reference_crops if isinstance(item, np.ndarray) and item.size > 0]
        if not self.reference_crops:
            raise RuntimeError("No valid reference crops in preprocessed data")

        # Set up detector with visual prompt
        self.detector.set_visual_prompt(class_name, initial_vpe)

        # Pre-encode support set for matcher
        self.matcher.set_support_embeddings(self.reference_crops)

        # Initialize state
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.tracker = None
        self.frames_since_detection = 0
        self.last_fused_score = self.inference_config.fused_accept_threshold
        self.best_fused_seen = self.inference_config.fused_accept_threshold
        self.tracker_init_count = 0
        self.tracker_update_frames = 0
        self.tracker_successful_updates = 0
        self.tracker_bbox_return_frames = 0
        self.accepted_score_components = []

        print(f"[Pipeline] Initialized for class='{class_name}', frame_size={frame_width}x{frame_height}, "
              f"support_crops={len(self.reference_crops)}")

    def _crop_object(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Extract and optionally mask a region from an image."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_w = int((x2 - x1) * self.inference_config.crop_padding_ratio)
        pad_h = int((y2 - y1) * self.inference_config.crop_padding_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        if x1 >= x2 or y1 >= y2:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if mask is None:
            return crop

        # Apply mask with alpha channel
        full_mask = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = full_mask[y1:y2, x1:x2]
        binary_mask = (mask_crop > 0.4).astype(np.uint8)
        alpha = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=2) * 255

        if alpha.shape[:2] != crop.shape[:2]:
            alpha = cv2.resize(alpha, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        out = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = alpha
        return out

    def _bbox_iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bboxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        union = area_a + area_b - inter

        if union <= 0.0:
            return 0.0

        return inter / union

    def _score_candidate(
        self,
        frame: np.ndarray,
        candidate: dict[str, Any],
        det_score: float,
        tracker_bbox: tuple[int, int, int, int] | None,
    ) -> dict[str, Any] | None:
        """Score a single candidate detection using MobileCLIP2."""
        crop = self._crop_object(frame, candidate["xyxy"], candidate["mask"])
        if crop is None:
            return None

        clip_embedding = self.matcher.encode_single(crop)
        if clip_embedding is None:
            return None

        # Compute similarity against support set
        clip_scores = F.cosine_similarity(
            clip_embedding.expand(self.matcher.support_embeddings.shape[0], -1),
            self.matcher.support_embeddings,
            dim=1,
        ).detach().cpu().numpy()

        clip_cos = aggregate_support_scores(clip_scores, self.inference_config.support_aggregation)
        clip_sim = cosine_to_similarity(clip_cos)

        # Tracker bonus: small boost if candidate overlaps with tracker bbox
        bonus = 0.0
        if tracker_bbox is not None:
            overlap = self._bbox_iou(candidate["xyxy"], tracker_bbox)
            bonus = self.inference_config.tracker_bonus * max(0.0, overlap - 0.2)

        # Fuse detector + matcher score
        fused = fuse_scores(
            det_score=det_score,
            clip_score=clip_sim,
            w_det=self.inference_config.w_det,
            w_clip=self.inference_config.w_clip,
            bonus=bonus,
        )

        return {
            "candidate": candidate,
            "crop": crop,
            "det_score": det_score,
            "clip_score": clip_sim,
            "fused_score": fused,
            "clip_embedding": clip_embedding,
        }

    def _should_reset_tracker(self, bbox: tuple[int, int, int, int]) -> bool:
        """Check if bbox is too close to frame edges (invalidate tracker)."""
        x1, y1, x2, y2 = bbox
        t = self.inference_config.edge_proximity_threshold
        return x1 <= t or y1 <= t or x2 >= self.frame_w - t or y2 >= self.frame_h - t

    def _update_reference_cache(self, scored: dict[str, Any]) -> None:
        """Update reference gallery with high-confidence detections."""
        if scored["fused_score"] < self.inference_config.similarity_add_threshold:
            return

        self.reference_crops.append(scored["crop"])
        self.matcher.support_embeddings = torch.cat(
            [self.matcher.support_embeddings, scored["clip_embedding"]],
            dim=0,
        )

        # Trim cache if too large
        overflow = len(self.reference_crops) - self.inference_config.max_reference_samples
        if overflow > 0:
            self.reference_crops = self.reference_crops[overflow:]
            self.matcher.support_embeddings = self.matcher.support_embeddings[overflow:]

    def predict_frame(self, frame: np.ndarray, frame_index: int) -> tuple[list[int] | None, float]:
        """
        Process single frame: detect, match, track, return bbox + score.

        Args:
            frame: Input frame (BGR)
            frame_index: Frame number in video

        Returns:
            Tuple of (bbox_xyxy or None, score)
        """
        # Try tracking first
        run_detector = self.tracker is None or self.frames_since_detection >= self.inference_config.tracker_reinit_interval

        if not run_detector and self.tracker is not None:
            # Use tracker
            ok, box = self.tracker.update(frame)
            if ok and box is not None:
                x, y, w, h = box
                x1, y1, x2, y2 = max(0, x), max(0, y), min(self.frame_w, x + w), min(self.frame_h, y + h)
                if x2 > x1 and y2 > y1:
                    self.frames_since_detection += 1
                    self.tracker_bbox_return_frames += 1
                    self.last_fused_score *= 0.995
                    return [x1, y1, x2, y2], float(np.clip(self.last_fused_score, 0.0, 1.0))

            # Tracker failed, fall through to detector
            self.tracker = None
            run_detector = True

        # Run detector
        candidates = self.detector.detect(frame)
        if not candidates:
            # No candidates, try returning tracker bbox if it exists
            if self.tracker is not None:
                ok, box = self.tracker.update(frame)
                if ok and box is not None:
                    x, y, w, h = box
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(self.frame_w, x + w), min(self.frame_h, y + h)
                    if x2 > x1 and y2 > y1 and not self._should_reset_tracker((x1, y1, x2, y2)):
                        self.frames_since_detection += 1
                        self.tracker_bbox_return_frames += 1
                        self.last_fused_score *= 0.99
                        return [x1, y1, x2, y2], float(np.clip(self.last_fused_score, 0.0, 1.0))

            self.tracker = None
            return None, 0.0

        # Score candidates
        det_scores = normalize_detector_scores([item["conf"] for item in candidates])
        
        # Get current tracker bbox if available
        tracker_bbox = None
        if self.tracker is not None:
            ok, box = self.tracker.update(frame)
            if ok and box is not None:
                x, y, w, h = box
                tracker_bbox = (max(0, x), max(0, y), min(self.frame_w, x + w), min(self.frame_h, y + h))

        scored_candidates: list[dict[str, Any]] = []
        for candidate, det_score in zip(candidates, det_scores):
            scored = self._score_candidate(frame, candidate, float(det_score), tracker_bbox)
            if scored is not None:
                scored_candidates.append(scored)

        if not scored_candidates:
            # No valid candidates after scoring
            if tracker_bbox is not None and not self._should_reset_tracker(tracker_bbox):
                self.frames_since_detection += 1
                self.tracker_bbox_return_frames += 1
                self.last_fused_score *= 0.99
                return list(tracker_bbox), float(np.clip(self.last_fused_score, 0.0, 1.0))

            self.tracker = None
            return None, 0.0

        # Pick best candidate
        best = max(scored_candidates, key=lambda item: item["fused_score"])
        clip_ok = best["clip_score"] >= self.inference_config.min_clip_similarity
        accept = best["fused_score"] >= self.inference_config.fused_accept_threshold and clip_ok

        if accept:
            # Initialize tracker
            try:
                tracker = KCFTracker()
                initialized = tracker.init(frame, best["candidate"]["xywh"])
                self.tracker = tracker if initialized else None

                if initialized:
                    self.tracker_init_count += 1
                    if self.tracker_init_count == 1:
                        print(f"[Pipeline] Tracker initialized")

            except Exception as e:
                print(f"[Pipeline] Tracker init failed: {e}")
                self.tracker = None

            self.frames_since_detection = 0
            self.best_fused_seen = max(self.best_fused_seen, best["fused_score"])
            self.last_fused_score = float(best["fused_score"])
            self.accepted_score_components.append(
                {
                    "det_score": float(best["det_score"]),
                    "clip_score": float(best["clip_score"]),
                    "fused_score": float(best["fused_score"]),
                }
            )
            self._update_reference_cache(best)
            return list(best["candidate"]["xyxy"]), float(best["fused_score"])

        # No good detection, fall back to tracker
        if tracker_bbox is not None and not self._should_reset_tracker(tracker_bbox):
            self.frames_since_detection += 1
            self.tracker_bbox_return_frames += 1
            self.last_fused_score *= 0.99
            return list(tracker_bbox), float(np.clip(self.last_fused_score, 0.0, 1.0))

        self.tracker = None
        return None, 0.0

    def process_video(self, video_path: str, preprocessed_data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Process entire video.

        Args:
            video_path: Path to video file
            preprocessed_data: Support set data

        Returns:
            Tuple of (detections, stats)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        start_frame = max(0, int(self.inference_config.frame_start))
        end_frame = (
            int(self.inference_config.frame_end)
            if int(self.inference_config.frame_end) > 0
            else total_frames
        )
        end_frame = min(total_frames, max(start_frame, end_frame))

        if start_frame >= total_frames:
            cap.release()
            raise RuntimeError(f"frame_start {start_frame} >= total_frames {total_frames}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.initialize_for_video(preprocessed_data, frame_w, frame_h)

        detections: list[dict[str, Any]] = []
        fused_scores: list[float] = []

        if self.inference_config.save_frames:
            Path(self.inference_config.frames_output_dir).mkdir(parents=True, exist_ok=True)

        frame_limit = (
            self.inference_config.max_frames
            if self.inference_config.max_frames > 0
            else (end_frame - start_frame)
        )
        processed_limit = min(end_frame - start_frame, frame_limit)

        for offset in range(processed_limit):
            frame_idx = start_frame + offset
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            bbox, fused_score = self.predict_frame(frame, frame_idx)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                detections.append(
                    {
                        "frame": frame_idx,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "score": round(fused_score, 6),
                    }
                )
                fused_scores.append(fused_score)

                if self.inference_config.save_frames and frame_idx % self.inference_config.save_frame_interval == 0:
                    debug = frame.copy()
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        debug,
                        f"frame={frame_idx} fused={fused_score:.3f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imwrite(
                        str(
                            Path(self.inference_config.frames_output_dir) / f"frame_{frame_idx:06d}.jpg"
                        ),
                        debug,
                    )

        cap.release()

        frames_with_detection = len({item["frame"] for item in detections})
        processed_frames = max(0, min(processed_limit, total_frames - start_frame))
        component_stats = {
            key: float(np.mean([item[key] for item in self.accepted_score_components]))
            if self.accepted_score_components
            else 0.0
            for key in ("det_score", "clip_score", "fused_score")
        }

        stats = {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "frame_start": start_frame,
            "frame_end": end_frame,
            "frames_with_detection": frames_with_detection,
            "detection_rate": frames_with_detection / max(1, processed_frames),
            "mean_fused_score": float(np.mean(fused_scores)) if fused_scores else 0.0,
            "class_name": preprocessed_data["class_name"],
            "tracker_init_count": self.tracker_init_count,
            "tracker_update_frames": self.tracker_update_frames,
            "tracker_successful_updates": self.tracker_successful_updates,
            "tracker_bbox_return_frames": self.tracker_bbox_return_frames,
            "mean_det_score": component_stats["det_score"],
            "mean_clip_score": component_stats["clip_score"],
        }

        return detections, stats

    @staticmethod
    def _group_consecutive_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group consecutive frame detections into segments."""
        if not detections:
            return []
        detections.sort(key=lambda item: item["frame"])
        groups: list[dict[str, Any]] = []
        current = [detections[0]]
        for item in detections[1:]:
            if item["frame"] == current[-1]["frame"] + 1:
                current.append(item)
            else:
                groups.append({"bboxes": current})
                current = [item]
        groups.append({"bboxes": current})
        return groups

    def _load_preprocessed_sample(self, preprocessed_dir: Path, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Load preprocessed sample from directory."""
        sample_id = str(metadata.get("video_id", metadata.get("sample_id", "")))
        if not sample_id:
            raise KeyError("Metadata missing 'video_id' or 'sample_id'")

        sample_dir = preprocessed_dir / sample_id
        if not sample_dir.exists():
            raise FileNotFoundError(f"Preprocessed sample not found: {sample_dir}")

        initial_vpe = torch.load(sample_dir / "initial_vpe.pt", map_location=self.device)
        reference_crops = np.load(sample_dir / "reference_crops.npy", allow_pickle=True)
        video_path = str(metadata["video_path"])
        class_name = str(metadata["class_name"])

        return sample_id, {
            "initial_vpe": initial_vpe,
            "reference_crops": reference_crops,
            "class_name": class_name,
            "video_path": video_path,
        }

    def run_inference(
        self,
        preprocessed_dir: str,
        output_json: str,
        limit_videos: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Run full inference on preprocessed dataset.

        Args:
            preprocessed_dir: Directory with preprocessed samples
            output_json: Path to output submission JSON
            limit_videos: Max videos to process (0 = all)

        Returns:
            Tuple of (submission_list, video_stats_dict)
        """
        preprocessed_root = Path(preprocessed_dir)
        metadata_path = preprocessed_root / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if limit_videos > 0:
            dataset_metadata = dataset_metadata[:limit_videos]

        submission: list[dict[str, Any]] = []
        video_stats: dict[str, Any] = {}

        print(f"[Pipeline] Found {len(dataset_metadata)} videos to process")

        for metadata in dataset_metadata:
            sample_id, context = self._load_preprocessed_sample(preprocessed_root, metadata)
            print(f"\n{'=' * 80}\n[Pipeline] Processing: {sample_id}")

            try:
                detections, stats = self.process_video(context["video_path"], context)
                video_stats[sample_id] = stats
                print(
                    f"  detection_rate={stats['detection_rate'] * 100:.2f}% "
                    f"mean_fused={stats['mean_fused_score']:.3f} "
                    f"updates={stats['tracker_successful_updates']}"
                )
                submission.append(
                    {
                        "video_id": sample_id,
                        "detections": self._group_consecutive_detections(detections),
                    }
                )
            except Exception as exc:
                print(f"  ERROR {sample_id}: {exc}")
                submission.append({"video_id": sample_id, "detections": []})

        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
        print(f"\n[Pipeline] Inference output: {output_path}")

        if self.inference_config.generate_report:
            report_path = output_path.parent / f"{output_path.stem}_report.txt"
            self._save_analysis_report(video_stats, report_path)

        if self.inference_config.generate_plots:
            plot_path = output_path.parent / f"{output_path.stem}_stats.png"
            self._plot_detection_stats(video_stats, plot_path)

        return submission, video_stats

    @staticmethod
    def _save_analysis_report(video_stats: dict[str, Any], output_path: Path) -> None:
        """Save analysis report."""
        lines = ["=" * 60, "INFERENCE ANALYSIS REPORT", "=" * 60, ""]
        for video_id, stats in sorted(video_stats.items()):
            lines.append(f"Video: {video_id}")
            lines.append(f"  - Class: {stats.get('class_name', 'N/A')}")
            lines.append(f"  - Processed Frames: {stats.get('processed_frames', 0)}")
            lines.append(f"  - Frames with Detections: {stats.get('frames_with_detection', 0)}")
            lines.append(f"  - Detection Rate: {stats.get('detection_rate', 0.0) * 100:.2f}%")
            lines.append(f"  - Mean Fused Score: {stats.get('mean_fused_score', 0.0):.3f}")
            lines.append(f"  - Mean Det/CLIP: {stats.get('mean_det_score', 0.0):.3f} / {stats.get('mean_clip_score', 0.0):.3f}")
            lines.append(f"  - Tracker Init Count: {stats.get('tracker_init_count', 0)}")
            lines.append(f"  - Tracker Successful Updates: {stats.get('tracker_successful_updates', 0)}")
            lines.append(f"  - Tracker BBox Returns: {stats.get('tracker_bbox_return_frames', 0)}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Pipeline] Report saved: {output_path}")

    @staticmethod
    def _plot_detection_stats(video_stats: dict[str, Any], output_path: Path) -> None:
        """Plot detection statistics."""
        if not video_stats:
            return

        names = list(video_stats.keys())
        rates = [video_stats[name]["detection_rate"] * 100.0 for name in names]

        plt.figure(figsize=(12, 6))
        colors = ["green" if rate > 50 else "orange" if rate > 20 else "red" for rate in rates]
        bars = plt.bar(names, rates, color=colors)
        plt.ylim(0, 100)
        plt.ylabel("Detection Rate (%)")
        plt.title("Few-Shot Detection Pipeline - Detection Rate")
        plt.xticks(rotation=45, ha="right")

        for bar in bars:
            score = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() * 0.5, score, f"{score:.1f}%", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()
        print(f"[Pipeline] Plot saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Few-shot object detection and tracking pipeline using YOLOE + MobileCLIP2 + KCF.",
    )

    # Data args
    parser.add_argument("--preprocessed-dir", default="preprocessed_data/public_test")
    parser.add_argument("--output-json", default="result/submission.json")
    parser.add_argument("--limit-videos", type=int, default=0)

    # Model args
    parser.add_argument("--yoloe-weights", default="models/yoloe-11l-seg.pt")
    parser.add_argument("--clip-model-name", default="MobileCLIP2-S0")
    parser.add_argument("--clip-encoder-path", default="models/mobileclip2_image_encoder_fp16.pt")
    parser.add_argument("--clip-pretrained", default="dfndr2b")
    parser.add_argument("--device", default="auto")

    # Detection args
    parser.add_argument("--yoloe-conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=24)

    # Frame selection
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)

    # Tracker args
    parser.add_argument("--tracker-reinit", type=int, default=10)

    # Matcher aggregation
    parser.add_argument("--support-aggregation", choices=["mean", "max"], default="mean")

    # Score fusion
    parser.add_argument("--w-det", type=float, default=0.30)
    parser.add_argument("--w-clip", type=float, default=0.35)

    # Thresholds
    parser.add_argument("--fused-threshold", type=float, default=0.52)
    parser.add_argument("--clip-threshold", type=float, default=0.10)
    parser.add_argument("--similarity-add-threshold", type=float, default=0.80)
    parser.add_argument("--tracker-bonus", type=float, default=0.04)

    # Reference gallery
    parser.add_argument("--max-reference-samples", type=int, default=20)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)

    # Output args
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-frame-interval", type=int, default=25)
    parser.add_argument("--frames-output-dir", default="result/debug_frames")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    # Check model weights
    weights = args.yoloe_weights
    if not os.path.exists(weights):
        fallback = "models/yoloe-11s-seg.pt"
        if os.path.exists(fallback):
            print(f"[Main] Warning: {weights} not found, using {fallback}")
            weights = fallback
        else:
            raise FileNotFoundError(f"Missing YOLOE weights: {weights}")

    # Build config
    inference_cfg = InferenceConfig(
        yoloe_conf=args.yoloe_conf,
        top_k_proposals=args.top_k,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        tracker_reinit_interval=args.tracker_reinit,
        support_aggregation=args.support_aggregation,
        w_det=args.w_det,
        w_clip=args.w_clip,
        fused_accept_threshold=args.fused_threshold,
        min_clip_similarity=args.clip_threshold,
        similarity_add_threshold=args.similarity_add_threshold,
        tracker_bonus=args.tracker_bonus,
        max_reference_samples=args.max_reference_samples,
        crop_padding_ratio=args.crop_padding_ratio,
        save_frames=args.save_frames,
        save_frame_interval=args.save_frame_interval,
        frames_output_dir=args.frames_output_dir,
        max_frames=args.max_frames,
    )

    model_cfg = ModelConfig(
        yoloe_weights=weights,
        clip_model_name=args.clip_model_name,
        clip_encoder_path=args.clip_encoder_path,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
    )

    data_cfg = DataConfig(
        preprocessed_dir=args.preprocessed_dir,
        output_json=args.output_json,
        limit_videos=args.limit_videos,
    )

    # Create and run pipeline
    pipeline = FewShotDetectionPipeline(inference_cfg, model_cfg, data_cfg)
    pipeline.run_inference(
        preprocessed_dir=args.preprocessed_dir,
        output_json=args.output_json,
        limit_videos=args.limit_videos,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
