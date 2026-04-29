"""Hybrid few-shot inference: detector + MobileCLIP + Siamese + tracker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLOE

from fusion import aggregate_support_scores, cosine_to_similarity, fuse_scores, normalize_detector_scores
from siamese.siamese_model import SiameseVerifier
from tracker_adapter import available_tracker_backends, create_tracker
from utils import seed_everything


def _bootstrap_open_clip() -> Any:
    repo_root = Path(__file__).resolve().parent
    open_clip_src = repo_root / "ml-mobileclip" / "open_clip" / "src"
    if open_clip_src.exists() and str(open_clip_src) not in sys.path:
        sys.path.insert(0, str(open_clip_src))
    import open_clip

    return open_clip


open_clip = _bootstrap_open_clip()


@dataclass
class InferenceConfig:
    yoloe_conf: float = 0.001
    top_k_proposals: int = 24
    tracker_reinit_interval: int = 10
    edge_proximity_threshold: int = 10
    support_aggregation: str = "mean"
    w_det: float = 0.30
    w_clip: float = 0.35
    w_siam: float = 0.35
    fused_accept_threshold: float = 0.52
    min_clip_similarity: float = 0.10
    min_siam_similarity: float = 0.10
    tracker_bonus: float = 0.04
    similarity_add_threshold: float = 0.80
    max_reference_samples: int = 20
    crop_padding_ratio: float = 0.04
    save_frames: bool = False
    save_frame_interval: int = 25
    frames_output_dir: str = "result/debug_frames"
    generate_report: bool = True
    generate_plots: bool = True
    max_frames: int = 0


class VideoInference:
    """Run hybrid verification on one video stream at a time."""

    def __init__(
        self,
        yoloe_model_path: str,
        clip_model_name: str,
        clip_encoder_path: str,
        siamese_checkpoint: str | None,
        clip_pretrained: str | None,
        config: InferenceConfig,
    ) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLOE(yoloe_model_path)
        self.model.predictor = None

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=None,
        )
        if os.path.exists(clip_encoder_path):
            payload = torch.load(clip_encoder_path, map_location=self.device)
            self.clip_model.visual.load_state_dict(payload)
            print(f"Loaded MobileCLIP image encoder: {clip_encoder_path}")
        elif clip_pretrained:
            print(f"MobileCLIP encoder path missing, fallback to pretrained tag '{clip_pretrained}'")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=clip_pretrained,
            )
        else:
            raise FileNotFoundError(f"Missing clip encoder path: {clip_encoder_path}")

        self.clip_model.to(self.device)
        self.clip_model.eval()

        self.siamese_verifier: SiameseVerifier | None = None
        if siamese_checkpoint and os.path.exists(siamese_checkpoint):
            self.siamese_verifier = SiameseVerifier(siamese_checkpoint, device=self.device)
            print(f"Loaded Siamese checkpoint: {siamese_checkpoint}")
        else:
            print("Siamese checkpoint not found. Siamese branch will be disabled.")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.tracker: Any | None = None
        self.frames_since_detection = 0
        self.frame_w = 0
        self.frame_h = 0
        self.best_fused_seen = self.config.fused_accept_threshold
        self.last_fused_score = self.config.fused_accept_threshold
        self.tracker_backend_name: str | None = None
        self.tracker_init_count = 0
        self.tracker_update_frames = 0
        self.tracker_successful_updates = 0
        self.tracker_bbox_return_frames = 0
        self.accepted_score_components: list[dict[str, float]] = []
        print(f"Tracker backend candidates (KCF policy): {', '.join(available_tracker_backends())}")

        self.reference_crops: list[np.ndarray] = []
        self.clip_support_embeddings: torch.Tensor | None = None
        self.siamese_support_embeddings: torch.Tensor | None = None

    @staticmethod
    def _to_rgb_pil(image_crop: np.ndarray) -> Image.Image:
        if image_crop.shape[2] == 4:
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _encode_clip_many(self, crops: list[np.ndarray]) -> torch.Tensor:
        batch = torch.cat(
            [self.clip_preprocess(self._to_rgb_pil(crop)).unsqueeze(0) for crop in crops],
            dim=0,
        ).to(self.device)
        with torch.inference_mode():
            embeddings = self.clip_model.encode_image(batch)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _encode_clip_single(self, crop: np.ndarray) -> torch.Tensor | None:
        try:
            tensor = self.clip_preprocess(self._to_rgb_pil(crop)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embedding = self.clip_model.encode_image(tensor)
                embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding
        except Exception:
            return None

    def _encode_siamese_single(self, crop: np.ndarray) -> torch.Tensor | None:
        if self.siamese_verifier is None:
            return None
        try:
            embedding = self.siamese_verifier.encode(crop)
            embedding = F.normalize(embedding.to(self.device), p=2, dim=-1)
            return embedding
        except Exception:
            return None

    @staticmethod
    def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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

    def _crop_object(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_w = int((x2 - x1) * self.config.crop_padding_ratio)
        pad_h = int((y2 - y1) * self.config.crop_padding_ratio)
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

        full_mask = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = full_mask[y1:y2, x1:x2]
        binary_mask = (mask_crop > 0.4).astype(np.uint8)
        alpha = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=2) * 255
        if alpha.shape[:2] != crop.shape[:2]:
            alpha = cv2.resize(alpha, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = alpha
        return out

    def initialize_for_streaming(self, preprocessed_data: dict[str, Any], frame_width: int, frame_height: int) -> None:
        initial_vpe = preprocessed_data["initial_vpe"]
        class_name = preprocessed_data["class_name"]
        reference_crops = preprocessed_data["reference_crops"]
        self.reference_crops = [item for item in reference_crops if isinstance(item, np.ndarray) and item.size > 0]
        if not self.reference_crops:
            raise RuntimeError("No reference crops available for support encoding.")

        self.model.set_classes([class_name], initial_vpe)
        self.model.predictor = None

        self.clip_support_embeddings = self._encode_clip_many(self.reference_crops)
        self.siamese_support_embeddings = None
        if self.siamese_verifier is not None:
            with torch.inference_mode():
                siamese_embeddings = self.siamese_verifier.encode_many(self.reference_crops)
                self.siamese_support_embeddings = F.normalize(siamese_embeddings.to(self.device), p=2, dim=-1)

        self.frame_w = frame_width
        self.frame_h = frame_height
        self.tracker = None
        self.frames_since_detection = 0
        self.best_fused_seen = self.config.fused_accept_threshold
        self.last_fused_score = self.config.fused_accept_threshold
        self.tracker_backend_name = None
        self.tracker_init_count = 0
        self.tracker_update_frames = 0
        self.tracker_successful_updates = 0
        self.tracker_bbox_return_frames = 0
        self.accepted_score_components = []

    def _detect_candidates(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model(
            frame,
            save=False,
            conf=self.config.yoloe_conf,
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
        return detections[: self.config.top_k_proposals]

    def _tracker_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if self.tracker is None:
            return None
        self.tracker_update_frames += 1
        ok, box = self.tracker.update(frame)
        if not ok:
            self.tracker = None
            return None
        self.tracker_successful_updates += 1
        x, y, w, h = [int(v) for v in box]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.frame_w, x + w)
        y2 = min(self.frame_h, y + h)
        if x2 <= x1 or y2 <= y1:
            self.tracker = None
            return None
        return (x1, y1, x2, y2)

    def _score_candidate(
        self,
        frame: np.ndarray,
        candidate: dict[str, Any],
        det_score: float,
        tracker_bbox: tuple[int, int, int, int] | None,
    ) -> dict[str, Any] | None:
        if self.clip_support_embeddings is None:
            return None

        crop = self._crop_object(frame, candidate["xyxy"], candidate["mask"])
        if crop is None:
            return None

        clip_embedding = self._encode_clip_single(crop)
        if clip_embedding is None:
            return None
        clip_scores = F.cosine_similarity(
            clip_embedding.expand(self.clip_support_embeddings.shape[0], -1),
            self.clip_support_embeddings,
            dim=1,
        ).detach().cpu().numpy()
        clip_cos = aggregate_support_scores(clip_scores, self.config.support_aggregation)
        clip_sim = cosine_to_similarity(clip_cos)

        siam_sim = 0.0
        siam_embedding: torch.Tensor | None = None
        if self.siamese_support_embeddings is not None:
            siam_embedding = self._encode_siamese_single(crop)
            if siam_embedding is not None:
                siam_scores = F.cosine_similarity(
                    siam_embedding.expand(self.siamese_support_embeddings.shape[0], -1),
                    self.siamese_support_embeddings,
                    dim=1,
                ).detach().cpu().numpy()
                siam_cos = aggregate_support_scores(siam_scores, self.config.support_aggregation)
                siam_sim = cosine_to_similarity(siam_cos)

        w_det, w_clip, w_siam = self.config.w_det, self.config.w_clip, self.config.w_siam
        if self.siamese_support_embeddings is None:
            w_clip = min(1.0, w_clip + w_siam)
            w_siam = 0.0

        bonus = 0.0
        if tracker_bbox is not None:
            overlap = self._bbox_iou(candidate["xyxy"], tracker_bbox)
            bonus = self.config.tracker_bonus * max(0.0, overlap - 0.2)

        fused = fuse_scores(
            det_score=det_score,
            clip_score=clip_sim,
            siam_score=siam_sim,
            w_det=w_det,
            w_clip=w_clip,
            w_siam=w_siam,
            bonus=bonus,
        )

        return {
            "candidate": candidate,
            "crop": crop,
            "det_score": det_score,
            "clip_score": clip_sim,
            "siam_score": siam_sim,
            "fused_score": fused,
            "clip_embedding": clip_embedding,
            "siam_embedding": siam_embedding,
        }

    def _should_reset_tracker(self, bbox: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        t = self.config.edge_proximity_threshold
        return x1 <= t or y1 <= t or x2 >= self.frame_w - t or y2 >= self.frame_h - t

    def _update_reference_cache(self, scored: dict[str, Any]) -> None:
        if scored["fused_score"] < self.config.similarity_add_threshold:
            return

        self.reference_crops.append(scored["crop"])
        if self.clip_support_embeddings is not None:
            self.clip_support_embeddings = torch.cat(
                [self.clip_support_embeddings, scored["clip_embedding"]],
                dim=0,
            )
        if self.siamese_support_embeddings is not None and scored["siam_embedding"] is not None:
            self.siamese_support_embeddings = torch.cat(
                [self.siamese_support_embeddings, scored["siam_embedding"]],
                dim=0,
            )

        overflow = len(self.reference_crops) - self.config.max_reference_samples
        if overflow > 0:
            self.reference_crops = self.reference_crops[overflow:]
            if self.clip_support_embeddings is not None:
                self.clip_support_embeddings = self.clip_support_embeddings[overflow:]
            if self.siamese_support_embeddings is not None:
                self.siamese_support_embeddings = self.siamese_support_embeddings[overflow:]

    def predict_streaming(self, frame: np.ndarray, frame_index: int) -> tuple[list[int] | None, float]:
        _ = frame_index
        tracker_bbox = self._tracker_bbox(frame)
        run_detector = self.tracker is None or self.frames_since_detection >= self.config.tracker_reinit_interval

        if not run_detector:
            if tracker_bbox is not None:
                self.frames_since_detection += 1
                self.tracker_bbox_return_frames += 1
                self.last_fused_score *= 0.995
                return list(tracker_bbox), float(np.clip(self.last_fused_score, 0.0, 1.0))
            return None, 0.0

        candidates = self._detect_candidates(frame)
        if not candidates:
            if tracker_bbox is not None and not self._should_reset_tracker(tracker_bbox):
                self.frames_since_detection += 1
                self.tracker_bbox_return_frames += 1
                self.last_fused_score *= 0.99
                return list(tracker_bbox), float(np.clip(self.last_fused_score, 0.0, 1.0))
            self.tracker = None
            return None, 0.0

        det_scores = normalize_detector_scores([item["conf"] for item in candidates])
        scored_candidates: list[dict[str, Any]] = []
        for candidate, det_score in zip(candidates, det_scores):
            scored = self._score_candidate(frame, candidate, float(det_score), tracker_bbox)
            if scored is not None:
                scored_candidates.append(scored)

        if not scored_candidates:
            if tracker_bbox is not None:
                self.frames_since_detection += 1
                self.tracker_bbox_return_frames += 1
                self.last_fused_score *= 0.99
                return list(tracker_bbox), float(np.clip(self.last_fused_score, 0.0, 1.0))
            return None, 0.0

        best = max(scored_candidates, key=lambda item: item["fused_score"])
        clip_ok = best["clip_score"] >= self.config.min_clip_similarity
        siam_ok = best["siam_score"] >= self.config.min_siam_similarity or self.siamese_support_embeddings is None
        accept = best["fused_score"] >= self.config.fused_accept_threshold and (clip_ok or siam_ok)

        if accept:
            tracker, backend_name = create_tracker(allow_debug_fallback=False)
            initialized = bool(tracker.init(frame, best["candidate"]["xywh"]))
            self.tracker = tracker if initialized else None
            if initialized:
                if backend_name not in {"kcf", "legacy_kcf"}:
                    raise RuntimeError(f"Non-KCF tracker backend selected: {backend_name}")
                self.tracker_backend_name = backend_name
                self.tracker_init_count += 1
                if self.tracker_init_count == 1:
                    print(f"Tracker backend in use: {backend_name}")
            self.frames_since_detection = 0
            self.best_fused_seen = max(self.best_fused_seen, best["fused_score"])
            self.last_fused_score = float(best["fused_score"])
            self.accepted_score_components.append(
                {
                    "det_score": float(best["det_score"]),
                    "clip_score": float(best["clip_score"]),
                    "siam_score": float(best["siam_score"]),
                    "fused_score": float(best["fused_score"]),
                }
            )
            self._update_reference_cache(best)
            return list(best["candidate"]["xyxy"]), float(best["fused_score"])

        if tracker_bbox is not None and not self._should_reset_tracker(tracker_bbox):
            self.frames_since_detection += 1
            self.tracker_bbox_return_frames += 1
            self.last_fused_score *= 0.99
            return list(tracker_bbox), float(np.clip(self.last_fused_score, 0.0, 1.0))

        self.tracker = None
        return None, 0.0

    @staticmethod
    def _group_consecutive_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    def process_video(self, video_path: str, preprocessed_data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.initialize_for_streaming(preprocessed_data, frame_w, frame_h)

        detections: list[dict[str, Any]] = []
        fused_scores: list[float] = []
        if self.config.save_frames:
            Path(self.config.frames_output_dir).mkdir(parents=True, exist_ok=True)

        frame_limit = self.config.max_frames if self.config.max_frames > 0 else total_frames
        for frame_idx in range(min(total_frames, frame_limit)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            bbox, fused_score = self.predict_streaming(frame, frame_idx)
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
                if self.config.save_frames and frame_idx % self.config.save_frame_interval == 0:
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
                    cv2.imwrite(str(Path(self.config.frames_output_dir) / f"frame_{frame_idx:06d}.jpg"), debug)
        cap.release()

        frames_with_detection = len({item["frame"] for item in detections})
        component_stats = {
            key: float(np.mean([item[key] for item in self.accepted_score_components]))
            if self.accepted_score_components
            else 0.0
            for key in ("det_score", "clip_score", "siam_score", "fused_score")
        }
        stats = {
            "total_frames": total_frames,
            "processed_frames": min(total_frames, frame_limit),
            "frames_with_detection": frames_with_detection,
            "detection_rate": frames_with_detection / max(1, min(total_frames, frame_limit)),
            "mean_fused_score": float(np.mean(fused_scores)) if fused_scores else 0.0,
            "class_name": preprocessed_data["class_name"],
            "tracker_backend": self.tracker_backend_name or "uninitialized",
            "tracker_init_count": self.tracker_init_count,
            "tracker_update_frames": self.tracker_update_frames,
            "tracker_successful_updates": self.tracker_successful_updates,
            "tracker_bbox_return_frames": self.tracker_bbox_return_frames,
            "mean_det_score": component_stats["det_score"],
            "mean_clip_score": component_stats["clip_score"],
            "mean_siam_score": component_stats["siam_score"],
        }
        return detections, stats

    def _load_preprocessed_sample(self, preprocessed_dir: Path, metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        sample_id = str(metadata.get("video_id", metadata.get("sample_id", "")))
        if not sample_id:
            raise KeyError("Metadata entry missing 'video_id'/'sample_id'.")
        sample_dir = preprocessed_dir / sample_id
        if not sample_dir.exists():
            raise FileNotFoundError(f"Missing preprocessed sample dir: {sample_dir}")

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

    def run_inference(self, preprocessed_dir: str, output_json: str, limit_videos: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        preprocessed_root = Path(preprocessed_dir)
        metadata_path = preprocessed_root / "dataset_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        dataset_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if limit_videos > 0:
            dataset_metadata = dataset_metadata[:limit_videos]

        submission: list[dict[str, Any]] = []
        video_stats: dict[str, Any] = {}
        print(f"Found {len(dataset_metadata)} videos to process.")

        for metadata in dataset_metadata:
            sample_id, context = self._load_preprocessed_sample(preprocessed_root, metadata)
            print(f"\n{'=' * 80}\nProcessing: {sample_id}")
            try:
                detections, stats = self.process_video(context["video_path"], context)
                video_stats[sample_id] = stats
                print(
                    f"  detection_rate={stats['detection_rate'] * 100:.2f}% "
                    f"mean_fused={stats['mean_fused_score']:.3f} "
                    f"tracker={stats['tracker_backend']} updates={stats['tracker_successful_updates']}"
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
        print(f"\nInference output: {output_path}")

        if self.config.generate_report:
            report_path = output_path.parent / f"{output_path.stem}_report.txt"
            self.save_analysis_report(video_stats, report_path)
        if self.config.generate_plots:
            plot_path = output_path.parent / f"{output_path.stem}_stats.png"
            self.plot_detection_stats(video_stats, plot_path)

        return submission, video_stats

    @staticmethod
    def save_analysis_report(video_stats: dict[str, Any], output_path: Path) -> None:
        lines = ["=" * 60, "ANALYSIS REPORT", "=" * 60, ""]
        for video_id, stats in sorted(video_stats.items()):
            lines.append(f"Video: {video_id}")
            lines.append(f"  - Class: {stats.get('class_name', 'N/A')}")
            lines.append(f"  - Processed Frames: {stats.get('processed_frames', 0)}")
            lines.append(f"  - Frames with Detections: {stats.get('frames_with_detection', 0)}")
            lines.append(f"  - Detection Rate: {stats.get('detection_rate', 0.0) * 100:.2f}%")
            lines.append(f"  - Mean Fused Score: {stats.get('mean_fused_score', 0.0):.3f}")
            lines.append(f"  - Mean Det/CLIP/Siam: {stats.get('mean_det_score', 0.0):.3f} / {stats.get('mean_clip_score', 0.0):.3f} / {stats.get('mean_siam_score', 0.0):.3f}")
            lines.append(f"  - Tracker Backend: {stats.get('tracker_backend', 'N/A')}")
            lines.append(f"  - Tracker Init Count: {stats.get('tracker_init_count', 0)}")
            lines.append(f"  - Tracker Update Frames: {stats.get('tracker_update_frames', 0)}")
            lines.append(f"  - Tracker Successful Updates: {stats.get('tracker_successful_updates', 0)}")
            lines.append(f"  - Tracker BBoxes Used: {stats.get('tracker_bbox_return_frames', 0)}")
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Report saved: {output_path}")

    @staticmethod
    def plot_detection_stats(video_stats: dict[str, Any], output_path: Path) -> None:
        if not video_stats:
            return
        names = list(video_stats.keys())
        rates = [video_stats[name]["detection_rate"] * 100.0 for name in names]
        plt.figure(figsize=(12, 6))
        colors = ["green" if rate > 50 else "orange" if rate > 20 else "red" for rate in rates]
        bars = plt.bar(names, rates, color=colors)
        plt.ylim(0, 100)
        plt.ylabel("Detection Rate (%)")
        plt.title("Hybrid Pipeline Detection Rate")
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            score = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() * 0.5, score, f"{score:.1f}%", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()
        print(f"Plot saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid few-shot detector with MobileCLIP + Siamese + tracker.")
    parser.add_argument("--preprocessed-dir", default="preprocessed_data/public_test")
    parser.add_argument("--output-json", default="result/submission.json")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--yoloe-weights", default="models/yoloe-11l-seg.pt")
    parser.add_argument("--clip-model-name", default="MobileCLIP2-S0")
    parser.add_argument("--clip-encoder-path", default="models/mobileclip2_image_encoder_fp16.pt")
    parser.add_argument("--clip-pretrained", default="dfndr2b")
    parser.add_argument("--siamese-checkpoint", default="result/siamese/best.pt")
    parser.add_argument("--yoloe-conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--tracker-reinit", type=int, default=10)
    parser.add_argument("--fused-threshold", type=float, default=0.52)
    parser.add_argument("--clip-threshold", type=float, default=0.10)
    parser.add_argument("--siam-threshold", type=float, default=0.10)
    parser.add_argument("--support-aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--w-det", type=float, default=0.30)
    parser.add_argument("--w-clip", type=float, default=0.35)
    parser.add_argument("--w-siam", type=float, default=0.35)
    parser.add_argument("--similarity-add-threshold", type=float, default=0.80)
    parser.add_argument("--max-reference-samples", type=int, default=20)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.04)
    parser.add_argument("--disable-siamese", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-frame-interval", type=int, default=25)
    parser.add_argument("--frames-output-dir", default="result/debug_frames")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    weights = args.yoloe_weights
    if not os.path.exists(weights):
        fallback = "models/yoloe-11s-seg.pt"
        if os.path.exists(fallback):
            print(f"Warning: {weights} missing, fallback to {fallback}")
            weights = fallback
        else:
            raise FileNotFoundError(f"Missing YOLOE weight: {weights}")

    config = InferenceConfig(
        yoloe_conf=args.yoloe_conf,
        top_k_proposals=args.top_k,
        tracker_reinit_interval=args.tracker_reinit,
        support_aggregation=args.support_aggregation,
        w_det=args.w_det,
        w_clip=args.w_clip,
        w_siam=args.w_siam,
        fused_accept_threshold=args.fused_threshold,
        min_clip_similarity=args.clip_threshold,
        min_siam_similarity=args.siam_threshold,
        similarity_add_threshold=args.similarity_add_threshold,
        max_reference_samples=args.max_reference_samples,
        crop_padding_ratio=args.crop_padding_ratio,
        save_frames=args.save_frames,
        save_frame_interval=args.save_frame_interval,
        frames_output_dir=args.frames_output_dir,
        max_frames=args.max_frames,
    )

    inference = VideoInference(
        yoloe_model_path=weights,
        clip_model_name=args.clip_model_name,
        clip_encoder_path=args.clip_encoder_path,
        siamese_checkpoint=None if args.disable_siamese or args.w_siam <= 0.0 else args.siamese_checkpoint,
        clip_pretrained=args.clip_pretrained,
        config=config,
    )
    inference.run_inference(
        preprocessed_dir=args.preprocessed_dir,
        output_json=args.output_json,
        limit_videos=args.limit_videos,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
