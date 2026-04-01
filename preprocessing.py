"""Preprocess reference data for YOLOE and optional Siamese verification."""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from ultralytics import YOLO, YOLOE

try:
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
except Exception:  # pragma: no cover - compatibility fallback for older layouts
    from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

from utils import ensure_dir, resolve_path, save_json, seed_everything


def natural_key(value: str) -> list[int | str]:
    """Sort strings containing numeric suffixes in natural order."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def sample_id_to_class_name(sample_id: str) -> str:
    """Strip the trailing numeric suffix from a sample id."""
    return re.sub(r"_\d+$", "", sample_id)


class YOLOEPreprocessor:
    """Build initial VPE tensors and reference features from a dataset split."""

    def __init__(
        self,
        yolov8_model_path: str | Path,
        yoloe_model_path: str | Path,
        seed: int = 42,
        siamese_checkpoint: str | Path | None = None,
    ) -> None:
        seed_everything(seed)
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = YOLO(str(yolov8_model_path))
        self.yoloe_model = YOLOE(str(yoloe_model_path))
        self.siamese_verifier = None

        if siamese_checkpoint is not None:
            from siamese.siamese_model import SiameseVerifier

            self.siamese_verifier = SiameseVerifier(siamese_checkpoint, device=self.device)

    @staticmethod
    def _crop_object(
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
        mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        if x1 >= x2 or y1 >= y2:
            return None

        crop = image[y1:y2, x1:x2]
        if mask is None:
            return crop

        full_mask = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = full_mask[y1:y2, x1:x2]
        binary_mask = (mask_crop > 0.4).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.dilate(binary_mask, kernel, iterations=2) * 255

        if alpha.shape[:2] != crop.shape[:2]:
            alpha = cv2.resize(alpha, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)

        output = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        output[:, :, 3] = alpha
        return output

    def _extract_vpe(self, image: np.ndarray, bboxes: list[list[int]]) -> torch.Tensor | None:
        """Use YOLOE visual prompting to extract a VPE tensor."""
        if not bboxes:
            return None
        temp_path = Path(f"temp_frame_for_vpe_{random.randint(0, 999999)}.jpg")
        cv2.imwrite(str(temp_path), image)
        try:
            prompts = {
                "bboxes": np.array(bboxes, dtype=np.float32),
                "cls": np.array([0] * len(bboxes), dtype=np.int32),
            }
            predictor = YOLOEVPSegPredictor(
                overrides={
                    "task": getattr(self.yoloe_model, "task", "segment"),
                    "mode": "predict",
                    "save": False,
                    "verbose": False,
                    "batch": 1,
                    "device": self.device,
                    "imgsz": 640,
                },
                _callbacks=self.yoloe_model.callbacks,
            )
            self.yoloe_model.model.model[-1].nc = 1
            self.yoloe_model.model.names = ["object0"]
            predictor.set_prompts(prompts.copy())
            predictor.setup_model(model=self.yoloe_model.model)
            return predictor.get_vpe(str(temp_path))
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _extract_dominant_colors(
        self,
        crop: np.ndarray,
        num_colors: int = 5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract dominant LAB colors from one crop."""
        if crop is None or crop.size == 0:
            return None

        if crop.shape[2] == 4:
            mask = crop[:, :, 3] > 0
            if not np.any(mask):
                return None
            pixels = crop[:, :, :3][mask]
        else:
            pixels = crop.reshape(-1, 3)

        if pixels.shape[0] < num_colors:
            return None

        max_pixels = 10000
        if pixels.shape[0] > max_pixels:
            indices = np.linspace(0, pixels.shape[0] - 1, num=max_pixels, dtype=np.int32)
            pixels = pixels[indices]

        lab_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=self.seed, n_init="auto")
        kmeans.fit(lab_pixels)
        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        weights = counts / counts.sum()
        dominant_colors_lab = kmeans.cluster_centers_
        order = np.argsort(weights)[::-1]
        return dominant_colors_lab[order], weights[order]

    def _score_detection(
        self,
        bbox: tuple[int, int, int, int],
        confidence: float,
        area_weight: float,
        conf_weight: float,
    ) -> float:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        normalized_area = min(1.0, area / float(640 * 640))
        return area_weight * normalized_area + conf_weight * confidence

    def detect_and_extract_objects(
        self,
        object_images_dir: Path,
        conf_threshold: float,
        top_k: int,
        area_weight: float,
        conf_weight: float,
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
        """Extract high-quality crops from the reference images."""
        crops: list[tuple[np.ndarray, tuple[int, int, int, int], float]] = []
        image_files = sorted([path for path in object_images_dir.iterdir() if path.is_file()], key=lambda item: natural_key(item.name))

        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            results = self.detector.predict(str(image_path), save=False, conf=conf_threshold, verbose=False)
            result = results[0]
            if not result.boxes:
                continue

            detections = [
                {
                    "bbox": tuple(map(int, box)),
                    "conf": float(conf),
                    "score": self._score_detection(tuple(map(int, box)), float(conf), area_weight, conf_weight),
                }
                for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
            ]
            for detection in sorted(detections, key=lambda item: item["score"], reverse=True)[:top_k]:
                x1, y1, x2, y2 = detection["bbox"]
                crop = image[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    crops.append((crop, detection["bbox"], detection["conf"]))

        return crops

    def extract_reference_data(
        self,
        object_images_dir: Path,
        conf_threshold: float,
    ) -> list[dict[str, Any]]:
        """Extract reference crops and their color descriptors."""
        image_files = sorted([path for path in object_images_dir.iterdir() if path.is_file()], key=lambda item: natural_key(item.name))
        references: list[dict[str, Any]] = []

        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            results = self.detector.predict(str(image_path), conf=conf_threshold, verbose=False)
            result = results[0]
            if not result.boxes:
                continue

            best_index = int(result.boxes.conf.argmax())
            best_box = result.boxes.xyxy[best_index].cpu().numpy()
            crop = self._crop_object(image, tuple(best_box))
            if crop is None:
                continue

            color_features = self._extract_dominant_colors(crop)
            if color_features is None:
                continue

            references.append(
                {
                    "crop": crop,
                    "color_features": color_features,
                    "source_image": str(image_path),
                }
            )

        return references

    def sample_video_frames(self, video_path: Path, num_frames: int) -> list[np.ndarray]:
        """Sample evenly spaced frames from the reference video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return []

        frame_indices = np.linspace(0, frame_count - 1, num=max(1, num_frames), dtype=np.int32)
        frames: list[np.ndarray] = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    @staticmethod
    def jitter_brightness_contrast(patch: np.ndarray, brightness_jitter: float = 0.35, contrast_jitter: float = 0.2) -> np.ndarray:
        """Apply lightweight appearance jitter to synthetic pasted crops."""
        brightness = 1.0 + random.uniform(-brightness_jitter, brightness_jitter)
        contrast = 1.0 + random.uniform(-contrast_jitter, contrast_jitter)
        patch_float = (patch.astype(np.float32) - patch.mean()) * contrast + patch.mean()
        return np.clip(patch_float * brightness, 0, 255).astype(np.uint8)

    def create_augmented_backgrounds(
        self,
        video_path: Path,
        object_crops: list[tuple[np.ndarray, tuple[int, int, int, int], float]],
        num_backgrounds: int,
        objects_per_background: int,
        min_scale: float,
        max_scale: float,
        prefer_high_conf: bool,
    ) -> tuple[list[np.ndarray], list[list[tuple[int, int, int, int]]]]:
        """Paste sampled crops onto background frames to create synthetic VPE prompts."""
        if not object_crops:
            raise ValueError("No object crops provided for augmentation.")

        backgrounds = self.sample_video_frames(video_path, num_backgrounds)
        if not backgrounds:
            raise RuntimeError(f"Failed to sample frames from video: {video_path}")

        weights = np.array([crop[2] for crop in object_crops], dtype=np.float32)
        weights = weights / weights.sum() if weights.sum() > 0 else None
        augmented_images: list[np.ndarray] = []
        all_bboxes: list[list[tuple[int, int, int, int]]] = []

        for background in backgrounds:
            height, width, _ = background.shape
            background_copy = background.copy()
            bboxes: list[tuple[int, int, int, int]] = []
            for _ in range(min(objects_per_background, len(object_crops))):
                if prefer_high_conf and weights is not None:
                    crop_index = int(np.random.choice(len(object_crops), p=weights))
                else:
                    crop_index = random.randint(0, len(object_crops) - 1)

                crop, _, _ = object_crops[crop_index]
                augmented_crop = self.jitter_brightness_contrast(crop.copy())
                scale = random.uniform(min_scale, max_scale)
                target_width = max(8, int(round(width * scale)))
                target_height = max(8, int(round(augmented_crop.shape[0] * (target_width / max(1, augmented_crop.shape[1])))))
                if target_width >= width or target_height >= height:
                    continue

                resized = cv2.resize(augmented_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
                x1 = random.randint(0, width - target_width - 1)
                y1 = random.randint(0, height - target_height - 1)
                background_copy[y1 : y1 + target_height, x1 : x1 + target_width] = resized
                bboxes.append((x1, y1, x1 + target_width, y1 + target_height))

            augmented_images.append(background_copy)
            all_bboxes.append(bboxes)

        return augmented_images, all_bboxes

    def create_initial_vpe(
        self,
        augmented_images: list[np.ndarray],
        bboxes_list: list[list[tuple[int, int, int, int]]],
    ) -> torch.Tensor:
        """Average VPE features from several augmented prompt images."""
        initial_vpes: list[torch.Tensor] = []
        for image, bboxes in zip(augmented_images, bboxes_list):
            vpe = self._extract_vpe(image, [list(bbox) for bbox in bboxes])
            if vpe is not None:
                initial_vpes.append(vpe)

        if not initial_vpes:
            raise RuntimeError("Could not generate an initial VPE from augmented images.")
        return torch.mean(torch.cat(initial_vpes, dim=0), dim=0, keepdim=True)

    def preprocess_sample(self, sample_dir: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Any] | None:
        """Preprocess one sample folder into reusable YOLOE reference data."""
        sample_id = sample_dir.name
        class_name = sample_id_to_class_name(sample_id)
        object_images_dir = sample_dir / "object_images"
        video_path = sample_dir / "drone_video.mp4"
        if not object_images_dir.exists() or not video_path.exists():
            print(f"Skipping {sample_id}: missing object_images or drone_video.mp4")
            return None

        object_crops = self.detect_and_extract_objects(
            object_images_dir=object_images_dir,
            conf_threshold=float(config["yolov8_conf"]),
            top_k=int(config["top_k_detections"]),
            area_weight=float(config["detection_area_weight"]),
            conf_weight=float(config["detection_conf_weight"]),
        )
        if not object_crops:
            print(f"Skipping {sample_id}: no augmentation crops found.")
            return None

        reference_data = self.extract_reference_data(object_images_dir, float(config["yolov8_conf"]))
        if not reference_data:
            print(f"Skipping {sample_id}: no valid reference crops found.")
            return None

        augmented_images, bboxes_list = self.create_augmented_backgrounds(
            video_path=video_path,
            object_crops=object_crops,
            num_backgrounds=int(config["num_backgrounds"]),
            objects_per_background=int(config["objects_per_background"]),
            min_scale=float(config["min_aug_scale"]),
            max_scale=float(config["max_aug_scale"]),
            prefer_high_conf=bool(config["prefer_high_conf_crops"]),
        )
        initial_vpe = self.create_initial_vpe(augmented_images, bboxes_list)

        sample_output_dir = ensure_dir(output_dir / sample_id)
        torch.save(initial_vpe.cpu(), sample_output_dir / "initial_vpe.pt")

        reference_crops = np.empty(len(reference_data), dtype=object)
        reference_color_features = np.empty(len(reference_data), dtype=object)
        for index, item in enumerate(reference_data):
            reference_crops[index] = item["crop"]
            reference_color_features[index] = item["color_features"]
        np.save(sample_output_dir / "reference_crops.npy", reference_crops)
        np.save(sample_output_dir / "reference_color_features.npy", reference_color_features)

        if self.siamese_verifier is not None:
            embeddings = self.siamese_verifier.encode_many([item["crop"] for item in reference_data])
            torch.save(embeddings, sample_output_dir / "reference_embeddings.pt")

        metadata = {
            "sample_id": sample_id,
            "class_name": class_name,
            "video_path": str(video_path.resolve()),
            "num_reference_crops": len(reference_data),
            "num_augmented_backgrounds": len(augmented_images),
            "siamese_checkpoint_used": bool(self.siamese_verifier is not None),
        }
        save_json(sample_output_dir / "metadata.json", metadata)
        return metadata

    def preprocess_dataset(self, dataset_path: Path, output_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Preprocess every sample in a dataset split."""
        samples_dir = dataset_path / "samples"
        if not samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

        sample_dirs = sorted([path for path in samples_dir.iterdir() if path.is_dir()], key=lambda item: natural_key(item.name))
        requested_ids = set(config.get("sample_ids", []))
        if requested_ids:
            sample_dirs = [sample_dir for sample_dir in sample_dirs if sample_dir.name in requested_ids]
        limit_samples = int(config.get("limit_samples", 0))
        if limit_samples > 0:
            sample_dirs = sample_dirs[:limit_samples]
        metadata = []
        for sample_dir in sample_dirs:
            print(f"Preprocessing {sample_dir.name}...")
            sample_metadata = self.preprocess_sample(sample_dir, output_dir, config)
            if sample_metadata is not None:
                metadata.append(sample_metadata)

        save_json(output_dir / "dataset_metadata.json", metadata)
        return metadata


def build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI arguments to a preprocessing config."""
    return {
        "yolov8_conf": args.yolov8_conf,
        "top_k_detections": args.top_k_detections,
        "detection_area_weight": args.detection_area_weight,
        "detection_conf_weight": args.detection_conf_weight,
        "num_backgrounds": args.num_backgrounds,
        "objects_per_background": args.objects_per_background,
        "min_aug_scale": args.min_aug_scale,
        "max_aug_scale": args.max_aug_scale,
        "prefer_high_conf_crops": args.prefer_high_conf_crops,
        "limit_samples": args.limit_samples,
        "sample_ids": args.sample_id,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Prepare YOLOE reference data and optional Siamese embeddings.")
    parser.add_argument("--dataset", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed_data/train"))
    parser.add_argument("--yolov8-weights", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--yoloe-weights", type=Path, default=Path("models/yoloe-11s-seg.pt"))
    parser.add_argument("--siamese-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolov8-conf", type=float, default=0.1)
    parser.add_argument("--top-k-detections", type=int, default=1)
    parser.add_argument("--detection-area-weight", type=float, default=0.7)
    parser.add_argument("--detection-conf-weight", type=float, default=0.3)
    parser.add_argument("--num-backgrounds", type=int, default=2)
    parser.add_argument("--objects-per-background", type=int, default=5)
    parser.add_argument("--min-aug-scale", type=float, default=0.05)
    parser.add_argument("--max-aug-scale", type=float, default=0.15)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--prefer-high-conf-crops", dest="prefer_high_conf_crops", action="store_true")
    parser.add_argument("--no-prefer-high-conf-crops", dest="prefer_high_conf_crops", action="store_false")
    parser.set_defaults(prefer_high_conf_crops=True)
    args = parser.parse_args(argv)

    dataset_path = resolve_path(args.dataset)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    preprocessor = YOLOEPreprocessor(
        yolov8_model_path=resolve_path(args.yolov8_weights),
        yoloe_model_path=resolve_path(args.yoloe_weights),
        seed=args.seed,
        siamese_checkpoint=None if args.siamese_checkpoint is None else resolve_path(args.siamese_checkpoint),
    )
    metadata = preprocessor.preprocess_dataset(dataset_path, output_dir, build_config_from_args(args))
    print(f"Preprocessed {len(metadata)} samples into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
