"""Dataset preparation and pair sampling for Siamese training."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

from siamese.siamese_model import build_image_transform
from utils import ensure_dir, load_json, save_json


def sample_id_to_class_name(sample_id: str) -> str:
    """Strip the numeric suffix from a sample id."""
    return re.sub(r"_\d+$", "", sample_id)


def _evenly_spaced_indices(total: int, limit: int) -> list[int]:
    """Select up to ``limit`` evenly spaced indices from a sequence."""
    if total <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit <= 1:
        return [total // 2]
    return [round(index * (total - 1) / (limit - 1)) for index in range(limit)]


def _valid_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_crop_cache(
    annotations_path: str | Path,
    samples_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    max_crops_per_track: int = 24,
    min_crop_size: int = 16,
    overwrite: bool = False,
) -> Path:
    """Extract annotated training crops and build a manifest for Siamese training."""
    annotations = load_json(annotations_path)
    samples_root = Path(samples_dir)
    crops_root = ensure_dir(output_dir)
    manifest_file = Path(manifest_path) if manifest_path is not None else crops_root / "manifest.json"

    if manifest_file.exists() and not overwrite:
        return manifest_file

    records: list[dict[str, Any]] = []
    crop_count = 0

    for sample in annotations:
        sample_id = str(sample["video_id"])
        class_name = sample_id_to_class_name(sample_id)
        sample_dir = samples_root / sample_id
        video_path = sample_dir / "drone_video.mp4"
        object_images_dir = sample_dir / "object_images"

        if object_images_dir.exists():
            for image_path in sorted(path for path in object_images_dir.iterdir() if _valid_image_file(path)):
                records.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "source": "reference",
                        "track_id": None,
                        "frame": None,
                        "image_path": str(image_path.resolve()),
                    }
                )

        if not video_path.exists():
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            continue

        for track_index, track in enumerate(sample.get("annotations", [])):
            bboxes = list(track.get("bboxes", []))
            for bbox_index in _evenly_spaced_indices(len(bboxes), max_crops_per_track):
                bbox = bboxes[bbox_index]
                frame_id = int(bbox["frame"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                height, width = frame.shape[:2]
                x1 = max(0, min(width - 1, int(bbox["x1"])))
                y1 = max(0, min(height - 1, int(bbox["y1"])))
                x2 = max(x1 + 1, min(width, int(bbox["x2"])))
                y2 = max(y1 + 1, min(height, int(bbox["y2"])))
                if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                track_dir = ensure_dir(crops_root / sample_id / f"track_{track_index:02d}")
                crop_path = track_dir / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                records.append(
                    {
                        "sample_id": sample_id,
                        "class_name": class_name,
                        "source": "track",
                        "track_id": track_index,
                        "frame": frame_id,
                        "image_path": str(crop_path.resolve()),
                    }
                )
                crop_count += 1

        cap.release()

    stats = {
        "num_records": len(records),
        "num_video_crops": crop_count,
        "num_samples": len({record["sample_id"] for record in records}),
        "num_classes": len({record["class_name"] for record in records}),
        "max_crops_per_track": max_crops_per_track,
        "min_crop_size": min_crop_size,
    }
    manifest = {
        "annotations_path": str(Path(annotations_path).resolve()),
        "samples_dir": str(samples_root.resolve()),
        "crops_dir": str(crops_root.resolve()),
        "records": records,
        "stats": stats,
    }
    save_json(manifest_file, manifest)
    return manifest_file


def split_sample_ids(
    sample_ids: list[str],
    train_ratio: float,
    seed: int,
    split_strategy: str = "random",
    loco_val_class: str | None = None,
    loco_fold_index: int = 0,
) -> tuple[list[str], list[str]]:
    """Split sample ids deterministically into train and validation sets."""
    if len(sample_ids) < 2:
        raise ValueError("Need at least two samples to build a Siamese split.")
    strategy = split_strategy.lower()
    if strategy in {"leave_one_class_out", "loco"}:
        class_to_ids: dict[str, list[str]] = {}
        for sample_id in sample_ids:
            class_to_ids.setdefault(sample_id_to_class_name(sample_id), []).append(sample_id)
        classes = sorted(class_to_ids)
        heldout_class = loco_val_class or classes[loco_fold_index % len(classes)]
        if heldout_class not in class_to_ids:
            raise ValueError(f"LOCO class '{heldout_class}' not found in sample ids.")
        val_ids = sorted(class_to_ids[heldout_class])
        train_ids = sorted(sample_id for sample_id in sample_ids if sample_id not in val_ids)
        if len(train_ids) < 2 or len(val_ids) < 2:
            raise ValueError("LOCO split needs at least two train and two validation samples.")
        return train_ids, val_ids

    if strategy == "stratified":
        rng = random.Random(seed)
        class_to_ids: dict[str, list[str]] = {}
        for sample_id in sample_ids:
            class_to_ids.setdefault(sample_id_to_class_name(sample_id), []).append(sample_id)

        val_target = max(2, int(round(len(sample_ids) * (1.0 - train_ratio))))
        val_target = min(val_target, len(sample_ids) - 1)
        class_names = sorted(class_to_ids)
        rng.shuffle(class_names)
        val_ids: list[str] = []
        train_ids: list[str] = []

        for class_name in class_names:
            ids = list(class_to_ids[class_name])
            rng.shuffle(ids)
            if len(val_ids) < val_target and len(ids) > 1:
                val_ids.append(ids[0])
                train_ids.extend(ids[1:])
            else:
                train_ids.extend(ids)

        if len(val_ids) < val_target:
            movable = list(train_ids)
            rng.shuffle(movable)
            move_count = min(val_target - len(val_ids), max(0, len(train_ids) - 1))
            for sample_id in movable[:move_count]:
                train_ids.remove(sample_id)
                val_ids.append(sample_id)
        return sorted(train_ids), sorted(val_ids)

    if strategy != "random":
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    shuffled = list(sample_ids)
    random.Random(seed).shuffle(shuffled)
    val_size = max(2, int(round(len(shuffled) * (1.0 - train_ratio))))
    val_size = min(val_size, len(shuffled) - 1)
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return train_ids, val_ids


@dataclass(frozen=True)
class CropRecord:
    """One image record from a reference image or an extracted training crop."""

    sample_id: str
    class_name: str
    source: str
    track_id: int | None
    frame: int | None
    image_path: str


class SiamesePairDataset(Dataset):
    """Generate positive and negative image pairs from a crop manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int,
        num_pairs: int,
        train_ratio: float = 0.8,
        positive_ratio: float = 0.5,
        hard_negative_ratio: float = 0.35,
        seed: int = 42,
        split_strategy: str = "random",
        loco_val_class: str | None = None,
        loco_fold_index: int = 0,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        manifest = load_json(manifest_path)
        records = [CropRecord(**record) for record in manifest["records"]]
        grouped_records: dict[str, list[CropRecord]] = {}
        for record in records:
            grouped_records.setdefault(record.sample_id, []).append(record)

        train_ids, val_ids = split_sample_ids(
            sorted(grouped_records),
            train_ratio=train_ratio,
            seed=seed,
            split_strategy=split_strategy,
            loco_val_class=loco_val_class,
            loco_fold_index=loco_fold_index,
        )
        selected_ids = train_ids if split == "train" else val_ids
        self.groups = {sample_id: grouped_records[sample_id] for sample_id in selected_ids}
        self.group_ids = sorted(self.groups)
        self.class_to_group_ids: dict[str, list[str]] = {}
        for group_id in self.group_ids:
            class_name = self.groups[group_id][0].class_name
            self.class_to_group_ids.setdefault(class_name, []).append(group_id)

        if len(self.group_ids) < 2:
            raise ValueError(f"Split '{split}' has fewer than two groups.")

        self.split = split
        self.seed = seed
        self.num_pairs = num_pairs
        self.positive_ratio = positive_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.transform = build_image_transform(image_size=image_size, train=(split == "train"))

    def __len__(self) -> int:
        return self.num_pairs

    def _load_image(self, record: CropRecord) -> torch.Tensor:
        with Image.open(record.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor

    def _sample_positive(self, rng: random.Random, group_id: str) -> tuple[CropRecord, CropRecord]:
        records = self.groups[group_id]
        if len(records) == 1:
            return records[0], records[0]
        return tuple(rng.sample(records, 2))  # type: ignore[return-value]

    def _sample_negative_group(self, rng: random.Random, anchor: CropRecord) -> str:
        same_class_groups = [
            group_id
            for group_id in self.class_to_group_ids.get(anchor.class_name, [])
            if group_id != anchor.sample_id
        ]
        if same_class_groups and rng.random() < self.hard_negative_ratio:
            return rng.choice(same_class_groups)

        different_groups = [group_id for group_id in self.group_ids if group_id != anchor.sample_id]
        return rng.choice(different_groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        anchor_group_id = self.group_ids[index % len(self.group_ids)] if self.split == "val" else rng.choice(self.group_ids)
        anchor_records = self.groups[anchor_group_id]
        anchor = rng.choice(anchor_records)
        is_positive = rng.random() < self.positive_ratio

        if is_positive:
            record_a, record_b = self._sample_positive(rng, anchor_group_id)
            label = 1.0
        else:
            negative_group_id = self._sample_negative_group(rng, anchor)
            record_a = anchor
            record_b = rng.choice(self.groups[negative_group_id])
            label = 0.0

        return {
            "image_a": self._load_image(record_a),
            "image_b": self._load_image(record_b),
            "label": torch.tensor(label, dtype=torch.float32),
            "sample_a": record_a.sample_id,
            "sample_b": record_b.sample_id,
            "class_a": record_a.class_name,
            "class_b": record_b.class_name,
            "pair_type": "positive" if is_positive else "negative",
        }
