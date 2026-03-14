# ======================================================================
# PREPROCESSING STEP - preprocessing.py (YOLOv8 + YOLOE ONLY)
# ======================================================================
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)


class DataPreprocessor:
    """Handles preprocessing using YOLOv8 detection and YOLOE VPE extraction."""

    def __init__(self,
                 yolov8_model_path: str = "yolov8n.pt",
                 yoloe_model_path: str = "yoloe-11s-seg.pt",
                 seed: Optional[int] = None):
        """Initialize YOLOv8 and YOLOE models."""
        self.yolov8_model_path = yolov8_model_path
        self.yoloe_model_path = yoloe_model_path
        self.seed = seed
        if self.seed is not None:
            self._set_seed(self.seed)

        print("Initializing preprocessing models...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.detector = YOLO(self.yolov8_model_path)
        self.yoloe_model = YOLOE(self.yoloe_model_path)
        print("YOLO preprocessing models initialized successfully.")

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        cv2.setRNGSeed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")

    @staticmethod
    def _natural_key(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    @staticmethod
    def _crop_object(image: np.ndarray, bbox: Tuple[float, float, float, float],
                     mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x1 >= x2 or y1 >= y2: return None
        
        crop = image[y1:y2, x1:x2]
        if mask is None: return crop
        
        full_mask = mask.squeeze() if mask.ndim == 3 else mask
        mask_crop = full_mask[y1:y2, x1:x2]
        binary_mask = (mask_crop > 0.4).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        alpha_channel = dilated_mask * 255
        
        if alpha_channel.shape[:2] != crop.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        BGRa_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        BGRa_crop[:, :, 3] = alpha_channel
        return BGRa_crop


    def _extract_vpe(self, image: np.ndarray, bboxes: List[List[int]]) -> Optional[torch.Tensor]:
        self.yoloe_model.predictor = None
        if not bboxes: return None
        temp_img_path = f"temp_frame_for_vpe_{random.randint(0, 99999)}.jpg"
        cv2.imwrite(temp_img_path, image)
        try:
            visual_prompts = {'bboxes': [np.array(bboxes)], 'cls': [np.array([0] * len(bboxes))]}
            self.yoloe_model.predict(
                temp_img_path, prompts=visual_prompts, predictor=YOLOEVPSegPredictor,
                return_vpe=True, verbose=False
            )
            vpe = self.yoloe_model.predictor.vpe
        finally:
            self.yoloe_model.predictor = None
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        return vpe

    def extract_reference_data(self, object_images_dir: str, 
                               conf_threshold: float = 0.1) -> List[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Extracts crop and dominant color features from original sample images.
        """
        reference_data = []
        image_files = [f for f in os.listdir(object_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in sorted(image_files, key=self._natural_key):
            img_path = os.path.join(object_images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            results = self.detector.predict(img_path, conf=conf_threshold, verbose=False)
            result = results[0]
            if result.boxes:
                best_box_coords = result.boxes.xyxy[result.boxes.conf.argmax()].cpu().numpy()
                crop = self._crop_object(img, best_box_coords)

                if crop is not None:
                    color_features = self._extract_dominant_colors(crop)

                    if color_features is not None:
                        reference_data.append((crop, color_features))

        print(f"     Extracted {len(reference_data)} reference sets (crop + color features).")
        return reference_data

    def _score_detection(self, bbox: Tuple[int, int, int, int], confidence: float,
                        area_weight: float, conf_weight: float) -> float:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return area_weight * min(1.0, area / (640 * 640)) + conf_weight * confidence

    def detect_and_extract_objects(self, object_images_dir: str, conf_threshold: float,
                                   top_k: int, area_weight: float, 
                                   conf_weight: float) -> List[Tuple[np.ndarray, Tuple, float]]:
        """Detects objects and extracts the top-k scored CROPS for augmentation."""
        crops = []
        image_files = [f for f in os.listdir(object_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"     Found {len(image_files)} object images.")
        for img_file in sorted(image_files, key=self._natural_key):
            img_path = os.path.join(object_images_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = self.detector.predict(img_path, save=False, conf=conf_threshold, verbose=False)
            result = results[0]
            if result.boxes:
                detections = [
                    {'bbox': tuple(map(int, box)), 'conf': conf, 
                     'score': self._score_detection(tuple(map(int, box)), conf, area_weight, conf_weight)}
                    for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
                ]
                for det in sorted(detections, key=lambda x: x['score'], reverse=True)[:top_k]:
                    x1, y1, x2, y2 = det['bbox']
                    crop = img[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        crops.append((crop, det['bbox'], det['conf']))
        print(f"     Extracted {len(crops)} top-quality object crops for augmentation.")
        return crops

    def sample_video_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def jitter_brightness_contrast(self, patch: np.ndarray, jb: float = 0.35, 
                                   jc: float = 0.2) -> np.ndarray:
        b_factor = 1.0 + random.uniform(-jb, jb)
        c_factor = 1.0 + random.uniform(-jc, jc)
        patch_f = (patch.astype(np.float32) - patch.mean()) * c_factor + patch.mean()
        patch_f = np.clip(patch_f * b_factor, 0, 255).astype(np.uint8)
        return patch_f


    def create_initial_vpe(self, augmented_images: List[np.ndarray], 
                          bboxes_list: List[List[Tuple]]) -> torch.Tensor:
        """Creates initial VPE from augmented images."""
        initial_vpes = []
        for aug_img, bboxes in zip(augmented_images, bboxes_list):
            vpe = self._extract_vpe(aug_img, bboxes)
            if vpe is not None:
                initial_vpes.append(vpe)
        
        if not initial_vpes:
            raise RuntimeError("Could not generate an initial VPE from augmented images.")
        
        initial_vpe = torch.mean(torch.cat(initial_vpes, dim=0), dim=0, keepdim=True)
        return initial_vpe

    def extract_class_name_from_folder(self, folder_name: str) -> str:
        return re.sub(r'_\d+$', '', folder_name)

    def preprocess_video(self, video_dir: Path, output_dir: Path, config: Dict) -> Dict:
        video_id = video_dir.name
        print(f"\nPreprocessing: {video_id}")
        
        class_name = self.extract_class_name_from_folder(video_id)
        object_images_dir = video_dir / "object_images"
        video_file = video_dir / "drone_video.mp4"

        if not (object_images_dir.exists() and video_file.exists()):
            print(f"  -> WARNING: Missing files for {video_id}. Skipping.")
            return None

        try:
            print("  1. Extracting object crops for augmentation...")
            object_crops = self.detect_and_extract_objects(
                str(object_images_dir), config['yolov8_conf'], config['top_k_detections'],
                config['detection_area_weight'], config['detection_conf_weight']
            )
            if not object_crops:
                print("  -> WARNING: No objects detected for augmentation. Skipping.")
                return None

            print("  2. Extracting reference data (crops, colors)...")
            reference_data = self.extract_reference_data(str(object_images_dir), config['yolov8_conf'])
            if not reference_data:
                print("  -> WARNING: No valid reference data could be extracted. Skipping.")
                return None

            print("  3. Creating augmented backgrounds...")
            augmented_images, bboxes_list = self.create_augmented_backgrounds(
                str(video_file), object_crops, config['num_backgrounds'], config['objects_per_background'],
                config['min_aug_scale'], config['max_aug_scale'], config['prefer_high_conf_crops']
            )

            print("  4. Creating initial VPE...")
            initial_vpe = self.create_initial_vpe(augmented_images, bboxes_list)

            # Save preprocessed data
            video_output_dir = output_dir / video_id
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(initial_vpe, video_output_dir / "initial_vpe.pt")
            
            # Unpack the reference data for saving
            ref_crops = [crop for crop, _ in reference_data]
            ref_color_features = [colors for _, colors in reference_data]

            np.save(video_output_dir / "reference_crops.npy", np.array(ref_crops, dtype=object))
            color_features_to_save = np.empty(len(ref_color_features), dtype=object)
            for i, item in enumerate(ref_color_features):
                color_features_to_save[i] = item
            
            np.save(video_output_dir / "reference_color_features.npy", color_features_to_save)
            
            metadata = {
                'video_id': video_id, 'class_name': class_name, 'video_path': str(video_file),
                'num_references': len(reference_data)
            }
            with open(video_output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"  ✓ Preprocessing complete for {video_id}")
            return metadata

        except Exception as e:
            print(f"  -> ERROR preprocessing {video_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_dataset(self, dataset_path: str, output_dir: str, config: Dict):
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted([d for d in (dataset_path / "samples").iterdir() if d.is_dir()])
        print(f"Found {len(video_dirs)} video directories to preprocess.")

        all_metadata = [md for video_dir in video_dirs if (md := self.preprocess_video(video_dir, output_dir, config)) is not None]

        with open(output_dir / "dataset_metadata.json", 'w') as f:
            json.dump(all_metadata, f, indent=4)
        
        print(f"\n{'='*80}\n✓ Preprocessing complete! Data saved to: {output_dir}")
        return all_metadata


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        yolov8_model_path="models/yolov8n.pt",
        yoloe_model_path="models/yoloe-11l-seg.pt",
        seed=42
    )

    config = {
        'yolov8_conf': 0.1, 'top_k_detections': 1, 'detection_area_weight': 0.7,
        'detection_conf_weight': 0.3, 'num_backgrounds': 1, 'objects_per_background': 5,
        'min_aug_scale': 0.05, 'max_aug_scale': 0.15, 'prefer_high_conf_crops': True,
    }

    dataset_path = "/data/"
    output_dir = "preprocessed_data"

    preprocessor.preprocess_dataset(dataset_path, output_dir, config)