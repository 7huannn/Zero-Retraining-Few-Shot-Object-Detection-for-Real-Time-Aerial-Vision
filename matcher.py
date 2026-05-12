"""MobileCLIP2 matcher for semantic similarity verification."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _bootstrap_open_clip() -> Any:
    """Dynamically load OpenCLIP from ml-mobileclip submodule."""
    repo_root = Path(__file__).resolve().parent
    open_clip_src = repo_root / "ml-mobileclip" / "open_clip" / "src"
    if open_clip_src.exists() and str(open_clip_src) not in sys.path:
        sys.path.insert(0, str(open_clip_src))
    import open_clip

    return open_clip


open_clip = _bootstrap_open_clip()


class MobileCLIP2Matcher:
    """MobileCLIP2 semantic matcher for few-shot detection."""

    def __init__(
        self,
        model_name: str = "MobileCLIP2-S0",
        encoder_path: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize MobileCLIP2 matcher.

        Args:
            model_name: MobileCLIP2 model name (e.g., 'MobileCLIP2-S0')
            encoder_path: Path to pretrained image encoder weights
            pretrained: Fallback pretrained tag if encoder_path not found
            device: torch device (default: auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.support_embeddings: torch.Tensor | None = None
        self._load_model(encoder_path, pretrained)

    def _load_model(self, encoder_path: str | None, pretrained: str | None) -> None:
        """Load MobileCLIP2 model with weights."""
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=None,
        )

        # Try loading custom encoder weights
        if encoder_path and os.path.exists(encoder_path):
            payload = torch.load(encoder_path, map_location=self.device)
            self.model.visual.load_state_dict(payload)
            print(f"[MobileCLIP2] Loaded image encoder: {encoder_path}")
        elif pretrained:
            print(f"[MobileCLIP2] Fallback to pretrained tag: '{pretrained}'")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=pretrained,
            )
        else:
            raise FileNotFoundError(f"Missing MobileCLIP2 encoder path: {encoder_path}")

        self.model.to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _to_rgb_pil(image_crop: np.ndarray) -> Image.Image:
        """Convert BGR crop to RGB PIL Image."""
        if image_crop.shape[2] == 4:
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def encode_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        """
        Encode a batch of image crops into normalized embeddings.

        Args:
            crops: List of BGR numpy arrays

        Returns:
            Tensor of normalized embeddings [batch, embedding_dim]
        """
        batch = torch.cat(
            [self.preprocess(self._to_rgb_pil(crop)).unsqueeze(0) for crop in crops],
            dim=0,
        ).to(self.device)

        with torch.inference_mode():
            embeddings = self.model.encode_image(batch)
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode_single(self, crop: np.ndarray) -> torch.Tensor | None:
        """
        Encode a single image crop into normalized embedding.

        Args:
            crop: BGR numpy array

        Returns:
            Normalized embedding tensor [1, embedding_dim], or None on error
        """
        try:
            tensor = self.preprocess(self._to_rgb_pil(crop)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                embedding = self.model.encode_image(tensor)
                embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding
        except Exception as e:
            print(f"[MobileCLIP2] Error encoding crop: {e}")
            return None

    def set_support_embeddings(self, crops: list[np.ndarray]) -> None:
        """
        Pre-encode support set for similarity matching.

        Args:
            crops: List of reference image crops
        """
        if not crops:
            raise ValueError("Support set cannot be empty")
        self.support_embeddings = self.encode_batch(crops)
        print(f"[MobileCLIP2] Set support embeddings: {self.support_embeddings.shape}")

    def compute_similarity(self, crop_embedding: torch.Tensor | None) -> float:
        """
        Compute max cosine similarity against support set.

        Args:
            crop_embedding: Single embedding from crop (or None)

        Returns:
            Cosine similarity score in [0, 1]
        """
        if crop_embedding is None or self.support_embeddings is None:
            return 0.0

        # Compute cosine similarity against all support embeddings
        similarities = F.cosine_similarity(
            crop_embedding.expand(self.support_embeddings.shape[0], -1),
            self.support_embeddings,
            dim=1,
        ).detach().cpu().numpy()

        # Return mean similarity (matching aggregation strategy)
        return float(np.mean(similarities))

    def aggregate_scores(self, scores: np.ndarray, mode: str = "mean") -> float:
        """
        Aggregate candidate scores (e.g., mean vs max).

        Args:
            scores: Numpy array of similarity scores
            mode: 'mean' or 'max'

        Returns:
            Aggregated score
        """
        if scores.size == 0:
            return 0.0
        if mode == "max":
            return float(scores.max())
        return float(scores.mean())

    def cosine_to_normalized_score(self, cosine_value: float) -> float:
        """
        Map cosine similarity [-1, 1] to normalized score [0, 1].

        Args:
            cosine_value: Cosine similarity in [-1, 1]

        Returns:
            Normalized score in [0, 1]
        """
        return float(np.clip((cosine_value + 1.0) * 0.5, 0.0, 1.0))
