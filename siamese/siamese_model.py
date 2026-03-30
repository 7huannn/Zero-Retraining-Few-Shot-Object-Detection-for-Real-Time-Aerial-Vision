"""Simple Siamese-style image similarity runner for smoke tests."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from fsod_drone.utils import ensure_dir


class SiameseEmbeddingNet(nn.Module):
    """Encode images with a shared backbone for pairwise comparison."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return normalized image embeddings."""
        features = self.backbone(image)
        return F.normalize(features, p=2, dim=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode two images with the same weights."""
        return self.encode(image_a), self.encode(image_b)


class SiameseDemoRunner:
    """Compare image pairs with a shared ResNet18 backbone."""

    def __init__(self, device: str | None = None, pretrained_backbone: bool = True) -> None:
        self.device = self._resolve_device(device)
        self.model, self.transform = self._build_model(pretrained_backbone)

    def _resolve_device(self, device: str | None) -> torch.device:
        """Map CLI or YAML device values to a torch device."""
        if device in (None, "", "auto"):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.isdigit():
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            return torch.device("cpu")
        return torch.device(device)

    def _build_model(
        self,
        pretrained_backbone: bool,
    ) -> tuple[SiameseEmbeddingNet, transforms.Compose | Any]:
        """Create a shared backbone and the matching preprocessing pipeline."""
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        try:
            backbone = models.resnet18(weights=weights)
        except Exception as exc:
            warnings.warn(
                f"Falling back to randomly initialized ResNet18 because pretrained weights are unavailable: {exc}",
                stacklevel=2,
            )
            backbone = models.resnet18(weights=None)
            weights = None

        backbone.fc = nn.Identity()
        model = SiameseEmbeddingNet(backbone).to(self.device)
        model.eval()

        if weights is not None:
            transform = weights.transforms()
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        return model, transform

    def _load_tensor(self, image_path: str | Path) -> torch.Tensor:
        """Load one image and move it to the configured device."""
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def compare(self, image_a: str | Path, image_b: str | Path, pair_name: str | None = None) -> dict[str, Any]:
        """Compute cosine similarity and L2 distance for one pair."""
        tensor_a = self._load_tensor(image_a)
        tensor_b = self._load_tensor(image_b)
        embedding_a, embedding_b = self.model(tensor_a, tensor_b)

        cosine_similarity = float(F.cosine_similarity(embedding_a, embedding_b).item())
        l2_distance = float(torch.norm(embedding_a - embedding_b, p=2).item())
        return {
            "pair_name": pair_name or f"{Path(image_a).stem}_vs_{Path(image_b).stem}",
            "image_a": str(image_a),
            "image_b": str(image_b),
            "device": str(self.device),
            "cosine_similarity": round(cosine_similarity, 6),
            "l2_distance": round(l2_distance, 6),
        }

    def compare_many(
        self,
        pairs: list[dict[str, Any]],
        output_dir: str | Path = "outputs/siamese_demo",
    ) -> tuple[list[dict[str, Any]], Path]:
        """Compare many image pairs and persist a JSON report."""
        resolved_output_dir = ensure_dir(output_dir)
        results: list[dict[str, Any]] = []

        for index, pair in enumerate(pairs, start=1):
            results.append(
                self.compare(
                    image_a=pair["image_a"],
                    image_b=pair["image_b"],
                    pair_name=pair.get("name", f"pair_{index}"),
                )
            )

        report_path = resolved_output_dir / "siamese_report.json"
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results, report_path
