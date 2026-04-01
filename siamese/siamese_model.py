"""Siamese embedding network, checkpoint helpers, and comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms

from utils import ensure_dir, save_json

ImageInput = str | Path | Image.Image | np.ndarray


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    """Resolve CLI-style device values into a valid torch.device."""
    if isinstance(device, torch.device):
        return device
    if device in (None, "", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")
    return torch.device(str(device))


def build_image_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Build a torchvision transform for Siamese training or inference."""
    interpolation = transforms.InterpolationMode.BILINEAR
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15), interpolation=interpolation),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), interpolation=interpolation),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_backbone(backbone_name: str, pretrained_backbone: bool) -> tuple[nn.Module, int]:
    """Construct the requested torchvision backbone and return its feature size."""
    name = backbone_name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    if name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet34(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported backbone: {backbone_name}")


class SiameseEmbeddingNet(nn.Module):
    """Shared-weight embedding network for pairwise image comparison."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 128,
        pretrained_backbone: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.pretrained_backbone = pretrained_backbone
        self.backbone, feature_dim = _build_backbone(backbone_name, pretrained_backbone)
        hidden_dim = max(embedding_dim * 2, 256)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one batch of images into normalized embeddings."""
        features = self.backbone(image)
        embedding = self.projection(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a pair of image batches."""
        return self.encode(image_a), self.encode(image_b)


class ContrastiveLoss(nn.Module):
    """Standard contrastive loss for positive and negative pairs."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distances = F.pairwise_distance(embedding_a, embedding_b)
        positive = labels * distances.pow(2)
        negative = (1.0 - labels) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return (positive + negative).mean()


def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity for two batches of embeddings."""
    return F.cosine_similarity(embedding_a, embedding_b)


def build_model_from_config(config: dict[str, Any]) -> SiameseEmbeddingNet:
    """Create a Siamese model from a config dictionary."""
    return SiameseEmbeddingNet(
        backbone_name=str(config.get("backbone", "resnet18")),
        embedding_dim=int(config.get("embedding_dim", 128)),
        pretrained_backbone=bool(config.get("pretrained_backbone", True)),
        dropout=float(config.get("dropout", 0.1)),
    )


def checkpoint_to_device(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor payload values to the target device when needed."""
    model_state = payload.get("model_state_dict", payload.get("state_dict"))
    if model_state is None:
        raise KeyError("Checkpoint does not contain model weights.")
    payload["model_state_dict"] = model_state
    payload["device"] = str(device)
    return payload


def save_siamese_checkpoint(
    path: Path | str,
    model: SiameseEmbeddingNet,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a training checkpoint with config and metrics."""
    target = Path(path)
    ensure_dir(target.parent)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, target)
    return target


def load_siamese_checkpoint(
    checkpoint_path: Path | str,
    device: str | torch.device | None = None,
) -> tuple[SiameseEmbeddingNet, dict[str, Any]]:
    """Load a trained Siamese model checkpoint."""
    resolved_device = resolve_torch_device(device)
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    payload = checkpoint_to_device(payload, resolved_device)
    config = dict(payload.get("config", {}))
    model = build_model_from_config(config).to(resolved_device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


@dataclass
class SiameseComparison:
    """Structured result for a pairwise comparison."""

    image_a: str
    image_b: str
    cosine_similarity: float
    l2_distance: float


class SiameseVerifier:
    """Helper for encoding crops and comparing candidates against references."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: str | None = None,
        image_size: int | None = None,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.model, payload = load_siamese_checkpoint(checkpoint_path, self.device)
        config = dict(payload.get("config", {}))
        self.image_size = image_size or int(config.get("image_size", 224))
        self.transform = build_image_transform(self.image_size, train=False)

    def _prepare_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as handle:
                return handle.convert("RGB").copy()
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("Expected an HWC image array.")
            if image.shape[2] == 4:
                return Image.fromarray(image[:, :, :3][:, :, ::-1]).convert("RGB")
            return Image.fromarray(image[:, :, ::-1]).convert("RGB")
        raise TypeError(f"Unsupported image input: {type(image)!r}")

    def _to_tensor(self, image: ImageInput) -> torch.Tensor:
        pil_image = self._prepare_image(image)
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def encode(self, image: ImageInput) -> torch.Tensor:
        """Encode one image and return a CPU tensor."""
        tensor = self._to_tensor(image)
        embedding = self.model.encode(tensor)
        return embedding.cpu()

    @torch.inference_mode()
    def encode_many(self, images: Sequence[ImageInput]) -> torch.Tensor:
        """Encode a list of images into a single tensor."""
        if not images:
            raise ValueError("Expected at least one image to encode.")
        batch = torch.cat([self._to_tensor(image) for image in images], dim=0)
        embeddings = self.model.encode(batch)
        return embeddings.cpu()

    @torch.inference_mode()
    def compare(self, image_a: ImageInput, image_b: ImageInput) -> SiameseComparison:
        """Compare two images with the trained Siamese network."""
        embedding_a = self.encode(image_a)
        embedding_b = self.encode(image_b)
        cosine = float(cosine_similarity(embedding_a, embedding_b).item())
        distance = float(F.pairwise_distance(embedding_a, embedding_b).item())
        return SiameseComparison(
            image_a=str(image_a),
            image_b=str(image_b),
            cosine_similarity=round(cosine, 6),
            l2_distance=round(distance, 6),
        )

    @torch.inference_mode()
    def score_candidate(self, references: Sequence[ImageInput], candidate: ImageInput) -> dict[str, float]:
        """Compare one candidate image against a reference gallery."""
        reference_embeddings = self.encode_many(references)
        candidate_embedding = self.encode(candidate)
        cosine_scores = cosine_similarity(reference_embeddings, candidate_embedding.expand_as(reference_embeddings))
        distances = F.pairwise_distance(reference_embeddings, candidate_embedding.expand_as(reference_embeddings))
        return {
            "cosine_mean": round(float(cosine_scores.mean().item()), 6),
            "cosine_max": round(float(cosine_scores.max().item()), 6),
            "l2_mean": round(float(distances.mean().item()), 6),
            "l2_min": round(float(distances.min().item()), 6),
        }


class SiameseDemoRunner:
    """Compare image pairs using either a checkpoint or a plain backbone."""

    def __init__(
        self,
        device: str | None = None,
        pretrained_backbone: bool = True,
        checkpoint_path: str | Path | None = None,
        backbone_name: str = "resnet18",
        embedding_dim: int = 128,
        image_size: int = 224,
    ) -> None:
        self.device = resolve_torch_device(device)
        self.image_size = image_size
        if checkpoint_path is not None:
            self.model, payload = load_siamese_checkpoint(checkpoint_path, self.device)
            config = dict(payload.get("config", {}))
            self.image_size = int(config.get("image_size", image_size))
        else:
            self.model = SiameseEmbeddingNet(
                backbone_name=backbone_name,
                embedding_dim=embedding_dim,
                pretrained_backbone=pretrained_backbone,
            ).to(self.device)
            self.model.eval()
        self.transform = build_image_transform(self.image_size, train=False)

    def _load_tensor(self, image_path: str | Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def compare(self, image_a: str | Path, image_b: str | Path, pair_name: str | None = None) -> dict[str, Any]:
        """Compute cosine similarity and distance for one pair."""
        tensor_a = self._load_tensor(image_a)
        tensor_b = self._load_tensor(image_b)
        embedding_a, embedding_b = self.model(tensor_a, tensor_b)
        return {
            "pair_name": pair_name or f"{Path(image_a).stem}_vs_{Path(image_b).stem}",
            "image_a": str(image_a),
            "image_b": str(image_b),
            "device": str(self.device),
            "cosine_similarity": round(float(cosine_similarity(embedding_a, embedding_b).item()), 6),
            "l2_distance": round(float(F.pairwise_distance(embedding_a, embedding_b).item()), 6),
        }

    def compare_many(
        self,
        pairs: Sequence[dict[str, Any]],
        output_dir: str | Path = "result/siamese",
        report_name: str = "siamese_report.json",
    ) -> tuple[list[dict[str, Any]], Path]:
        """Compare many pairs and write a JSON report."""
        results = [
            self.compare(
                image_a=pair["image_a"],
                image_b=pair["image_b"],
                pair_name=pair.get("name"),
            )
            for pair in pairs
        ]
        output_path = ensure_dir(output_dir) / report_name
        save_json(output_path, results)
        return results, output_path
