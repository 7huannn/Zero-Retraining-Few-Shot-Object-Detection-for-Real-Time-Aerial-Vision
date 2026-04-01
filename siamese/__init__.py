"""Siamese training and verification utilities."""

from .data import SiamesePairDataset, build_crop_cache, sample_id_to_class_name
from .siamese_model import (
    ContrastiveLoss,
    SiameseDemoRunner,
    SiameseEmbeddingNet,
    SiameseVerifier,
    build_image_transform,
    load_siamese_checkpoint,
    save_siamese_checkpoint,
)

__all__ = [
    "ContrastiveLoss",
    "SiameseDemoRunner",
    "SiameseEmbeddingNet",
    "SiamesePairDataset",
    "SiameseVerifier",
    "build_crop_cache",
    "build_image_transform",
    "load_siamese_checkpoint",
    "sample_id_to_class_name",
    "save_siamese_checkpoint",
]
