"""Configuration classes for few-shot detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""

    # Detection
    yoloe_conf: float = 0.001
    top_k_proposals: int = 24

    # Frame selection
    frame_start: int = 0
    frame_end: int = 0
    max_frames: int = 0

    # Tracker
    tracker_reinit_interval: int = 10
    edge_proximity_threshold: int = 10

    # Matcher aggregation
    support_aggregation: str = "mean"

    # Score fusion weights
    w_det: float = 0.30
    # Kept as w_clip for CLI/backward compatibility; used as the selected matcher weight.
    w_clip: float = 0.35

    # Thresholds
    fused_accept_threshold: float = 0.52
    # Kept as min_clip_similarity for CLI/backward compatibility; used as selected matcher threshold.
    min_clip_similarity: float = 0.10
    similarity_add_threshold: float = 0.80
    tracker_bonus: float = 0.04

    # Reference gallery
    max_reference_samples: int = 20
    crop_padding_ratio: float = 0.04

    # Output
    save_frames: bool = False
    save_frame_interval: int = 25
    frames_output_dir: str = "result/debug_frames"
    generate_report: bool = True
    generate_plots: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        # Weights don't need to sum to 1.0 as fusion function clips to [0, 1]
        if self.w_det < 0 or self.w_clip < 0:
            raise ValueError("Weights must be non-negative")


@dataclass
class ModelConfig:
    """Configuration for model paths and initialization."""

    # YOLOE
    yoloe_weights: str = "models/yoloe-11l-seg.pt"

    # MobileCLIP2
    matcher_type: str = "mobileclip2"
    clip_model_name: str = "MobileCLIP2-S0"
    clip_encoder_path: str = "models/mobileclip2_image_encoder_fp16.pt"
    clip_pretrained: str | None = "dfndr2b"

    # Siamese
    siamese_checkpoint: str | None = "result/siamese/best.pt"

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0", etc.

    def get_device(self) -> str:
        """Get resolved device string."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


@dataclass
class DataConfig:
    """Configuration for data paths."""

    preprocessed_dir: str = "preprocessed_data/public_test"
    output_json: str = "result/submission.json"
    limit_videos: int = 0  # 0 = no limit

    def get_preprocessed_path(self) -> Path:
        """Get preprocessed data directory as Path."""
        return Path(self.preprocessed_dir)

    def get_output_path(self) -> Path:
        """Get output JSON path as Path."""
        return Path(self.output_json)


@dataclass
class PipelineConfig:
    """Combined configuration for entire pipeline."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_args(cls, args: object) -> PipelineConfig:
        """Create config from argparse Namespace."""
        # Extract inference config args
        inference_kwargs = {}
        for field_name in InferenceConfig.__dataclass_fields__:
            if hasattr(args, field_name):
                inference_kwargs[field_name] = getattr(args, field_name)

        # Extract model config args
        model_kwargs = {}
        for field_name in ModelConfig.__dataclass_fields__:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                # Handle custom arg names
                if field_name == "yoloe_weights" and hasattr(args, "yoloe_weights"):
                    model_kwargs[field_name] = value
                elif field_name == "matcher_type" and hasattr(args, "matcher"):
                    model_kwargs[field_name] = getattr(args, "matcher")
                elif field_name == "clip_model_name" and hasattr(args, "clip_model_name"):
                    model_kwargs[field_name] = value
                elif field_name == "clip_encoder_path" and hasattr(args, "clip_encoder_path"):
                    model_kwargs[field_name] = value
                elif field_name == "clip_pretrained" and hasattr(args, "clip_pretrained"):
                    model_kwargs[field_name] = value
                elif field_name == "siamese_checkpoint" and hasattr(args, "siamese_checkpoint"):
                    model_kwargs[field_name] = value
                elif field_name == "device" and hasattr(args, "device"):
                    model_kwargs[field_name] = value
        if hasattr(args, "matcher"):
            model_kwargs["matcher_type"] = getattr(args, "matcher")

        # Extract data config args
        data_kwargs = {}
        for field_name in DataConfig.__dataclass_fields__:
            if hasattr(args, field_name):
                data_kwargs[field_name] = getattr(args, field_name)

        return cls(
            inference=InferenceConfig(**inference_kwargs),
            model=ModelConfig(**model_kwargs),
            data=DataConfig(**data_kwargs),
        )
