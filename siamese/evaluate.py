"""Evaluate a trained Siamese checkpoint on validation pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from siamese.data import SiamesePairDataset
from siamese.siamese_model import ContrastiveLoss, TripletBatchHardLoss, cosine_similarity, load_siamese_checkpoint
from utils import ensure_dir, load_json, load_yaml, resolve_path, save_json


def create_loader(dataset: SiamesePairDataset, batch_size: int, num_workers: int) -> DataLoader:
    """Create a validation dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def build_criterion(config: dict[str, Any]) -> tuple[nn.Module, str]:
    """Build the same validation loss used by training."""
    loss_name = str(config.get("loss", "contrastive")).lower()
    margin = float(config.get("margin", 1.0))
    if loss_name == "contrastive":
        return ContrastiveLoss(margin=margin), loss_name
    if loss_name in {"triplet", "batch_hard_triplet"}:
        return TripletBatchHardLoss(margin=margin), "triplet"
    raise ValueError(f"Unsupported Siamese loss: {loss_name}")


def batch_sample_labels(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Map the sample ids present in one batch to compact integer labels."""
    sample_ids = list(batch["sample_a"]) + list(batch["sample_b"])
    mapping: dict[str, int] = {}
    label_ids: list[int] = []
    for sample_id in sample_ids:
        if sample_id not in mapping:
            mapping[sample_id] = len(mapping)
        label_ids.append(mapping[sample_id])
    return torch.tensor(label_ids, dtype=torch.long, device=device)


def classification_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute binary precision/recall/F1 for pair verification."""
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def evaluate(
    checkpoint_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a checkpoint and return aggregate metrics."""
    model, payload = load_siamese_checkpoint(checkpoint_path, config.get("device", "auto"))
    checkpoint_config = dict(payload.get("config", {}))
    merged_config = dict(checkpoint_config)
    merged_config.update(config)

    manifest_path = resolve_path(merged_config.get("manifest_path", "preprocessed_data/siamese/manifest.json"))
    manifest = load_json(manifest_path)
    val_dataset = SiamesePairDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=int(merged_config.get("image_size", 224)),
        num_pairs=int(merged_config.get("val_pairs", 1500)),
        train_ratio=float(merged_config.get("train_ratio", 0.8)),
        positive_ratio=float(merged_config.get("positive_ratio", 0.5)),
        hard_negative_ratio=float(merged_config.get("hard_negative_ratio", 0.35)),
        seed=int(merged_config.get("seed", 42)),
        split_strategy=str(merged_config.get("split_strategy", "random")),
        loco_val_class=merged_config.get("loco_val_class"),
        loco_fold_index=int(merged_config.get("loco_fold_index", 0)),
    )
    loader = create_loader(
        dataset=val_dataset,
        batch_size=int(merged_config.get("batch_size", 32)),
        num_workers=int(merged_config.get("num_workers", 0)),
    )
    criterion, loss_name = build_criterion(merged_config)
    criterion = criterion.to(next(model.parameters()).device)
    similarity_threshold = float(merged_config.get("similarity_threshold", 0.65))
    device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_cosine = 0.0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.inference_mode():
        for batch in loader:
            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            labels = batch["label"].to(device)
            embedding_a, embedding_b = model(image_a, image_b)
            if loss_name == "triplet":
                loss = criterion(torch.cat([embedding_a, embedding_b], dim=0), batch_sample_labels(batch, device))
            else:
                loss = criterion(embedding_a, embedding_b, labels)
            cosine_scores = cosine_similarity(embedding_a, embedding_b)
            predictions = (cosine_scores >= similarity_threshold).float()
            predicted_positive = predictions.bool()
            actual_positive = labels.bool()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_cosine += float(cosine_scores.mean().item()) * batch_size
            total_samples += batch_size
            true_positive += int((predicted_positive & actual_positive).sum().item())
            false_positive += int((predicted_positive & ~actual_positive).sum().item())
            false_negative += int((~predicted_positive & actual_positive).sum().item())

    if total_samples == 0:
        raise RuntimeError("Validation dataset produced zero samples.")

    metrics = {
        "loss_name": loss_name,
        "loss": round(total_loss / total_samples, 6),
        "accuracy": round(total_correct / total_samples, 6),
        "mean_cosine": round(total_cosine / total_samples, 6),
    }
    metrics.update(classification_metrics(true_positive, false_positive, false_negative))
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "manifest": manifest.get("stats", {}),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Siamese checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("result/siamese/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("siamese/train_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("result/siamese/evaluation.json"))
    args = parser.parse_args(argv)

    config = load_yaml(resolve_path(args.config))
    report = evaluate(resolve_path(args.checkpoint), config)
    output_path = ensure_dir(resolve_path(args.output).parent) / Path(args.output).name
    save_json(output_path, report)
    print(f"Evaluation report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
