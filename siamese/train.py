"""Train a Siamese network on annotated drone crops."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from siamese.data import SiamesePairDataset, build_crop_cache
from siamese.siamese_model import ContrastiveLoss, SiameseEmbeddingNet, cosine_similarity, save_siamese_checkpoint
from utils import ensure_dir, load_json, load_yaml, resolve_path, save_json, seed_everything


def resolve_device(device_value: str | None) -> torch.device:
    """Map CLI/config device values to a torch device."""
    if device_value in (None, "", "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_value.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_value}")
        return torch.device("cpu")
    return torch.device(device_value)


def merge_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply selected CLI overrides to a config dictionary."""
    merged = dict(config)
    if args.epochs is not None:
        merged["epochs"] = args.epochs
    if args.batch_size is not None:
        merged["batch_size"] = args.batch_size
    if args.train_pairs is not None:
        merged["train_pairs"] = args.train_pairs
    if args.val_pairs is not None:
        merged["val_pairs"] = args.val_pairs
    if args.max_crops_per_track is not None:
        merged["max_crops_per_track"] = args.max_crops_per_track
    if args.force_rebuild_cache:
        merged["force_rebuild_cache"] = True
    return merged


def create_loader(dataset: SiamesePairDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    """Create a dataloader with conservative worker defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def run_epoch(
    model: SiameseEmbeddingNet,
    loader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
    optimizer: AdamW | None,
    similarity_threshold: float,
) -> dict[str, float]:
    """Run one train or validation epoch and collect metrics."""
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_cosine = 0.0

    for batch in loader:
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        labels = batch["label"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            embedding_a, embedding_b = model(image_a, image_b)
            loss = criterion(embedding_a, embedding_b, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        cosine_scores = cosine_similarity(embedding_a, embedding_b)
        predictions = (cosine_scores >= similarity_threshold).float()

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_cosine += float(cosine_scores.mean().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were produced for this epoch.")

    return {
        "loss": round(total_loss / total_samples, 6),
        "accuracy": round(total_correct / total_samples, 6),
        "mean_cosine": round(total_cosine / total_samples, 6),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Siamese training."""
    parser = argparse.ArgumentParser(description="Train a Siamese model on drone crops.")
    parser.add_argument("--config", type=Path, default=Path("siamese/train_config.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-pairs", type=int)
    parser.add_argument("--val-pairs", type=int)
    parser.add_argument("--max-crops-per-track", type=int)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    config = merge_overrides(load_yaml(config_path), args)
    seed_everything(int(config.get("seed", 42)))

    annotations_path = resolve_path(config.get("annotations_path", "data/train/annotations/annotations.json"))
    samples_dir = resolve_path(config.get("samples_dir", "data/train/samples"))
    crop_cache_dir = resolve_path(config.get("crop_cache_dir", "preprocessed_data/siamese/crops"))
    manifest_path = resolve_path(config.get("manifest_path", "preprocessed_data/siamese/manifest.json"))
    checkpoint_dir = ensure_dir(resolve_path(config.get("checkpoint_dir", "result/siamese")))

    manifest_file = build_crop_cache(
        annotations_path=annotations_path,
        samples_dir=samples_dir,
        output_dir=crop_cache_dir,
        manifest_path=manifest_path,
        max_crops_per_track=int(config.get("max_crops_per_track", 24)),
        min_crop_size=int(config.get("min_crop_size", 16)),
        overwrite=bool(config.get("force_rebuild_cache", False)),
    )
    manifest = load_json(manifest_file)

    image_size = int(config.get("image_size", 224))
    train_dataset = SiamesePairDataset(
        manifest_path=manifest_file,
        split="train",
        image_size=image_size,
        num_pairs=int(config.get("train_pairs", 6000)),
        train_ratio=float(config.get("train_ratio", 0.8)),
        positive_ratio=float(config.get("positive_ratio", 0.5)),
        hard_negative_ratio=float(config.get("hard_negative_ratio", 0.35)),
        seed=int(config.get("seed", 42)),
    )
    val_dataset = SiamesePairDataset(
        manifest_path=manifest_file,
        split="val",
        image_size=image_size,
        num_pairs=int(config.get("val_pairs", 1500)),
        train_ratio=float(config.get("train_ratio", 0.8)),
        positive_ratio=float(config.get("positive_ratio", 0.5)),
        hard_negative_ratio=float(config.get("hard_negative_ratio", 0.35)),
        seed=int(config.get("seed", 42)),
    )

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    train_loader = create_loader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = create_loader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = resolve_device(str(config.get("device", "auto")))
    model = SiameseEmbeddingNet(
        backbone_name=str(config.get("backbone", "resnet18")),
        embedding_dim=int(config.get("embedding_dim", 128)),
        pretrained_backbone=bool(config.get("pretrained_backbone", True)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)
    criterion = ContrastiveLoss(margin=float(config.get("margin", 1.0)))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(config.get("epochs", 10))))
    similarity_threshold = float(config.get("similarity_threshold", 0.65))

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_val_loss = float("inf")

    for epoch in range(1, int(config.get("epochs", 10)) + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            similarity_threshold=similarity_threshold,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            similarity_threshold=similarity_threshold,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(epoch_metrics)

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_model = copy.deepcopy(model)
    best_model.load_state_dict(best_state)

    used_config = dict(config)
    used_config.update(
        {
            "annotations_path": str(annotations_path),
            "samples_dir": str(samples_dir),
            "crop_cache_dir": str(crop_cache_dir),
            "manifest_path": str(manifest_file),
            "checkpoint_dir": str(checkpoint_dir),
            "image_size": image_size,
        }
    )
    save_siamese_checkpoint(
        checkpoint_dir / "best.pt",
        model=best_model,
        config=used_config,
        metrics=best_metrics,
        extra={"manifest_stats": manifest.get("stats", {})},
    )
    save_siamese_checkpoint(
        checkpoint_dir / "last.pt",
        model=model,
        config=used_config,
        metrics=history[-1],
        extra={"manifest_stats": manifest.get("stats", {})},
    )
    save_json(checkpoint_dir / "history.json", history)
    save_json(checkpoint_dir / "metrics.json", {"best": best_metrics, "last": history[-1], "manifest": manifest.get("stats", {})})

    print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
