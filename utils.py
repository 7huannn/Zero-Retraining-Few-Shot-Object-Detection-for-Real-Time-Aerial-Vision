"""Shared helpers for data preparation, training, and inference."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist and return it."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a path relative to the repository root or an explicit base dir."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    anchor = base_dir if base_dir is not None else repo_root()
    return (anchor / path).resolve()


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path | str, payload: dict[str, Any]) -> None:
    """Write a YAML payload to disk."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_json(path: Path | str) -> Any:
    """Load a JSON document from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path | str, payload: Any) -> None:
    """Persist a JSON document with stable formatting."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def detect_optional_package(module_name: str) -> bool:
    """Return True if an optional package can be imported."""
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def seed_everything(seed: int = 42) -> None:
    """Seed common RNG sources used by the pipeline."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
