"""Minimal helpers for the installation and demo stage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root from the package location."""
    return Path(__file__).resolve().parents[1]


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


def detect_optional_package(module_name: str) -> bool:
    """Return True if an optional package can be imported."""
    try:
        __import__(module_name)
        return True
    except Exception:
        return False
