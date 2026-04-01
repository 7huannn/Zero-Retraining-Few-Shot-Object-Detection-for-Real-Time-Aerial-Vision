"""Install the official YOLOE repository and its editable dependencies."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import repo_root


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run one subprocess command and stream output."""
    print("$", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def ensure_git_available() -> None:
    """Fail fast if git is not available."""
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone the official YOLOE repository.")


def clone_or_update_repo(repo_url: str, target_dir: Path, branch: str, refresh: bool) -> None:
    """Clone the repository if absent, or refresh it when requested."""
    ensure_git_available()
    if target_dir.exists() and (target_dir / ".git").exists():
        if refresh:
            run_command(["git", "fetch", "origin", branch], cwd=target_dir)
            run_command(["git", "checkout", branch], cwd=target_dir)
            run_command(["git", "pull", "--ff-only", "origin", branch], cwd=target_dir)
        else:
            print(f"Using existing YOLOE checkout at {target_dir}")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(target_dir)])


def install_requirements(project_root: Path, yoloe_root: Path, install_core_deps: bool) -> None:
    """Install project requirements and the editable YOLOE stack."""
    python = sys.executable
    if install_core_deps:
        run_command([python, "-m", "pip", "install", "-r", str(project_root / "requirements.txt")])
    run_command([python, "-m", "pip", "install", "-r", "requirements.txt"], cwd=yoloe_root)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Install the official YOLOE repository for this project.")
    parser.add_argument("--repo-url", default="https://github.com/THU-MIG/yoloe.git")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--target-dir", type=Path, default=Path("third_party/yoloe"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-core-deps", action="store_true")
    args = parser.parse_args(argv)

    project_root = repo_root()
    target_dir = args.target_dir if args.target_dir.is_absolute() else project_root / args.target_dir
    clone_or_update_repo(args.repo_url, target_dir, args.branch, args.refresh)
    install_requirements(project_root, target_dir, install_core_deps=not args.skip_core_deps)
    print(f"YOLOE installed from {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
