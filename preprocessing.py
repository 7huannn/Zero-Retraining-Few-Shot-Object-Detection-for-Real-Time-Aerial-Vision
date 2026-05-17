"""Backward-compatible CLI shim for moved app entrypoint."""

from src.apps.preprocessing import *  # noqa: F401,F403
from src.apps.preprocessing import main


if __name__ == "__main__":
    raise SystemExit(main())
