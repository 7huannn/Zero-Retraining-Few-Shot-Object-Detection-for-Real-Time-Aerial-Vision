"""Legacy compatibility wrapper for the main inference pipeline.

This file is intentionally kept so old commands like `python predict.py ...`
still work on `main`. The active pipeline is `inference.py`.
Siamese-specific flags are accepted for backward compatibility but ignored.
"""

from __future__ import annotations

import sys

_DEPRECATED_FLAGS_WITH_VALUE = {
    "--siamese-checkpoint",
    "--siam-threshold",
    "--w-siam",
}

_DEPRECATED_BOOL_FLAGS = {
    "--disable-siamese",
}


def _strip_legacy_flags(argv: list[str]) -> tuple[list[str], list[str]]:
    forwarded: list[str] = []
    ignored: list[str] = []
    index = 0

    while index < len(argv):
        token = argv[index]
        key = token.split("=", 1)[0]

        if key in _DEPRECATED_FLAGS_WITH_VALUE:
            ignored.append(key)
            if "=" not in token and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue

        if token in _DEPRECATED_BOOL_FLAGS:
            ignored.append(token)
            index += 1
            continue

        forwarded.append(token)
        index += 1

    return forwarded, sorted(set(ignored))


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    forwarded_args, ignored_flags = _strip_legacy_flags(raw_args)
    from inference import main as inference_main

    if ignored_flags:
        print(
            "[predict.py] Ignored legacy Siamese flags on main branch: "
            + ", ".join(ignored_flags)
        )

    print("[predict.py] Forwarding to inference.py (main pipeline).")
    return inference_main(forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
