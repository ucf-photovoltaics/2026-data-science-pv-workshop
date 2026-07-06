"""Fetch the EL defect-segmentation model weights (host-agnostic).

The DeepLabV3-ResNet50 checkpoint ``model_97.pth`` (~336 MB, 5 defect classes)
is distributed as a Release asset on the workshop repo, mirrored from
ucf-photovoltaics/UCF-EL-Defect so students don't hit that repo's Git-LFS
bandwidth limits. This downloads it into ``data/workshop/models/``.

Like the image downloader, this is host-agnostic (takes a URL), stdlib-only, and
verifies a SHA-256 checksum.

    from download_weights import download_weights
    download_weights()                    # from the workshop release
    download_weights(url="https://.../model_97.pth")   # any host
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

OWNER = "ucf-photovoltaics"
REPO = "2026-data-science-pv-workshop"
TAG = "v0.1.0"
ASSET_NAME = "model_97.pth"
WEIGHTS_URL = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{ASSET_NAME}"

EXPECTED_SHA256 = "7b622a54a60e0ea755a3f454c8f55dab2624cc175b590e2b7228224cb9774902"
EXPECTED_BYTES = 336424708
DEST_SUBPATH = Path("data/workshop/models") / ASSET_NAME


def _find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_weights(
    url: str = WEIGHTS_URL,
    dest: str | Path | None = None,
    expected_sha256: str | None = EXPECTED_SHA256,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Download ``model_97.pth`` to ``dest`` (default: data/workshop/models/)."""
    dest = Path(dest) if dest else _find_repo_root() / DEST_SUBPATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES and not force:
        print(f"✓ weights already present at {dest}")
        return dest

    print(f"Downloading model weights from:\n  {url}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted URL)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                pct = f"({100*done/total:5.1f}%)" if total else ""
                print(f"\r  {done/1e6:6.1f} MB {pct}", end="", flush=True)
        print()

    if verify and expected_sha256:
        actual = _sha256(dest)
        if actual != expected_sha256:
            raise ValueError(
                f"Checksum mismatch.\n  expected {expected_sha256}\n  actual   {actual}"
            )
        print("✓ checksum verified")
    print(f"✓ weights ready at {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the EL segmentation weights.")
    p.add_argument("--url", default=WEIGHTS_URL)
    p.add_argument("--dest", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args(argv)
    try:
        download_weights(url=args.url, dest=args.dest, force=args.force,
                         verify=not args.no_verify)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
