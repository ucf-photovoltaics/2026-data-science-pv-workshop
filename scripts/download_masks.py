"""Fetch the precomputed EL segmentation masks (optional, host-agnostic).

The full pipeline is precomputed once and the resulting *restitched module masks*
(one per EL image, defects overlaid) are published as a Release asset so students
can browse segmentation results without a GPU. This is optional — the modeling
section only needs ``metadata_with_el_features.csv`` (committed).

    from scripts.download_masks import download_masks
    download_masks()   # unzips into data/workshop/images/05_module_restitched/

Only the standard library is used (works in a bare Colab runtime).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

OWNER = "ucf-photovoltaics"
REPO = "2026-data-science-pv-workshop"
TAG = "v0.1.0"
ASSET_NAME = "el_precomputed_masks.zip"
MASKS_URL = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{ASSET_NAME}"

# Filled in once the asset is built/uploaded (shasum -a 256 el_precomputed_masks.zip).
EXPECTED_SHA256 = "61b46079821c6c94999f3371548f65bf440b73b151ef16e755b08b1d09799c77"
OUTPUT_SUBDIR = Path("data/workshop/images/05_module_restitched")
EXPECTED_PNG_COUNT = 102


def _find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for c in (here, *here.parents):
        if (c / "pyproject.toml").exists():
            return c
    return here


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _png_count(root: Path) -> int:
    return len(list((root / OUTPUT_SUBDIR).rglob("*.png")))


def download_masks(url: str = MASKS_URL, dest_root: str | Path | None = None,
                   expected_sha256: str | None = EXPECTED_SHA256,
                   force: bool = False, verify: bool = True) -> Path:
    """Download and unpack the precomputed restitched masks."""
    root = Path(dest_root).resolve() if dest_root else _find_repo_root()
    target = root / OUTPUT_SUBDIR

    if _png_count(root) >= EXPECTED_PNG_COUNT and not force:
        print(f"✓ masks already present at {target}")
        return target

    print(f"Downloading precomputed masks from:\n  {url}")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / ASSET_NAME
        with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted URL)
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(zip_path, "wb") as out:
                while (chunk := resp.read(1 << 20)):
                    out.write(chunk)
                    done += len(chunk)
                    pct = f"({100*done/total:5.1f}%)" if total else ""
                    print(f"\r  {done/1e6:6.1f} MB {pct}", end="", flush=True)
            print()
        if verify and expected_sha256 and not expected_sha256.startswith("__"):
            actual = _sha256(zip_path)
            if actual != expected_sha256:
                raise ValueError(f"Checksum mismatch.\n  expected {expected_sha256}\n"
                                 f"  actual   {actual}")
            print("✓ checksum verified")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    print(f"✓ masks ready at {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the precomputed EL masks.")
    p.add_argument("--url", default=MASKS_URL)
    p.add_argument("--dest", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args(argv)
    try:
        download_masks(url=a.url, dest_root=a.dest, force=a.force, verify=not a.no_verify)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
