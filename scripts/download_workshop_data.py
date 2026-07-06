"""Fetch the workshop EL image set from a release asset (host-agnostic).

The raw electroluminescence (EL) images (~177 MB) are distributed as a single
zip attached to a GitHub Release rather than committed to git, so cloning the
repo stays fast. This module downloads that zip and unpacks it back into
``data/workshop/images/01_input/<module>/``.

The downloader is host-agnostic: it takes a plain URL, so the same code works
against a GitHub Release asset now, or an OSF / Zenodo file URL later — just
change ``DATA_URL`` (or pass ``url=...``).

Usage
-----
Python / notebook:
    from download_workshop_data import download_data
    download_data()                        # uses DATA_URL below
    download_data(url="https://osf.io/<id>/download")   # different host later

Command line (from the repo root):
    python scripts/download_workshop_data.py
    python scripts/download_workshop_data.py --url <URL> --force

Only the Python standard library is used, so it runs in a bare Colab runtime
with no extra installs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# --- Release location -------------------------------------------------------
# Point this at the release asset. Update OWNER/REPO/TAG once the repo is public
# and the release is cut, or override --url on the command line.
OWNER = "ucf-photovoltaics"
REPO = "2026-data-science-pv-workshop"
TAG = "v0.1.0"                       # pin data to a release tag for reproducibility
ASSET_NAME = "el_input_images.zip"
DATA_URL = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{ASSET_NAME}"

# Integrity check for the current asset. If you rebuild the zip, update this
# (shasum -a 256 el_input_images.zip) or call download_data(verify=False).
EXPECTED_SHA256 = "b929cef3f813f3f4b7070bb5f6d274777ab39f5ce1cf45ebed4b3662b60b57cf"

# The zip unpacks to this path (relative to the repo root) — used to detect an
# existing download and skip re-fetching.
INPUT_SUBDIR = Path("data/workshop/images/01_input")
EXPECTED_PNG_COUNT = 102


def _find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (or cwd) to the dir containing pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here  # fall back to cwd if no marker found


def _png_count(root: Path) -> int:
    return len(list((root / INPUT_SUBDIR).rglob("*.png")))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a simple progress readout."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (trusted, user-supplied URL)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  downloading… {done/1e6:6.1f} / {total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  downloading… {done/1e6:6.1f} MB", end="", flush=True)
        print()


def download_data(
    url: str = DATA_URL,
    dest_root: str | Path | None = None,
    expected_sha256: str | None = EXPECTED_SHA256,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Download and unpack the workshop EL image set.

    Parameters
    ----------
    url : str
        Release-asset URL for the images zip. Any host works (GitHub, OSF, …).
    dest_root : path, optional
        Repo root to unpack into. Defaults to the detected repo root.
    expected_sha256 : str, optional
        Expected checksum; compared when ``verify`` is True.
    force : bool
        Re-download even if the images already appear present.
    verify : bool
        Verify the zip checksum before extracting.

    Returns
    -------
    Path to the populated ``…/01_input`` directory.
    """
    root = Path(dest_root).resolve() if dest_root else _find_repo_root()
    target = root / INPUT_SUBDIR

    existing = _png_count(root)
    if existing >= EXPECTED_PNG_COUNT and not force:
        print(f"✓ EL images already present ({existing} PNGs at {target}). "
              f"Use force=True to re-download.")
        return target

    print(f"Fetching workshop EL images from:\n  {url}")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / ASSET_NAME
        _download(url, zip_path)

        if verify and expected_sha256:
            actual = _sha256(zip_path)
            if actual != expected_sha256:
                raise ValueError(
                    "Checksum mismatch — download may be corrupt or the asset "
                    f"changed.\n  expected {expected_sha256}\n  actual   {actual}\n"
                    "If you intentionally rebuilt the zip, update EXPECTED_SHA256 "
                    "or pass verify=False."
                )
            print("✓ checksum verified")

        print(f"Extracting into {root} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)

    final = _png_count(root)
    if final != EXPECTED_PNG_COUNT:
        print(f"⚠ expected {EXPECTED_PNG_COUNT} PNGs but found {final} — "
              "check the asset contents.")
    else:
        print(f"✓ done: {final} EL images at {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the workshop EL image set.")
    p.add_argument("--url", default=DATA_URL, help="release-asset URL (default: %(default)s)")
    p.add_argument("--dest", default=None, help="repo root to unpack into (default: auto-detect)")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.add_argument("--no-verify", action="store_true", help="skip checksum verification")
    args = p.parse_args(argv)
    try:
        download_data(url=args.url, dest_root=args.dest,
                      force=args.force, verify=not args.no_verify)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the CLI
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
