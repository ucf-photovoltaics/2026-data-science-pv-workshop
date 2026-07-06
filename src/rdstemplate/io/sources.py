"""Data source abstractions: LocalSource, DriveSource, S3Source."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class DataSource(ABC):
    """Uniform interface for retrieving raw data files."""

    @abstractmethod
    def list_samples(self) -> list[str]:
        """Return all sample IDs available in this source."""

    @abstractmethod
    def get(self, sample_id: str, modality: str) -> Path | str:
        """Return a local path (or file-like) for the given sample + modality."""

    def sync(self, local_dir: str) -> None:
        """Bulk-fetch all data to *local_dir*. Optional for sources that are already local."""


class LocalSource(DataSource):
    """Data stored in a local directory tree.

    Expected layout::

        root/
          <sample_id>/
            curves/   spectra/   images/   timeseries/
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def list_samples(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def get(self, sample_id: str, modality: str) -> Path:
        return self.root / sample_id / modality

    def __repr__(self) -> str:
        return f"LocalSource(root={self.root!r})"


class DriveSource(LocalSource):
    """Google Drive mounted at a local path (identical to LocalSource once mounted).

    In Colab, mount first::

        from google.colab import drive
        drive.mount('/content/drive')
        source = DriveSource('/content/drive/MyDrive/myproject/data')
    """

    def __repr__(self) -> str:
        return f"DriveSource(root={self.root!r})"


class S3Source(DataSource):
    """Data stored in an S3 bucket.

    Credentials are read from environment variables — never hardcoded::

        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (optional) AWS_SESSION_TOKEN

    Uses *s3fs* for directory listing and streaming, or *boto3* for bulk sync.
    """

    def __init__(self, bucket: str, prefix: str = "", region: str | None = None) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.region = region
        self._fs = None  # lazy-init so import is deferred

    def _get_fs(self):
        if self._fs is None:
            import s3fs  # noqa: PLC0415

            kw: dict = {}
            if self.region:
                kw["client_kwargs"] = {"region_name": self.region}
            self._fs = s3fs.S3FileSystem(anon=False, **kw)
        return self._fs

    def list_samples(self) -> list[str]:
        fs = self._get_fs()
        prefix = f"{self.bucket}/{self.prefix}" if self.prefix else self.bucket
        paths = fs.ls(prefix, detail=False)
        return sorted(p.split("/")[-1] for p in paths if fs.isdir(p))

    def get(self, sample_id: str, modality: str) -> str:
        """Return an s3:// URI; callers that need a local path should call sync() first."""
        parts = [self.bucket]
        if self.prefix:
            parts.append(self.prefix)
        parts += [sample_id, modality]
        return "s3://" + "/".join(parts)

    def sync(self, local_dir: str) -> None:
        """Download the entire prefix to *local_dir* using boto3."""
        import boto3  # noqa: PLC0415

        s3 = boto3.client("s3", region_name=self.region)
        paginator = s3.get_paginator("list_objects_v2")
        prefix = self.prefix + "/" if self.prefix else ""
        local = Path(local_dir)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                dest = local / key[len(prefix):]
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(self.bucket, key, str(dest))

    def __repr__(self) -> str:
        return f"S3Source(bucket={self.bucket!r}, prefix={self.prefix!r})"


def source_from_config(cfg) -> DataSource:
    """Instantiate the correct DataSource from a DataSourceConfig."""
    t = cfg.type
    if t in ("local", "drive"):
        cls = LocalSource if t == "local" else DriveSource
        return cls(cfg.path)
    if t == "s3":
        return S3Source(bucket=cfg.bucket, prefix=cfg.prefix, region=cfg.region)
    raise ValueError(f"Unknown data_source.type: {t!r}")
