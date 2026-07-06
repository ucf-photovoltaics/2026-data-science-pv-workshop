"""Configuration: Pydantic model loaded from a YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class DataSourceConfig(BaseModel):
    type: Literal["local", "drive", "s3"] = "local"
    path: Optional[str] = None          # local / drive
    bucket: Optional[str] = None        # s3
    prefix: str = ""                    # s3 key prefix
    region: Optional[str] = None        # s3 region (optional)

    @model_validator(mode="after")
    def _check_required_fields(self) -> "DataSourceConfig":
        if self.type in ("local", "drive") and not self.path:
            raise ValueError(f"data_source.path is required for type '{self.type}'")
        if self.type == "s3" and not self.bucket:
            raise ValueError("data_source.bucket is required for type 's3'")
        return self


class MetadataConfig(BaseModel):
    file: str
    sample_id_col: str = "sample_id"
    exposure_step_col: str = "exposure_step"


class ExtractorEntry(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ExtractorsConfig(BaseModel):
    curves: List[ExtractorEntry] = Field(default_factory=list)
    spectra: List[ExtractorEntry] = Field(default_factory=list)
    images: List[ExtractorEntry] = Field(default_factory=list)
    timeseries: List[ExtractorEntry] = Field(default_factory=list)


class MergeConfig(BaseModel):
    gap_fill_policy: Literal["none", "ffill", "interpolate"] = "none"


class ModelConfig(BaseModel):
    name: str = "random_forest_regressor"
    target_col: str = "outcome"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class Config(BaseModel):
    data_source: DataSourceConfig
    metadata: MetadataConfig
    extractors: ExtractorsConfig = Field(default_factory=ExtractorsConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    random_seed: int = 42


def load_config(path: "str | Path") -> Config:
    """Load and validate a YAML config file, returning a Config object."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
