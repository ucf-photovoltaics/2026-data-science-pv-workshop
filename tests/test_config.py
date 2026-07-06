"""Tests: config loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from rdstemplate.config import load_config, Config


class TestLoadConfig:
    def test_loads_example_yaml(self, example_config_path):
        cfg = load_config(example_config_path)
        assert cfg.data_source.type == "local"
        assert cfg.merge.gap_fill_policy == "none"
        assert cfg.random_seed == 42

    def test_rejects_missing_path_for_local_source(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(textwrap.dedent("""\
            data_source:
              type: local
            metadata:
              file: data/sample/metadata.csv
        """))
        with pytest.raises(Exception):
            load_config(bad)

    def test_rejects_missing_bucket_for_s3(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(textwrap.dedent("""\
            data_source:
              type: s3
            metadata:
              file: data/sample/metadata.csv
        """))
        with pytest.raises(Exception):
            load_config(bad)

    def test_rejects_invalid_gap_fill_policy(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(textwrap.dedent("""\
            data_source:
              type: local
              path: data/sample
            metadata:
              file: data/sample/metadata.csv
            merge:
              gap_fill_policy: bogus_value
        """))
        with pytest.raises(Exception):
            load_config(bad)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_source_override_after_load(self, example_config_path):
        cfg = load_config(example_config_path)
        cfg.data_source.type = "drive"
        assert cfg.data_source.type == "drive"
