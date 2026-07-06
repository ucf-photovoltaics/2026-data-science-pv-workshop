"""Tests: end-to-end Pipeline on data/sample/; CLI subcommands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rdstemplate.config import load_config
from rdstemplate.pipeline import Pipeline


class TestPipelineE2E:
    def test_tidy_nonempty_and_correct_grain(self, example_config_path):
        cfg = load_config(example_config_path)
        pipe = Pipeline(cfg)
        pipe.load_metadata()
        pipe.extract_features()
        tidy = pipe.merge()

        assert len(tidy) > 0, "Tidy dataframe must not be empty"
        assert not tidy.duplicated(["sample_id", "exposure_step"]).any(), (
            "Tidy dataframe has duplicate (sample_id, exposure_step) rows"
        )

    def test_all_metadata_steps_present(self, example_config_path):
        cfg = load_config(example_config_path)
        pipe = Pipeline(cfg)
        meta = pipe.load_metadata()
        pipe.extract_features()
        tidy = pipe.merge()

        expected = set(zip(meta["sample_id"], meta["exposure_step"]))
        actual = set(zip(tidy["sample_id"], tidy["exposure_step"]))
        assert expected == actual, "Some metadata (sample, step) pairs are missing from tidy df"

    def test_misaligned_steps_produce_nans(self, example_config_path):
        """Spectra only at steps 1,3; images only at steps 2,4 in sample data."""
        cfg = load_config(example_config_path)
        pipe = Pipeline(cfg)
        pipe.load_metadata()
        pipe.extract_features()
        tidy = pipe.merge()

        spectra_cols = [c for c in tidy.columns if c.startswith("spectra_")]
        image_cols = [c for c in tidy.columns if c.startswith("image_")]

        if spectra_cols:
            missing_spectra = tidy[tidy["exposure_step"].isin([2, 4])][spectra_cols[0]]
            assert missing_spectra.isna().all(), "Spectra columns should be NaN at steps 2 and 4"

        if image_cols:
            missing_images = tidy[tidy["exposure_step"].isin([1, 3])][image_cols[0]]
            assert missing_images.isna().all(), "Image columns should be NaN at steps 1 and 3"

    def test_model_result_has_metrics(self, example_config_path):
        cfg = load_config(example_config_path)
        pipe = Pipeline(cfg)
        pipe.load_metadata()
        pipe.extract_features()
        pipe.merge()
        results = pipe.run_model()

        assert "metrics" in results
        assert "r2" in results["metrics"]
        assert "mae" in results["metrics"]

    def test_run_convenience_method(self, example_config_path):
        cfg = load_config(example_config_path)
        results = Pipeline(cfg).run()
        assert results["n_samples"] > 0


class TestCLI:
    """Smoke tests for CLI subcommands via subprocess."""

    def _run(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-m", "rdstemplate"] + list(args),
            capture_output=True,
            text=True,
            cwd=cwd,
        )

    def test_version(self, repo_root):
        r = self._run("version", cwd=repo_root)
        assert r.returncode == 0
        assert "0.1.0" in r.stdout

    def test_list_extractors(self, repo_root):
        r = self._run("list-extractors", cwd=repo_root)
        assert r.returncode == 0
        assert "curve_auc" in r.stdout
        assert "spectra_peaks" in r.stdout

    def test_list_models(self, repo_root):
        r = self._run("list-models", cwd=repo_root)
        assert r.returncode == 0
        assert "random_forest_regressor" in r.stdout

    def test_validate_good_config(self, repo_root):
        r = self._run("validate", "--config", "configs/example.yaml", cwd=repo_root)
        assert r.returncode == 0
        assert "Validation passed" in r.stdout

    def test_validate_bad_config(self, repo_root, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("data_source:\n  type: local\n")
        r = self._run("validate", "--config", str(bad), cwd=repo_root)
        assert r.returncode != 0

    def test_run_writes_outputs(self, repo_root, tmp_path):
        r = self._run(
            "run",
            "--config", "configs/example.yaml",
            "--out", str(tmp_path),
            cwd=repo_root,
        )
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "tidy.parquet").exists()
        assert (tmp_path / "metrics.json").exists()

    def test_extract_writes_tidy(self, repo_root, tmp_path):
        r = self._run(
            "extract",
            "--config", "configs/example.yaml",
            "--out", str(tmp_path),
            cwd=repo_root,
        )
        assert r.returncode == 0, r.stderr
        assert (tmp_path / "tidy.parquet").exists()
