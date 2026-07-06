"""Tests: merge() grain, outer-join, gap-fill, per-sample-constant broadcast, missing modality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdstemplate.merge import build_feature_table, merge_feature_tables


def _make_metadata(samples=("S1", "S2"), steps=(1, 2, 3, 4)):
    rows = [
        {"sample_id": s, "exposure_step": st, "outcome": float(st)}
        for s in samples
        for st in steps
    ]
    return pd.DataFrame(rows)


def _feature_table(records):
    return build_feature_table(records, "test")


class TestMergeGrain:
    def test_one_row_per_sample_exposure_step(self):
        meta = _make_metadata()
        feat = _feature_table([
            {"sample_id": "S1", "exposure_step": 1, "mod__val": 1.0},
            {"sample_id": "S1", "exposure_step": 2, "mod__val": 2.0},
            {"sample_id": "S2", "exposure_step": 1, "mod__val": 3.0},
            {"sample_id": "S2", "exposure_step": 2, "mod__val": 4.0},
        ])
        tidy = merge_feature_tables({"mod": feat}, meta)
        assert not tidy.duplicated(["sample_id", "exposure_step"]).any()
        assert len(tidy) == len(meta)

    def test_namespaced_columns_present(self):
        meta = _make_metadata(steps=(1, 2))
        feat = _feature_table([
            {"sample_id": "S1", "exposure_step": 1, "mod__feature_a": 0.5},
            {"sample_id": "S2", "exposure_step": 1, "mod__feature_a": 1.5},
        ])
        tidy = merge_feature_tables({"mod": feat}, meta)
        assert "mod__feature_a" in tidy.columns


class TestMisalignedStepOuterJoin:
    """Modality A measured at all steps, modality B only at a subset."""

    def _build_tables(self):
        meta = _make_metadata(samples=("S1",), steps=(1, 2, 3, 4))

        all_steps_feat = _feature_table([
            {"sample_id": "S1", "exposure_step": s, "modA__v": float(s)}
            for s in (1, 2, 3, 4)
        ])
        subset_feat = _feature_table([
            {"sample_id": "S1", "exposure_step": s, "modB__v": float(s * 10)}
            for s in (1, 3)   # only steps 1 and 3
        ])
        return meta, all_steps_feat, subset_feat

    def test_no_step_is_dropped(self):
        meta, modA, modB = self._build_tables()
        tidy = merge_feature_tables({"modA": modA, "modB": modB}, meta)
        assert set(tidy["exposure_step"].unique()) == {1, 2, 3, 4}

    def test_missing_step_has_nan(self):
        meta, modA, modB = self._build_tables()
        tidy = merge_feature_tables({"modA": modA, "modB": modB}, meta)
        missing = tidy[tidy["exposure_step"].isin([2, 4])]["modB__v"]
        assert missing.isna().all(), "Steps 2 and 4 should be NaN for modB"

    def test_present_step_has_value(self):
        meta, modA, modB = self._build_tables()
        tidy = merge_feature_tables({"modA": modA, "modB": modB}, meta)
        present = tidy[tidy["exposure_step"].isin([1, 3])]["modB__v"]
        assert present.notna().all(), "Steps 1 and 3 should have modB values"


class TestGapFillPolicy:
    def _tidy_with_gap(self, policy):
        meta = _make_metadata(samples=("S1",), steps=(1, 2, 3, 4))
        feat = _feature_table([
            {"sample_id": "S1", "exposure_step": 1, "mod__v": 10.0},
            {"sample_id": "S1", "exposure_step": 3, "mod__v": 30.0},
        ])
        return merge_feature_tables({"mod": feat}, meta, gap_fill_policy=policy)

    def test_none_leaves_nans(self):
        tidy = self._tidy_with_gap("none")
        assert tidy.loc[tidy["exposure_step"] == 2, "mod__v"].isna().all()
        assert tidy.loc[tidy["exposure_step"] == 4, "mod__v"].isna().all()

    def test_ffill_carries_value_forward(self):
        tidy = self._tidy_with_gap("ffill")
        # Step 2 should carry forward the value from step 1 (10.0).
        val_at_2 = tidy.loc[tidy["exposure_step"] == 2, "mod__v"].iloc[0]
        assert val_at_2 == pytest.approx(10.0)

    def test_ffill_does_not_leak_across_samples(self):
        meta = _make_metadata(samples=("S1", "S2"), steps=(1, 2))
        feat = _feature_table([
            {"sample_id": "S1", "exposure_step": 1, "mod__v": 99.0},
            # S2 has no data at any step — should remain NaN after ffill.
        ])
        tidy = merge_feature_tables({"mod": feat}, meta, gap_fill_policy="ffill")
        s2_vals = tidy.loc[tidy["sample_id"] == "S2", "mod__v"]
        assert s2_vals.isna().all(), "ffill must not propagate S1 values into S2"

    def test_interpolate_fills_between_known_values(self):
        tidy = self._tidy_with_gap("interpolate")
        val_at_2 = tidy.loc[tidy["exposure_step"] == 2, "mod__v"].iloc[0]
        # Linear interpolation between 10 (step 1) and 30 (step 3) → 20 at step 2.
        assert val_at_2 == pytest.approx(20.0, abs=1.0)


class TestPerSampleConstantBroadcast:
    def test_constant_feature_broadcast_across_steps(self):
        meta = _make_metadata(samples=("S1",), steps=(1, 2, 3))
        # Constant value returned at every step.
        feat = _feature_table([
            {"sample_id": "S1", "exposure_step": s, "mod__const": 42.0}
            for s in (1, 2, 3)
        ])
        tidy = merge_feature_tables({"mod": feat}, meta)
        assert (tidy["mod__const"] == 42.0).all()


class TestMissingModality:
    def test_missing_entire_modality_no_crash(self):
        meta = _make_metadata(samples=("S1",), steps=(1, 2))
        # modA has data, modB is completely absent for S1.
        modA = _feature_table([
            {"sample_id": "S1", "exposure_step": 1, "modA__v": 1.0},
            {"sample_id": "S1", "exposure_step": 2, "modA__v": 2.0},
        ])
        modB = _feature_table([])  # empty — no data for this modality
        tidy = merge_feature_tables({"modA": modA, "modB": modB}, meta)
        assert len(tidy) == 2
        assert "modA__v" in tidy.columns

    def test_empty_feature_tables_returns_metadata(self):
        meta = _make_metadata()
        tidy = merge_feature_tables({}, meta)
        assert list(tidy.columns) == list(meta.columns)
