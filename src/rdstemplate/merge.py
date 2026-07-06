"""Merge per-modality feature tables into a single tidy dataframe.

Row grain: one row per (sample_id, exposure_step).
Join strategy: outer join on (sample_id, exposure_step) so misaligned steps
across modalities are preserved, with NaNs where a modality was not measured
at a given step.

Gap-fill policy (configured via merge.gap_fill_policy):
  none        — leave NaNs as-is (default; safest for analysis)
  ffill       — carry the last observed value forward along each sample's
                exposure axis; does NOT leak across samples
  interpolate — linear interpolation along each sample's exposure axis
"""

from __future__ import annotations

import pandas as pd

GapFillPolicy = str  # "none" | "ffill" | "interpolate"


def merge_feature_tables(
    feature_tables: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    gap_fill_policy: GapFillPolicy = "none",
) -> pd.DataFrame:
    """Merge modality feature tables and metadata into one tidy dataframe.

    Parameters
    ----------
    feature_tables:
        Mapping of modality name → DataFrame with columns
        [sample_id, exposure_step, <feature cols…>].
    metadata:
        DataFrame with at minimum [sample_id, exposure_step] plus any
        sample-level columns to broadcast.
    gap_fill_policy:
        How to handle NaNs introduced by the outer join.

    Returns
    -------
    DataFrame with one row per (sample_id, exposure_step).
    """
    if not feature_tables:
        # Nothing extracted — return metadata grain only.
        return metadata.copy()

    combined: pd.DataFrame | None = None
    key = ["sample_id", "exposure_step"]

    for modality, df in feature_tables.items():
        if df is None or df.empty:
            continue
        # Ensure the join key is present.
        for col in key:
            if col not in df.columns:
                raise ValueError(
                    f"Feature table for modality '{modality}' is missing column '{col}'"
                )
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on=key, how="outer")

    if combined is None:
        return metadata.copy()

    # Outer-join against metadata so all metadata columns are broadcast.
    # Metadata already carries one row per (sample_id, exposure_step).
    tidy = metadata.merge(combined, on=key, how="outer")

    # Apply gap-fill policy per sample along the exposure axis.
    if gap_fill_policy != "none":
        tidy = tidy.sort_values(key)
        feature_cols = [c for c in tidy.columns if c not in key]
        if gap_fill_policy == "ffill":
            tidy[feature_cols] = (
                tidy.groupby("sample_id")[feature_cols]
                .transform(lambda g: g.ffill())
            )
        elif gap_fill_policy == "interpolate":
            tidy[feature_cols] = (
                tidy.groupby("sample_id")[feature_cols]
                .transform(lambda g: g.interpolate(method="linear", limit_direction="forward"))
            )

    tidy = tidy.reset_index(drop=True)
    return tidy


def build_feature_table(
    records: list[dict],
    modality_prefix: str,
) -> pd.DataFrame:
    """Convert a list of per-(sample, step) dicts into a namespaced DataFrame.

    Parameters
    ----------
    records:
        Each dict must contain 'sample_id' and 'exposure_step' plus feature keys.
        Keys are already namespaced by the extractor (e.g. ``curve_auc__auc``).
    modality_prefix:
        Used only if records contain feature keys that need prefixing; currently
        extractors self-namespace, so this is used for documentation/debugging.
    """
    if not records:
        return pd.DataFrame(columns=["sample_id", "exposure_step"])
    return pd.DataFrame(records)
