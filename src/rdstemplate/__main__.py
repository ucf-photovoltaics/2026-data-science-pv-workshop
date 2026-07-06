"""CLI entry point for rdstemplate.

Runnable as:
    python -m rdstemplate <subcommand> [options]
    rdstemplate <subcommand> [options]   (after pip install)

Subcommands
-----------
run          Full pipeline: metadata → extract → merge → model; writes outputs to --out.
extract      Feature extraction + merge only; stops before modeling.
validate     Validate config and check data source is reachable; exits non-zero on failure.
list-extractors  Print registered extractor names (no config required).
list-models      Print registered model names (no config required).
version      Print the package version.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _load(config_path: str):
    from rdstemplate.config import load_config  # noqa: PLC0415

    return load_config(config_path)


def _apply_source_override(cfg, source_type: str | None):
    if source_type is not None:
        cfg.data_source.type = source_type
    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    from rdstemplate.pipeline import Pipeline  # noqa: PLC0415

    cfg = _load(args.config)
    _apply_source_override(cfg, args.source)

    pipe = Pipeline(cfg)
    pipe.load_metadata()
    pipe.extract_features()
    tidy = pipe.merge()
    results = pipe.run_model()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy_path = out_dir / "tidy.parquet"
    tidy.to_parquet(tidy_path, index=False)
    print(f"Tidy dataframe written to {tidy_path} ({len(tidy)} rows × {tidy.shape[1]} cols)")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": results["model_name"],
                "n_samples": results["n_samples"],
                "metrics": results["metrics"],
            },
            indent=2,
        )
    )
    print(f"Model metrics written to {metrics_path}")
    print(f"Metrics: {results['metrics']}")
    return 0


def cmd_extract(args) -> int:
    from rdstemplate.pipeline import Pipeline  # noqa: PLC0415

    cfg = _load(args.config)
    _apply_source_override(cfg, getattr(args, "source", None))

    pipe = Pipeline(cfg)
    pipe.load_metadata()
    pipe.extract_features()
    tidy = pipe.merge()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy_path = out_dir / "tidy.parquet"
    tidy.to_parquet(tidy_path, index=False)
    print(f"Tidy dataframe written to {tidy_path} ({len(tidy)} rows × {tidy.shape[1]} cols)")
    return 0


def cmd_validate(args) -> int:
    try:
        cfg = _load(args.config)
        print(f"Config OK: {args.config}")
    except Exception as exc:
        print(f"Config INVALID: {exc}", file=sys.stderr)
        return 1

    try:
        from rdstemplate.io.sources import source_from_config  # noqa: PLC0415

        source = source_from_config(cfg.data_source)
        samples = source.list_samples()
        print(f"Data source reachable: {source!r}")
        print(f"Samples found ({len(samples)}): {samples[:5]}{'…' if len(samples) > 5 else ''}")
    except Exception as exc:
        print(f"Data source UNREACHABLE: {exc}", file=sys.stderr)
        return 1

    try:
        from rdstemplate.metadata import load_metadata  # noqa: PLC0415

        meta = load_metadata(cfg.metadata)
        print(
            f"Metadata OK: {len(meta)} rows, "
            f"{meta['sample_id'].nunique()} samples, "
            f"{meta['exposure_step'].nunique()} unique steps"
        )
    except Exception as exc:
        print(f"Metadata INVALID: {exc}", file=sys.stderr)
        return 1

    print("Validation passed.")
    return 0


def cmd_list_extractors(_args) -> int:
    # Import to trigger registration.
    import rdstemplate.features.curves  # noqa: F401
    import rdstemplate.features.images  # noqa: F401
    import rdstemplate.features.spectra  # noqa: F401
    import rdstemplate.features.timeseries  # noqa: F401
    from rdstemplate.features.base import EXTRACTOR_REGISTRY  # noqa: PLC0415

    print("Registered extractors:")
    for name in sorted(EXTRACTOR_REGISTRY):
        print(f"  {name}")
    return 0


def cmd_list_models(_args) -> int:
    import rdstemplate.models.registry  # noqa: F401
    from rdstemplate.models.base import MODEL_REGISTRY  # noqa: PLC0415

    print("Registered models:")
    for name in sorted(MODEL_REGISTRY):
        print(f"  {name}")
    return 0


def cmd_version(_args) -> int:
    from rdstemplate import __version__  # noqa: PLC0415

    print(f"rdstemplate {__version__}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rdstemplate",
        description="Research data-science template pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Full pipeline: metadata → extract → merge → model.")
    p_run.add_argument("--config", required=True, metavar="PATH", help="Path to YAML config.")
    p_run.add_argument(
        "--source",
        choices=["local", "drive", "s3"],
        default=None,
        help="Override data_source.type from config.",
    )
    p_run.add_argument("--out", default="outputs", metavar="DIR", help="Output directory.")

    # extract
    p_ext = sub.add_parser("extract", help="Feature extraction + merge only.")
    p_ext.add_argument("--config", required=True, metavar="PATH")
    p_ext.add_argument(
        "--source", choices=["local", "drive", "s3"], default=None
    )
    p_ext.add_argument("--out", default="outputs", metavar="DIR")

    # validate
    p_val = sub.add_parser("validate", help="Validate config and data source.")
    p_val.add_argument("--config", required=True, metavar="PATH")

    # list-extractors
    sub.add_parser("list-extractors", help="Print registered extractor names.")

    # list-models
    sub.add_parser("list-models", help="Print registered model names.")

    # version
    sub.add_parser("version", help="Print the package version.")

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "run": cmd_run,
        "extract": cmd_extract,
        "validate": cmd_validate,
        "list-extractors": cmd_list_extractors,
        "list-models": cmd_list_models,
        "version": cmd_version,
    }

    handler = dispatch[args.command]
    sys.exit(handler(args))


if __name__ == "__main__":
    main()
