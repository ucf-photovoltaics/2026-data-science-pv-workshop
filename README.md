# Research Data-Science Template

A reusable, pip-installable template for research data-science workflows that follow
the pattern: **sample metadata → multi-modality feature extraction → tidy merge → model**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ucf-photovoltaics/research-ds-template/blob/main/environments/colab/analysis_template.ipynb)
[![CI](https://github.com/ucf-photovoltaics/research-ds-template/actions/workflows/ci.yml/badge.svg)](https://github.com/ucf-photovoltaics/research-ds-template/actions/workflows/ci.yml)

---

## What it does

1. Reads **sample metadata** — a CSV that defines samples and their exposure histories
   (one row per `(sample_id, exposure_step)`).
2. Reads **multiple data modalities** per sample (curves, images, spectra, timeseries),
   each measured at one or more exposure steps.
3. Runs **feature extraction** on each modality independently, per exposure step.
4. **Merges** all extracted features into a single tidy dataframe.
   Row grain: **one row per `(sample_id, exposure_step)`**.
5. **Fits and evaluates** a model on that dataframe.

> **Grain note:** The unit of analysis is the *sample × exposure step*, not the sample.
> One physical sample is measured at incremental exposures (dose, time, cumulative stress, …),
> and each measurement becomes its own row. Exposure steps **can differ across modalities**;
> the merge uses an outer join so no step is ever silently dropped.

---

## Quickstart

### Option A — Google Colab (recommended for new users)

Click the badge above. The notebook will:
1. Clone this repo at a pinned release tag.
2. Install the package and dependencies.
3. Run the full pipeline on the bundled synthetic sample data.

> **Save a copy first:** `File → Save a copy in Drive` before editing,
> so your work is saved to your own Google account.

### Option B — Local / HPC

```bash
git clone --branch v0.1.0 https://github.com/ucf-photovoltaics/research-ds-template.git
cd research-ds-template
pip install -r requirements.txt
pip install -e .

# Validate config + data source
python -m rdstemplate validate --config configs/example.yaml

# Full pipeline (writes outputs/ by default)
python -m rdstemplate run --config configs/example.yaml --out outputs/
```

See `environments/hpc/` for SLURM batch script stubs.

---

## Notebook vs package update model

| Where your copy lives | What happens on re-open |
|-----------------------|------------------------|
| Saved Drive copy of the notebook | Frozen at your saved state — your edits are preserved. |
| Package code (`rdstemplate`) | Re-cloned fresh from the pinned `RELEASE` tag each Colab session. |

**Pin a release tag** (`RELEASE = "v0.1.0"` in Cell 1) so your analysis is reproducible.
When you want to pick up a new package version, update that one variable and re-run.

---

## Three tiers of customisation

| Tier | What you change | When to use |
|------|-----------------|-------------|
| **a — Config only** | Edit `configs/example.yaml` | Different extractors, model, gap-fill policy, target column |
| **b — Subclass** | Subclass `FeatureExtractor` or `ModelWrapper`, register with `@register_extractor` / `@register_model` | Add a new modality or model without touching core code |
| **c — Fork** | Fork the repo, modify `src/rdstemplate/` | Fundamentally different pipeline logic |

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add an extractor or model via tier b.

---

## Repository layout

```
src/rdstemplate/          # installable package — all real logic lives here
  config.py               # Pydantic settings, loads YAML
  metadata.py             # load + validate (sample_id, exposure_step) metadata
  io/sources.py           # DataSource ABC + Local/Drive/S3 implementations
  io/loaders.py           # per-modality file readers
  features/               # FeatureExtractor ABC, registry, built-in extractors
  merge.py                # outer-join on (sample_id, exposure_step), gap-fill
  models/                 # ModelWrapper ABC, registry, built-in wrappers
  pipeline.py             # Pipeline: load → extract → merge → model
  __main__.py             # CLI: run / extract / validate / list-* / version
environments/
  colab/                  # Colab notebook (thin caller of the package)
  hpc/                    # SLURM / HPC stubs (v0.1.0: documented stubs)
configs/example.yaml      # starter config — copy and customise
data/sample/              # tiny synthetic dataset for tests + demo
tests/                    # pytest suite (43 tests)
```

---

## CLI reference

```
python -m rdstemplate <subcommand> [options]

  run        --config PATH [--source local|drive|s3] [--out DIR]
             Full pipeline. Writes tidy.parquet + metrics.json to --out.

  extract    --config PATH [--out DIR]
             Feature extraction + merge only (no model). Useful as an HPC pre-step.

  validate   --config PATH
             Validate config and check the data source is reachable.
             Exit non-zero on failure — use as a pre-flight before submitting a job.

  list-extractors          Print registered extractor names (no config needed).
  list-models              Print registered model names (no config needed).
  version                  Print the package version.
```

---

## Configuration reference (`configs/example.yaml`)

```yaml
data_source:
  type: local          # local | drive | s3
  path: data/sample

metadata:
  file: data/sample/metadata.csv
  sample_id_col: sample_id
  exposure_step_col: exposure_step

extractors:
  curves:
    - name: curve_auc
  spectra:
    - name: spectra_peaks
  images:
    - name: image_basic_stats
  timeseries:
    - name: timeseries_summary

merge:
  gap_fill_policy: none    # none (default) | ffill | interpolate

model:
  name: random_forest_regressor
  target_col: outcome
  hyperparameters:
    n_estimators: 100
    random_state: 42

random_seed: 42
```

**Gap-fill policy** controls how NaNs introduced by misaligned exposure steps are handled:
- `none` — leave NaN (safest default; makes missingness visible)
- `ffill` — carry the last observed value forward along each sample's exposure axis; does not leak across samples
- `interpolate` — linear interpolation along each sample's exposure axis

---

## Security

Security controls implemented in this repo:

| Control | Implementation |
|---------|---------------|
| Secret scanning | `gitleaks` pre-commit hook |
| Notebook output stripping | `nbstripout` pre-commit hook |
| Python security linting | `bandit -r src/ -ll` (pre-commit + CI) |
| Dependency vulnerability scanning | `pip-audit` in CI |
| Hash-locked dependencies | `pip-compile --generate-hashes` |
| Pinned GitHub Actions | Full commit SHAs, not tags |
| No dangerous runtime patterns | No `eval`/`exec`, no `pickle.load`, no `curl\|bash` |
| Credentials | Never hardcoded; read from environment variables or Colab Secrets |

**GitHub repository settings checklist** (configure in repo Settings — these are UI actions, not code):

- [ ] **Branch protection on `main`**: require PR + ≥1 review, no direct pushes
- [ ] **Dependabot alerts**: enabled for dependencies and GitHub Actions
- [ ] **Secret scanning + push protection**: enabled
- [ ] **Signed-commit verification**: enabled for maintainers

> These controls reduce but do not eliminate risk.
> Human PR review is the highest-value control.

---

## Generating / regenerating the synthetic sample data

```bash
python tests/generate_sample_data.py
```

The script is deterministic (fixed seed) and produces a tiny dataset in `data/sample/`
(3 samples × 4 exposure steps; spectra at steps 1 & 3 only; images at steps 2 & 4 only)
that deliberately exercises the misaligned-step merge path.

---

## HPC note

The `environments/hpc/` directory contains documented stubs for running the pipeline
as a SLURM batch job. These are intentionally left as stubs for v0.1.0 because SLURM
partition names, account strings, and module names are site-specific. See
[`environments/hpc/README.md`](environments/hpc/README.md) for the intended approach.
The package itself is environment-agnostic — HPC just calls the same CLI as Colab.

---

## License

MIT — see [LICENSE](LICENSE).
