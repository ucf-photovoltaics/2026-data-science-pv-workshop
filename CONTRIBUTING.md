# Contributing

Thank you for contributing to the research data-science template.

---

## Development setup

```bash
git clone https://github.com/ucf-photovoltaics/research-ds-template.git
cd research-ds-template
pip install -e ".[dev]"
pip install pre-commit
pre-commit install
python tests/generate_sample_data.py
pytest tests/
```

---

## Adding a feature extractor (customisation tier b)

1. **Create or edit a module** under `src/rdstemplate/features/`.
   You can add to an existing modality file (e.g. `curves.py`) or create a new one.

2. **Subclass `FeatureExtractor`** and decorate with `@register_extractor`:

   ```python
   from rdstemplate.features.base import FeatureExtractor, register_extractor

   @register_extractor("my_curve_feature")
   class MyCurveFeature(FeatureExtractor):
       def extract(self, sample_id: str, exposure_step, data) -> dict:
           """Return {feature_name: value} for ONE (sample_id, exposure_step) pair."""
           if data is None or data.empty:
               return {}
           # ... compute features ...
           return {"my_curve_feature__result": 42.0}
   ```

   Key contract:
   - `extract` is called **once per `(sample_id, exposure_step)`** — one observation at a time.
   - Return an empty dict `{}` if data is None or the step can't be processed.
   - **Prefix every key** with your extractor name + `__` to avoid column collisions.
   - Do not return multiple rows — the pipeline handles the per-step loop.

3. **If you created a new file**, import it in `pipeline.py`'s `_import_all_extractors()`:

   ```python
   import rdstemplate.features.my_new_module  # noqa: F401
   ```

4. **Reference the name in your YAML config**:

   ```yaml
   extractors:
     curves:
       - name: my_curve_feature
   ```

5. **Write a test** in `tests/test_extractors.py` that runs your extractor on synthetic
   data for one `(sample, step)` and asserts the expected keys are returned.

---

## Adding a model (customisation tier b)

1. **Edit `src/rdstemplate/models/registry.py`** (or create a new module and import it
   in `pipeline.py`'s `_import_all_models()`).

2. **Subclass `ModelWrapper`** and decorate with `@register_model`:

   ```python
   from rdstemplate.models.base import ModelWrapper, register_model

   @register_model("my_model")
   class MyModel(ModelWrapper):
       def fit(self, X, y):
           # train self._model
           return self

       def predict(self, X):
           return self._model.predict(X)
   ```

3. **Reference the name in your YAML config**:

   ```yaml
   model:
     name: my_model
     hyperparameters:
       some_param: 10
   ```

---

## Pull request requirements

- **All PRs must be reviewed** by at least one other team member before merging.
  No direct pushes to `main`.
- The CI suite must pass: `pytest`, `bandit`, and `pip-audit`.
- Pre-commit hooks must pass locally before pushing:
  `pre-commit run --all-files`
- Keep PRs focused — one logical change per PR.
- Update or add tests for any new extractor or model.

---

## Safe coding rules — no exceptions

These apply to all code in `src/` and `tests/`:

| Rule | Reason |
|------|--------|
| No `eval()` or `exec()` on fetched or user-supplied data | Remote code execution risk |
| No `pickle.load()` on untrusted files | Arbitrary code execution on deserialise |
| No `curl \| bash` patterns in docs or scripts | Supply-chain attack vector |
| No hardcoded credentials, tokens, or API keys | Secret leakage |
| Use Parquet or `.npz` for data interchange, not pickle | Safe, portable, inspectable |
| Read credentials from environment variables or Colab Secrets only | Least-privilege |

`bandit -r src/ -ll` runs in CI and as a pre-commit hook and will catch most violations
automatically. Human PR review catches the rest.

---

## Running the test suite

```bash
pytest tests/ -v
```

The test suite requires the synthetic sample data to exist. Generate it first if needed:

```bash
python tests/generate_sample_data.py
```

---

## Updating dependencies

1. Edit `requirements.in` (add, remove, or bump a package).
2. Recompile the lockfile on Python ≥3.10:
   ```bash
   pip-compile --generate-hashes requirements.in -o requirements.txt
   ```
3. Run `pip-audit --requirement requirements.txt` to check for known vulnerabilities.
4. Update `requirements-colab.txt` to match.
5. Commit both `requirements.in` and `requirements.txt`.
