# PV Module Health — a Multimodal Data Science Case Study

**2026 Data Science Workshop (Oxford)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ucf-photovoltaics/2026-data-science-pv-workshop/blob/v0.1.0/environments/colab/workshop_case_study.ipynb)
[![CI](https://github.com/ucf-photovoltaics/2026-data-science-pv-workshop/actions/workflows/ci.yml/badge.svg)](https://github.com/ucf-photovoltaics/2026-data-science-pv-workshop/actions/workflows/ci.yml)

**Click the badge above to open the workshop notebook in Colab — no local setup required.**

---

## The case study

Assess photovoltaic **module health** by fusing two data modalities — electrical
**I-V curves** and visual **electroluminescence (EL) images** — into a single
interpretable model. It's genuinely multimodal (parametric measurements + spatial
images), which is exactly the *data fusion* problem this workshop's workflow is
built to solve.

Four real modules were put through a mechanical-load (pressure) test, from 0 Pa up
to as much as 5400 Pa. At each pressure step we have a paired EL image and I-V
measurement — some modules stay robust throughout, others crack and lose over a
third of their power output, giving the case study real signal to model.

**Target & framing:** predict module performance loss from EL-derived defect
features. **Unit of analysis:** one module at one exposure (pressure) step.

## What the notebook covers

The notebook mirrors the workshop's lifecycle, one section per agenda segment:

| # | Section | What happens |
|---|---|---|
| 0 | Setup | Clone the repo, download the EL images + model weights (pinned release) |
| 1 | Problem Formulation | State the question, the target, and what "good enough" looks like |
| 2 | Physical & Digital Workflows | How the I-V and EL streams are measured and merged |
| 3 | Data Acquisition | Load the metadata table and an EL image |
| 4 | Pre-Processing & Feature Extraction | Run the EL segmentation pipeline live on one module |
| 5 | Data Fusion, Modeling & Analysis | Train an interpretable model on the precomputed features |
| 6 | Insights & Interpretation | Feature importance vs. PV physics; correlation vs. mechanism |
| 7 | Iterative Improvements | Where the workflow is weakest, and what to try next |

**No GPU required.** The full EL segmentation (7,452 cells) is precomputed and
shipped as data, so Section 5's modeling runs identically whether or not you have
one — a GPU only speeds up the live demo in Section 4. See
[`data/workshop/README.md`](data/workshop/README.md#precomputed-results) for details.

## Quickstart

### Option A — Google Colab (recommended)

Click the **Open In Colab** badge above. The notebook will:
1. Clone this repo at the pinned release tag (`v0.1.0`).
2. Download the EL images (~177 MB) and segmentation model weights (~336 MB)
   from the GitHub Release — both checksum-verified.
3. Walk through Sections 0–7 above.

> **Save a copy first:** `File → Save a copy in Drive` before editing, so your
> work is saved to your own Google account.

### Option B — Local

```bash
git clone --branch v0.1.0 https://github.com/ucf-photovoltaics/2026-data-science-pv-workshop.git
cd 2026-data-science-pv-workshop
pip install -r requirements-workshop.txt
jupyter notebook environments/colab/workshop_case_study.ipynb
```

## The dataset

4 modules, ~100 EL images + paired I-V measurements, one row per
`(module, exposure_step)`. Full schema, the 5-stage EL image pipeline, and the
precomputed feature columns are documented in
**[`data/workshop/README.md`](data/workshop/README.md)**.

## Repository layout

```
environments/colab/workshop_case_study.ipynb   the workshop notebook (start here)
data/workshop/                                 dataset: metadata.csv, images/, models/
                                                — see data/workshop/README.md
scripts/                                       download_workshop_data.py, download_weights.py,
                                                download_masks.py — release-asset fetchers
src/rdstemplate/el/                            EL image pipeline: rectify -> cells ->
                                                segment -> stitch -> features
configs/workshop.yaml                          pipeline config for the workshop dataset
```

This repo also carries the general-purpose `rdstemplate` framework
(metadata → feature extraction → merge → model) it was built from, including a
CLI, HPC stubs, and a synthetic demo dataset (`data/sample/`) — unrelated to the
workshop content above. See **[`docs/FRAMEWORK.md`](docs/FRAMEWORK.md)** for that
reference material, and [`CONTRIBUTING.md`](CONTRIBUTING.md) to extend it.

## License

MIT — see [LICENSE](LICENSE).
