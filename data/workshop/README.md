# Workshop dataset — PV module health (I-V + EL)

A small, self-contained multimodal dataset for the case study: fuse electrical
**I-V measurements** with **electroluminescence (EL) images** to assess PV module
health under increasing mechanical load (pressure).

## Modules (4)

| sample_id (serial) | steps | pressure range | Pmp trend | behavior |
|---|---|---|---|---|
| `H5NE30994` | 28 | 0–5400 Pa | 319 → 316 W | robust |
| `609K3ZH1A4NA` | 28 | 0–5400 Pa | 321 → 315 W | robust |
| `321617132567500337` | 27 | 0–5100 Pa | 275 → 180 W | strongly degrades |
| `2617127550699` | 19 | 0–3600 Pa | 291 → 239 W | degrades (crack init.) |

(Make and model have been removed from this dataset; modules are identified by
serial number only.)

The mix of robust and degrading modules is intentional — it gives the regression
/ classification target real signal.

## Grain

One row per **(`sample_id`, `exposure_step`)** — i.e. one EL image and its paired
I-V measurement at a given pressure. `exposure_step` is a per-module index (1…N)
ordered by pressure then capture time; `pressure_pa` holds the physical value.
(An index is used because a couple of modules have two images at the same nominal
pressure — a retake and a before/after-initiation pair — which would otherwise
collide on pressure.)

## `metadata.csv` columns

- **Keys / provenance:** `sample_id`, `exposure_step`, `pressure_pa`,
  `el_image_filename`, `el_image_path`, `el_capture_datetime`, `el_note`
- **Module descriptors:** `module_id`, `cell_technology`,
  `number_busbars`, `cells_per_string`, `module_area_cm2`, `cell_area_cm2`,
  `nameplate_{pmp_w,voc_v,isc_a,vmp_v,imp_a}`
- **I-V features (Sinton):** `iv_isc_a`, `iv_voc_v`, `iv_imp_a`, `iv_vmp_v`,
  `iv_pmp_w`, `iv_ff_percent`, `iv_efficiency_percent`, `iv_rsh_ohm`,
  `iv_rs_ohm`, `iv_measured_temperature_c`, `iv_measurement_datetime`
- **Derived targets:** `pmp_normalized` (Pmp ÷ each module's initial Pmp),
  `power_loss_percent`
- **Match provenance:** `iv_match_type`, `iv_n_candidates` — how the I-V row was
  paired to the EL image (`exact_1to1` = unique at that pressure;
  `nearest_time` = nearest-timestamp pick among repeat flashes).

## Images — staged pipeline

`images/` mirrors the EL image-processing stages covered in the workshop:

| Stage | Folder | Contents |
|---|---|---|
| 1 | `01_input/` | Raw EL PNGs (committed) — one subfolder per module |
| 2 | `02_module_rectified/` | Perspective-corrected & cropped module images |
| 3 | `03_cells/` | Cropped, labeled individual cell images |
| 4 | `04_cell_masks/` | Cell images with semitransparent segmentation masks |
| 5 | `05_module_restitched/` | Re-stitched module images with masks overlaid |

The **input images are not committed to git** — they are distributed as a single
zip (`el_input_images.zip`, ~177 MB) attached to a GitHub Release, so the clone
stays small. Stages 2–5 are **generated during the workshop** (folders ship empty
with `.gitkeep`). EL features aggregated from stage 4/5 are merged back onto
`metadata.csv` in the modeling section.

## Getting the input images

`metadata.csv` (this dataframe) is committed and available on clone. The raw EL
PNGs are fetched on demand:

```python
# from the repo root
from scripts.download_workshop_data import download_data
download_data()          # downloads + unzips into images/01_input/<module>/
```

or on the command line: `python scripts/download_workshop_data.py`.

The downloader is **host-agnostic** — it takes a URL, so the GitHub Release asset
can be swapped for an OSF/Zenodo URL later without code changes
(`download_data(url="https://osf.io/<id>/download")`). It verifies a SHA-256
checksum and skips the download if the images are already present.

**Maintainer — publishing the asset:** rebuild the zip and attach it to the
release tag referenced in `scripts/download_workshop_data.py` (`TAG`):

```bash
zip -r -X el_input_images.zip data/workshop/images/01_input -x '*.DS_Store' -x '*/.gitkeep'
gh release create v0.1.0 el_input_images.zip --title "Workshop data v0.1.0" \
    --notes "EL input images for the 4 workshop modules (102 PNGs)."
```

If you rebuild the zip, update `EXPECTED_SHA256` in the downloader
(`shasum -a 256 el_input_images.zip`). Current asset SHA-256:
`b929cef3f813f3f4b7070bb5f6d274777ab39f5ce1cf45ebed4b3662b60b57cf`.

## Precomputed results

Running the full pipeline (segmenting all 7,452 cells) is a one-time ~1 h CPU job
(minutes on a GPU). Its outputs are shipped so nothing depends on live GPU compute:

- **`metadata_with_el_features.csv`** (committed) — `metadata.csv` plus the
  aggregated EL feature columns (`el_crack_frac_mean`, `el_defect_frac_total`,
  `el_frac_cells_cracked`, `el_crack_frac_max`, per-class means, `el_n_cells`).
  This is the table the notebook's modeling section reads, so it runs for every
  student regardless of GPU. Headline signal: `el_crack_frac_mean` vs
  `power_loss_percent` correlates at **r = 0.93**.
- **Restitched module masks** (optional, ~304 MB) — one defect-overlay image per
  EL image, published as the `el_precomputed_masks.zip` release asset:

  ```python
  from scripts.download_masks import download_masks
  download_masks()   # -> images/05_module_restitched/<module>/<image>.png
  ```

The per-cell masks (`04_cell_masks`) are intermediate and regenerated on demand.

## EL feature columns (`el_*`)

Every cell image is run through the segmentation model, which classifies each
pixel as **background** (no defect) or one of four defect classes: **crack**,
**contact**, **interconnect**, **corrosion**. A pixel is labeled background if
its predicted background probability is ≥ 0.52; otherwise it takes whichever
defect class has the highest probability. From that per-pixel mask, each cell
gets four **per-cell fractions** — for each defect class, the share of that
cell's pixels assigned to it. All `el_*` columns below are built by aggregating
these per-cell fractions across every cell in one EL photograph (60, 72, or 96
cells depending on the module).

| Column | Meaning | Calculation |
|---|---|---|
| `el_crack_frac_mean` | Average share of a cell's area classified as crack, averaged over the whole module image. The strongest single predictor of power loss (r ≈ 0.93). | Mean of the per-cell crack fraction across all cells in the image. |
| `el_contact_frac_mean` | Average per-cell share of pixels classified as a contact defect. | Mean of the per-cell contact fraction across all cells. |
| `el_interconnect_frac_mean` | Average per-cell share of pixels classified as an interconnect defect. | Mean of the per-cell interconnect fraction across all cells. |
| `el_corrosion_frac_mean` | Average per-cell share of pixels classified as corrosion. | Mean of the per-cell corrosion fraction across all cells. |
| `el_defect_frac_total` | Overall defect severity for the module image, combining all four defect types into one number instead of looking at crack alone. | Per cell, sum the four defect fractions into one per-cell total; then average that total across all cells. |
| `el_crack_frac_max` | The single worst cell's crack severity — catches a module that's fine everywhere except one badly cracked cell, which a mean would dilute. | Maximum (not average) of the per-cell crack fraction across all cells. |
| `el_frac_cells_cracked` | What fraction of the module's cells are meaningfully cracked at all — a prevalence measure rather than a severity measure. | A cell counts as cracked if its crack fraction exceeds 1% (`CRACKED_CELL_THRESHOLD` in `features.py`); this column is (# cracked cells) ÷ (total cells). |
| `el_n_cells` | How many cells were segmented for this image — a sanity/denominator column for spotting a partial or failed segmentation run. | Count of cell records grouped into that image. |

**Why both a mean and a max/prevalence version:** `el_crack_frac_mean` alone
can't distinguish "one severely cracked cell" from "mild cracking spread across
many cells" — both could average to the same number. `el_crack_frac_max` catches
the first case, `el_frac_cells_cracked` catches the second. `el_defect_frac_total`
exists because a module could show low crack area but meaningful contact/
interconnect/corrosion damage instead — summing before averaging keeps that
visible rather than requiring four separate mean columns to be checked.

These columns are produced by `rdstemplate.el.features.aggregate_cells`, grouped
by `(module, image)`, then joined onto `metadata.csv` by EL image filename
(`merge_features`) — see `src/rdstemplate/el/features.py`.

## Provenance

Built from the UCF-PVMCF module databases (`el-metadata`, `module-metadata`,
`sinton-iv-results`). I-V features come from `sinton-iv-results`, matched to each
EL image on (serial, date, pressure) with nearest-timestamp disambiguation.
