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

## Provenance

Built from the UCF-PVMCF module databases (`el-metadata`, `module-metadata`,
`sinton-iv-results`). I-V features come from `sinton-iv-results`, matched to each
EL image on (serial, date, pressure) with nearest-timestamp disambiguation.
