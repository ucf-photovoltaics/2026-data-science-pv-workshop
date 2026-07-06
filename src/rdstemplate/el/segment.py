"""Stage 4 — per-cell defect segmentation with DeepLabV3-ResNet50.

Loads the trained 5-class checkpoint (``model_97.pth``: background, crack,
contact, interconnect, corrosion), runs each cell image through the network,
applies the reference confidence threshold, and writes a semitransparent
mask overlay per cell. Also returns per-class defect pixel fractions, which
``features.py`` aggregates onto the dataframe.

Weights are fetched separately (``scripts/download_weights.py``) into
``data/workshop/models/model_97.pth``. On Colab this runs on GPU; on CPU it
works but is slow, so validate on a subset.

Typical use::

    from rdstemplate.el import segment
    model = segment.load_model("data/workshop/models/model_97.pth")
    segment.segment_dataset(model,
        "data/workshop/images/03_cells",
        "data/workshop/images/04_cell_masks")
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# class index -> name (0 = no defect / background)
CLASSES = ["background", "crack", "contact", "interconnect", "corrosion"]

# overlay colours for defect classes 1-4, BGR (match the reference colormap)
CLASS_COLORS_BGR = {
    1: (36, 26, 228),    # crack        - red
    2: (184, 126, 55),   # contact      - blue
    3: (163, 78, 152),   # interconnect - purple
    4: (0, 127, 255),    # corrosion    - orange
}

DEFAULT_WEIGHTS = "data/workshop/models/model_97.pth"
DEFECT_THRESHOLD = 0.52  # reference value: P(no-defect) below this => defect


def load_model(weights_path: str | Path = DEFAULT_WEIGHTS, *, device: str = "cpu"):
    """Build DeepLabV3-ResNet50 (5 classes) and load the trained weights."""
    import torch
    import torchvision
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead

    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier = DeepLabHead(2048, len(CLASSES))
    # checkpoint is a trusted, checksum-verified file (see download_weights.py)
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model


def predict_mask(model, gray: np.ndarray, *, threshold: float = DEFECT_THRESHOLD,
                 device: str = "cpu") -> np.ndarray:
    """Return an int mask (0 = no defect, 1-4 = defect class) for one cell."""
    import torch
    from torchvision import transforms as t

    trans = t.Compose([t.ToTensor(), t.Normalize(mean=0.5, std=0.2)])
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    x = trans(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)["out"][0]              # (5, H, W)
    soft = torch.softmax(out, dim=0)
    nodef = soft[0]
    mask = torch.where(nodef >= threshold,
                       torch.zeros_like(nodef, dtype=torch.long),
                       soft[1:].argmax(0) + 1)
    return mask.cpu().numpy().astype(np.uint8)


def overlay_mask(gray: np.ndarray, mask: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
    """Blend a semitransparent colour mask (defect classes only) over the cell."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mask.shape != gray.shape:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    for cls, color in CLASS_COLORS_BGR.items():
        sel = mask == cls
        if sel.any():
            vis[sel] = (alpha * np.array(color) + (1 - alpha) * vis[sel]).astype(np.uint8)
    return vis


def defect_fractions(mask: np.ndarray) -> dict[str, float]:
    """Per-class pixel fraction of a cell mask (crack/contact/interconnect/corrosion)."""
    total = mask.size
    return {CLASSES[c]: round(float(np.count_nonzero(mask == c) / total), 7)
            for c in range(1, len(CLASSES))}


def segment_file(model, src: str | Path, out_path: str | Path, *,
                 threshold: float = DEFECT_THRESHOLD, alpha: float = 0.4,
                 device: str = "cpu") -> dict[str, float]:
    """Segment one cell image, write its overlay, and return defect fractions."""
    src, out_path = Path(src), Path(out_path)
    gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    mask = predict_mask(model, gray, threshold=threshold, device=device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay_mask(gray, mask, alpha=alpha))
    return defect_fractions(mask)


def segment_dataset(model, input_root: str | Path, output_root: str | Path, *,
                    threshold: float = DEFECT_THRESHOLD, alpha: float = 0.4,
                    device: str = "cpu") -> list[dict]:
    """Segment every cell under ``input_root`` (03_cells layout) into ``output_root``.

    Returns one record per cell: {module, image, cell, crack, contact,
    interconnect, corrosion} — ready for ``features.py`` to aggregate.
    """
    input_root, output_root = Path(input_root), Path(output_root)
    records: list[dict] = []
    for module_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for image_dir in sorted(p for p in module_dir.iterdir() if p.is_dir()):
            for src in sorted(image_dir.glob("cell_*.png")):
                out = output_root / module_dir.name / image_dir.name / src.name
                frac = segment_file(model, src, out, threshold=threshold,
                                    alpha=alpha, device=device)
                records.append({"module": module_dir.name, "image": image_dir.name,
                                "cell": src.stem, **frac})
    return records
