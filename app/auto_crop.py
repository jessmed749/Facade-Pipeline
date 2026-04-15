"""
auto_crop.py — Automatic facade crop zone detection via a DINO preflight pass.

Instead of hard-coding CROP_*_FRACTION constants per photo, this module runs
a single DINO forward pass with broad "building facade" / "wall" prompts and
returns pixel crop bounds that replace the manual fractions.

Usage (inside run_facade_pipeline.py):

    from auto_crop import detect_facade_crop

    x_min_px, x_max_px, y_min_px, y_max_px = detect_facade_crop(
        image_pil, processor, grounding_model, DEVICE,
        W_px, H_px,
        padding_frac=0.02,   # small inward pad to avoid edge noise
    )

If no facade box is found, falls back to the manual CROP_*_FRACTION constants
so the pipeline degrades gracefully.
"""

import numpy as np
# torch is NOT imported at module level — it is imported lazily inside
# _run_dino() so that test_pipeline.py can import _largest_box and _fallback
# (pure NumPy helpers) without torch being installed.

# Prompts that reliably retrieve the main building facade box from DINO.
# Broad terms score higher than specific architectural features here.
FACADE_LABELS = [["building facade", "building wall", "wall", "building exterior"]]

# DINO thresholds for the preflight pass — lower than window detection because
# we want to capture the whole facade, not just high-confidence sub-regions.
FACADE_BOX_THRESHOLD  = 0.20
FACADE_TEXT_THRESHOLD = 0.15


def _run_dino(image_pil, processor, model, text_labels, box_thresh, text_thresh, device):
    """Single DINO forward pass; returns (boxes_np, labels, scores_np)."""
    import torch  # deferred — not needed by the pure helpers used in unit tests
    inputs = processor(images=image_pil, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return (
        results["boxes"].detach().cpu().numpy(),
        results["labels"],
        results["scores"].detach().cpu().numpy(),
    )


def _largest_box(boxes, scores):
    """
    Return the single best facade box.

    Strategy: take the highest-scoring box whose area is at least 10 % of the
    image area (to ignore small fragments).  If nothing qualifies, return the
    largest box by area regardless of score.
    """
    if len(boxes) == 0:
        return None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Prefer largest area first, break ties by score
    ranked = sorted(range(len(boxes)), key=lambda i: (areas[i], scores[i]), reverse=True)
    return boxes[ranked[0]]


def detect_facade_crop(
    image_pil,
    processor,
    grounding_model,
    device,
    W_px: int,
    H_px: int,
    padding_frac: float = 0.02,
    # Fallback fractions — used when DINO finds nothing
    fallback_top: float    = 0.08,
    fallback_bottom: float = 0.15,
    fallback_left: float   = 0.08,
    fallback_right: float  = 0.95,
):
    """
    Run a DINO preflight pass to auto-detect the facade bounding box.

    Returns
    -------
    (x_min_px, x_max_px, y_min_px, y_max_px) : tuple[int, int, int, int]
        Pixel crop bounds ready to use directly in the main pipeline.
    source : str
        "dino" if the box came from DINO, "fallback" if DINO found nothing.
    """
    print("Auto-crop: running DINO facade preflight pass …")

    boxes, labels, scores = _run_dino(
        image_pil, processor, grounding_model,
        FACADE_LABELS,
        FACADE_BOX_THRESHOLD,
        FACADE_TEXT_THRESHOLD,
        device,
    )

    print(f"  → {len(boxes)} facade candidate(s) detected")
    for i, (b, l, s) in enumerate(zip(boxes, labels, scores)):
        area_frac = (b[2]-b[0]) * (b[3]-b[1]) / (W_px * H_px)
        print(f"     [{i}] {str(l):<25}  score={s:.2f}  area={area_frac*100:.1f}%")

    best = _largest_box(boxes, scores)

    if best is None:
        print("  No facade box found — using fallback crop fractions.")
        return _fallback(W_px, H_px, fallback_top, fallback_bottom,
                         fallback_left, fallback_right), "fallback"

    x1, y1, x2, y2 = best

    # Inward padding keeps the crop just inside the detected facade boundary,
    # trimming the very edge pixels that often include adjacent structures.
    pad_x = int(W_px * padding_frac)
    pad_y = int(H_px * padding_frac)

    x_min = max(0,     int(x1) + pad_x)
    x_max = min(W_px,  int(x2) - pad_x)
    y_min = max(0,     int(y1) + pad_y)
    y_max = min(H_px,  int(y2) - pad_y)

    # Sanity check — if the box is degenerate, fall back
    if x_max - x_min < W_px * 0.10 or y_max - y_min < H_px * 0.10:
        print("  Facade box too small — using fallback crop fractions.")
        return _fallback(W_px, H_px, fallback_top, fallback_bottom,
                         fallback_left, fallback_right), "fallback"

    print(f"  Facade box  : x {x_min}–{x_max}  y {y_min}–{y_max} px")
    print(f"  Coverage    : {(x_max-x_min)/W_px*100:.0f}% wide  {(y_max-y_min)/H_px*100:.0f}% tall")
    return (x_min, x_max, y_min, y_max), "dino"


def _fallback(W, H, top, bottom, left, right):
    return (
        int(W * left),
        int(W * right),
        int(H * top),
        int(H * (1.0 - bottom)),
    )