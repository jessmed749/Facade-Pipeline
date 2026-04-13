"""
tests/regression_check.py

Compares output masks from the latest run against golden reference masks.
Fails if mean IoU across all matched classes drops below --iou-threshold.

Usage:
    python3 tests/regression_check.py --iou-threshold 0.80

Golden masks live in tests/golden/<class_name>.png
They are generated once from a verified good run via:
    cp output/masks/000_window.png tests/golden/window.png
    cp output/masks/001_door.png   tests/golden/door.png
    git add tests/golden/ && git commit -m "update golden masks"
"""

import os
import sys
import argparse
import numpy as np
import cv2

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
MASKS_DIR  = os.path.join("output", "masks")


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel-level IoU between two binary masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 1.0


def class_from_filename(fname: str) -> str:
    """Extract class label from filenames like '003_02_window.png'."""
    parts = os.path.splitext(fname)[0].split("_")
    # Skip leading index parts (all-digit segments)
    label_parts = [p for p in parts if not p.isdigit()]
    return "_".join(label_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou-threshold", type=float, default=0.80)
    args = parser.parse_args()

    if not os.path.isdir(GOLDEN_DIR):
        print(f"No golden masks found at {GOLDEN_DIR} — skipping regression.")
        sys.exit(0)

    golden_files = [f for f in os.listdir(GOLDEN_DIR) if f.endswith(".png")]
    if not golden_files:
        print("Golden mask directory is empty — skipping regression.")
        sys.exit(0)

    output_masks = [f for f in os.listdir(MASKS_DIR) if f.endswith(".png")]

    results = []
    for gf in golden_files:
        cls = os.path.splitext(gf)[0]  # e.g. "window"
        golden_path = os.path.join(GOLDEN_DIR, gf)
        golden_mask = cv2.imread(golden_path, cv2.IMREAD_GRAYSCALE)
        if golden_mask is None:
            print(f"  ⚠ Could not read golden mask: {gf}")
            continue

        # Find output masks that contain this class label
        matching = [f for f in output_masks if cls in class_from_filename(f)]
        if not matching:
            print(f"  ✗ No output mask found for class '{cls}'")
            results.append((cls, 0.0))
            continue

        # Merge all matching output masks into one
        combined = np.zeros_like(golden_mask, dtype=np.uint8)
        for mf in matching:
            m = cv2.imread(os.path.join(MASKS_DIR, mf), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape != golden_mask.shape:
                    m = cv2.resize(m, (golden_mask.shape[1], golden_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                combined = np.maximum(combined, m)

        iou = mask_iou(combined > 128, golden_mask > 128)
        status = "✓" if iou >= args.iou_threshold else "✗"
        print(f"  {status} {cls:<20}  IoU = {iou:.3f}  (threshold {args.iou_threshold})")
        results.append((cls, iou))

    if not results:
        print("No results to evaluate.")
        sys.exit(0)

    mean_iou = sum(iou for _, iou in results) / len(results)
    failures  = [(cls, iou) for cls, iou in results if iou < args.iou_threshold]

    print(f"\nMean IoU: {mean_iou:.3f}  |  Failures: {len(failures)}/{len(results)}")

    if failures:
        print("\nFailed classes:")
        for cls, iou in failures:
            print(f"  {cls}: {iou:.3f} < {args.iou_threshold}")
        sys.exit(1)

    print("Regression check passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()