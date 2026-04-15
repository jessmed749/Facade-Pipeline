"""
tests/test_pipeline.py

Unit tests for pure functions — no GPU, no models, no network.
Run with:  pytest tests/ -v

Integration tests (require models) are marked @pytest.mark.integration
and skipped in CI unless --run-integration is passed.
"""

import sys
import os
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

# Make app/ importable without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))


# ── Import helpers directly (no GPU deps) ────────────────────────────────────

from run_facade_pipeline import (
    box_iou,
    nms,
    compute_pixel_to_meter,
    clean_binary_mask,
    mask_to_polygons,
    normalize_phrase,
    phrase_matches,
    phrase_to_safe,
    get_extrude,
    get_sionna_mat,
    is_valid_box,
    _dms_to_decimal,
    KEEP_CLASSES,
    EXTRUDE_DEPTH_M,
    SIONNA_MATERIAL,
)
from auto_crop import _largest_box, _fallback
from osm_height import lookup_building_height


# ── box_iou ──────────────────────────────────────────────────────────────────

class TestBoxIou:
    def test_perfect_overlap(self):
        b = [0, 0, 10, 10]
        assert box_iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = [0, 0, 5, 5]
        b = [10, 10, 20, 20]
        assert box_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = [0, 0, 10, 10]   # area 100
        b = [5, 0, 15, 10]   # area 100, intersection = 50
        # union = 100 + 100 - 50 = 150; iou = 50/150
        assert box_iou(a, b) == pytest.approx(50 / 150)

    def test_contained(self):
        outer = [0, 0, 20, 20]  # area 400
        inner = [5, 5, 15, 15]  # area 100
        # intersection = 100; union = 400; iou = 100/400 = 0.25
        assert box_iou(outer, inner) == pytest.approx(0.25)

    def test_symmetry(self):
        a = [0, 0, 8, 6]
        b = [4, 3, 12, 9]
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))


# ── nms ──────────────────────────────────────────────────────────────────────

class TestNms:
    def test_empty(self):
        boxes = np.zeros((0, 4))
        phrases = []
        scores = np.array([])
        b, p, s = nms(boxes, phrases, scores)
        assert len(b) == 0

    def test_single_box_kept(self):
        boxes = np.array([[0, 0, 10, 10]])
        phrases = ["window"]
        scores = np.array([0.9])
        b, p, s = nms(boxes, phrases, scores)
        assert len(b) == 1
        assert p[0] == "window"

    def test_duplicate_suppressed(self):
        # Two nearly identical boxes — only highest score survives
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
        phrases = ["window", "window"]
        scores = np.array([0.8, 0.9])
        b, p, s = nms(boxes, phrases, scores, iou_thresh=0.4)
        assert len(b) == 1
        assert s[0] == pytest.approx(0.9)

    def test_non_overlapping_both_kept(self):
        boxes = np.array([[0, 0, 5, 5], [100, 100, 110, 110]])
        phrases = ["window", "door"]
        scores = np.array([0.8, 0.7])
        b, p, s = nms(boxes, phrases, scores, iou_thresh=0.4)
        assert len(b) == 2


# ── pixel-to-meter ────────────────────────────────────────────────────────────

class TestPixelToMeter:
    def test_basic(self):
        px2m = compute_pixel_to_meter(1000, 10.0)
        assert px2m == pytest.approx(0.01)

    def test_round_trip(self):
        px2m = compute_pixel_to_meter(2000, 28.0)
        # 2000 pixels * px2m should equal 28 m
        assert 2000 * px2m == pytest.approx(28.0)


# ── GPS DMS → decimal ─────────────────────────────────────────────────────────

class TestDmsToDecimal:
    def test_north(self):
        dms = ((30, 1), (17, 1), (1965, 100))  # 30°17'19.65" N
        result = _dms_to_decimal(dms, "N")
        assert result == pytest.approx(30.288791666, rel=1e-5)

    def test_west_negative(self):
        dms = ((97, 1), (44, 1), (1285, 100))
        result = _dms_to_decimal(dms, "W")
        assert result < 0

    def test_south_negative(self):
        dms = ((10, 1), (0, 1), (0, 1))
        assert _dms_to_decimal(dms, "S") == pytest.approx(-10.0)


# ── Phrase helpers ────────────────────────────────────────────────────────────

class TestPhraseHelpers:
    def test_normalize_window_variants(self):
        for phrase in ["glass window", "building window", "window frame", "window pane"]:
            assert normalize_phrase(phrase) == "window"

    def test_normalize_door_variants(self):
        for phrase in ["door", "building door"]:
            assert normalize_phrase(phrase) == "door"

    def test_phrase_matches_keep(self):
        assert phrase_matches("glass window", KEEP_CLASSES)
        assert phrase_matches("door", KEEP_CLASSES)

    def test_phrase_matches_rejects(self):
        assert not phrase_matches("car", KEEP_CLASSES)
        assert not phrase_matches("tree", KEEP_CLASSES)
        assert not phrase_matches("sky", KEEP_CLASSES)

    def test_phrase_to_safe(self):
        assert phrase_to_safe("Glass Window") == "glass_window"
        assert phrase_to_safe("window/frame") == "window_frame"

    def test_get_extrude_window(self):
        assert get_extrude("window") == EXTRUDE_DEPTH_M["window"]

    def test_get_extrude_door(self):
        assert get_extrude("door") == EXTRUDE_DEPTH_M["door"]

    def test_get_extrude_default(self):
        assert get_extrude("unknown_class") == EXTRUDE_DEPTH_M["default"]

    def test_get_sionna_mat_window(self):
        assert get_sionna_mat("window") == "itu_glass"

    def test_get_sionna_mat_door(self):
        assert get_sionna_mat("door") == "itu_wood"


# ── is_valid_box ──────────────────────────────────────────────────────────────

class TestIsValidBox:
    W, H = 1000, 800
    x_min, x_max = 80, 920
    y_min, y_max = 64, 720

    def _check(self, box, **kwargs):
        defaults = dict(
            W=self.W, H=self.H,
            x_min_px=self.x_min, x_max_px=self.x_max,
            y_min_px=self.y_min, y_max_px=self.y_max,
            min_area_frac=0.0003, max_area_frac=0.015,
            min_asp=0.2, max_asp=5.0,
        )
        defaults.update(kwargs)
        return is_valid_box(box, **defaults)

    def test_valid_window(self):
        # 80×60 box centred inside ROI
        ok, _ = self._check([350, 300, 430, 360])
        assert ok

    def test_outside_crop_left(self):
        ok, reason = self._check([20, 300, 60, 360])
        assert not ok
        assert "crop" in reason

    def test_too_small(self):
        ok, reason = self._check([400, 400, 402, 402])
        assert not ok
        assert "small" in reason

    def test_too_large(self):
        ok, reason = self._check([0, 0, 900, 700])
        assert not ok
        assert "large" in reason

    def test_bad_aspect_too_wide(self):
        # 200 wide × 20 tall → aspect = 10, exceeds MAX_ASPECT=5.0
        # area = 4000 / 800000 = 0.5% — within the 0.03%–1.5% area bounds
        ok, reason = self._check([300, 390, 500, 410])
        assert not ok
        assert "aspect" in reason


# ── clean_binary_mask ────────────────────────────────────────────────────────

class TestCleanBinaryMask:
    def test_removes_tiny_blobs(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Tiny blob — 5×5 = 25 px < min_area 300
        mask[10:15, 10:15] = 1
        result = clean_binary_mask(mask, min_area=300)
        assert result.sum() == 0

    def test_keeps_large_region(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        # 50×50 = 2500 px — should survive
        mask[50:100, 50:100] = 1
        result = clean_binary_mask(mask, min_area=300)
        assert result.sum() > 0


# ── mask_to_polygons ──────────────────────────────────────────────────────────

class TestMaskToPolygons:
    def test_square_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1
        px2m = 0.01
        polys = mask_to_polygons(mask, px2m, min_area_m2=0.0)
        assert len(polys) >= 1
        assert all(isinstance(p, Polygon) for p in polys)

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        polys = mask_to_polygons(mask, 0.01)
        assert polys == []


# ── auto_crop helpers ────────────────────────────────────────────────────────

class TestAutoCrop:
    def test_largest_box_picks_biggest(self):
        boxes = np.array([
            [0, 0, 10, 10],    # area 100
            [0, 0, 50, 50],    # area 2500 — should win
            [0, 0, 5, 5],      # area 25
        ])
        scores = np.array([0.9, 0.5, 0.8])
        best = _largest_box(boxes, scores)
        assert list(best) == [0, 0, 50, 50]

    def test_largest_box_empty(self):
        assert _largest_box(np.zeros((0, 4)), np.array([])) is None

    def test_fallback_values(self):
        x_min, x_max, y_min, y_max = _fallback(1000, 800, 0.1, 0.2, 0.05, 0.95)
        assert x_min == 50
        assert x_max == 950
        assert y_min == 80
        assert y_max == 640


# ── OSM height lookup (network-mocked) ───────────────────────────────────────

class TestOsmHeight:
    def test_fallback_on_network_error(self):
        # Intentionally invalid URL will raise → should return fallback
        import osm_height as oh
        original_url = oh.OVERPASS_URL
        oh.OVERPASS_URL = "http://127.0.0.1:1"  # nothing listening
        try:
            h, src = lookup_building_height(30.0, -97.0, fallback_m=15.0, timeout_s=2)
            assert h == pytest.approx(15.0)
            assert src == "fallback"
        finally:
            oh.OVERPASS_URL = original_url


# ── Integration: synthetic image → detection count ───────────────────────────

@pytest.mark.integration
class TestIntegration:
    """Requires GPU models. Run with: pytest tests/ -v -m integration --run-integration"""

    @pytest.fixture(scope="class")
    def models(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        MODEL_ID = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        return processor, model

    def _make_synthetic_facade(self, n_windows=6, n_doors=2):
        """
        Create a synthetic 800×600 building facade:
          - Grey wall background
          - White window rectangles arranged in a grid
          - Brown door rectangles at the bottom
        """
        img = np.ones((600, 800, 3), dtype=np.uint8) * 180  # grey wall
        # Windows: 2 rows × 3 cols
        for row in range(2):
            for col in range(3):
                x1 = 80 + col * 220
                y1 = 80 + row * 180
                img[y1:y1+100, x1:x1+140] = (230, 240, 255)  # light blue-white
        # Doors at bottom
        for d in range(n_doors):
            x1 = 150 + d * 400
            img[400:560, x1:x1+100] = (101, 67, 33)  # brown
        return Image.fromarray(img)

    def test_synthetic_window_count(self, models):
        """Pipeline should detect all 6 synthetic windows (±1 tolerance)."""
        from run_facade_pipeline import detect_on_image, nms, phrase_matches, KEEP_CLASSES
        import torch
        processor, model = models
        image_pil = self._make_synthetic_facade()
        TEXT_LABELS = [["window", "glass window", "door"]]
        boxes, phrases, scores = detect_on_image(
            image_pil, processor, model, TEXT_LABELS, "cpu"
        )
        window_count = sum(1 for p in phrases if "window" in str(p).lower())
        assert 5 <= window_count <= 7, f"Expected ~6 windows, got {window_count}"