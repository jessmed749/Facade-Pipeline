"""
Facade Segmentation Pipeline — windows and doors only, building surfaces only.
  - Exports combined_scene.obj  (the facade mesh, facing -Y axis)
  - Exports facade_<scene_name>.json with device_telemetry + mesh_metadata
  - GPS lat/lon and heading are extracted from the input image EXIF metadata
    automatically; if EXIF is missing, falls back to config values.
  - Crop zones are AUTO-DETECTED via a DINO facade preflight pass
    (see app/auto_crop.py) — manual CROP_*_FRACTION values are only used
    as a fallback when DINO finds nothing.
  - Wall height is AUTO-QUERIED from OpenStreetMap using EXIF GPS coordinates
    (see app/osm_height.py) — KNOWN_WALL_HEIGHT_M is the fallback.
"""

# ── Lightweight imports only at module level ──────────────────────────────────
# cv2, torch, transformers, and SAM are NOT imported here.
# They are loaded lazily in _load_runtime_deps() so that test_pipeline.py can
# import all pure helper functions (box_iou, nms, mask_to_polygons, etc.)
# without needing those packages installed (they are absent from
# requirements-dev.txt by design — no GPU in the unit-test runner).
import os
import json
import math
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from skimage import measure, morphology
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh

# Placeholders — replaced by _load_runtime_deps() at runtime
cv2 = None
torch = None
AutoProcessor = None
AutoModelForZeroShotObjectDetection = None
sam_model_registry = None
SamPredictor = None
detect_facade_crop = None
lookup_building_height = None


def _load_runtime_deps():
    """Import all GPU/model dependencies. Call once before running the pipeline."""
    global cv2, torch
    global AutoProcessor, AutoModelForZeroShotObjectDetection
    global sam_model_registry, SamPredictor
    global detect_facade_crop, lookup_building_height

    import cv2 as _cv2
    cv2 = _cv2

    import torch as _torch
    torch = _torch

    from transformers import (
        AutoProcessor as _AP,
        AutoModelForZeroShotObjectDetection as _AM,
    )
    AutoProcessor = _AP
    AutoModelForZeroShotObjectDetection = _AM

    from segment_anything import sam_model_registry as _sr, SamPredictor as _sp
    sam_model_registry = _sr
    SamPredictor = _sp

    from auto_crop import detect_facade_crop as _dc
    detect_facade_crop = _dc

    from osm_height import lookup_building_height as _lh
    lookup_building_height = _lh


# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH     = "input/building.jpg"
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR     = "output"
MODEL_ID       = "IDEA-Research/grounding-dino-base"
SCENE_NAME     = "combined_scene"

# Fallback telemetry — used ONLY if the image has no GPS EXIF data
FALLBACK_LATITUDE        = 30.288753
FALLBACK_LONGITUDE       = -97.736348
FALLBACK_HEADING_DEGREES = 94.67

# ── Detection thresholds ──────────────────────────────────────────────────────
BOX_THRESHOLD  = 0.18
TEXT_THRESHOLD = 0.15

USE_TILING     = True
TILE_SIZE      = 512
TILE_OVERLAP   = 128
NMS_IOU_THRESH = 0.25

# ── Manual crop fallbacks (used only when DINO auto-crop finds nothing) ───────
CROP_BOTTOM_FRACTION = 0.15
CROP_TOP_FRACTION    = 0.08
CROP_LEFT_FRACTION   = 0.08
CROP_RIGHT_FRACTION  = 0.95

# ── Scale fallback (used only when OSM lookup returns nothing) ────────────────
KNOWN_WALL_HEIGHT_M = 28.0

# ── Geometry filters ──────────────────────────────────────────────────────────
MIN_BOX_AREA_FRACTION = 0.0003
MAX_BOX_AREA_FRACTION = 0.015
MIN_ASPECT = 0.2
MAX_ASPECT = 5.0

KEEP_CLASSES = {
    "window", "glass window", "window frame", "window pane",
    "door", "building window", "building door",
}

TEXT_LABELS = [[
    "window", "glass window", "window frame", "window pane",
    "door", "building window", "building door",
]]

EXTRUDE_DEPTH_M = {
    "window": 0.05, "glass window": 0.05, "window frame": 0.08,
    "window pane": 0.05, "building window": 0.05,
    "door": 0.10, "building door": 0.10, "default": 0.05,
}

SIONNA_MATERIAL = {
    "window": "itu_glass", "glass window": "itu_glass",
    "window frame": "itu_concrete", "window pane": "itu_glass",
    "building window": "itu_glass", "door": "itu_wood",
    "building door": "itu_wood", "default": "itu_glass",
}

# NOTE: DEVICE is resolved inside main() after torch is loaded.
# It is NOT set at module level so that importing this file never touches torch.


# ── EXIF / Telemetry ──────────────────────────────────────────────────────────

def _dms_to_decimal(dms, ref):
    def to_float(v):
        if isinstance(v, tuple):
            return v[0] / v[1] if v[1] else 0.0
        try:
            return float(v)
        except Exception:
            return 0.0
    deg, mn, sec = (to_float(x) for x in dms)
    decimal = deg + mn / 60.0 + sec / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_telemetry(image_path):
    lat = lon = heading = None
    try:
        img = Image.open(image_path)
        exif_raw = img._getexif()
        if exif_raw:
            exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}
            gps_info_raw = exif.get("GPSInfo")
            if gps_info_raw:
                gps = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}
                if all(k in gps for k in ("GPSLatitude", "GPSLatitudeRef",
                                           "GPSLongitude", "GPSLongitudeRef")):
                    lat = _dms_to_decimal(gps["GPSLatitude"],  gps["GPSLatitudeRef"])
                    lon = _dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
                for tag in ("GPSImgDirection", "GPSTrack", "GPSDestBearing"):
                    if tag in gps:
                        v = gps[tag]
                        try:
                            heading = float(v)
                        except Exception:
                            try:
                                heading = v[0] / v[1]
                            except Exception:
                                pass
                        if heading is not None:
                            break
    except Exception as e:
        print(f"  EXIF read error: {e}")

    source = "exif"
    if lat is None or lon is None:
        print("  No GPS in EXIF — using fallback values.")
        lat, lon = FALLBACK_LATITUDE, FALLBACK_LONGITUDE
        source = "fallback"
    if heading is None:
        print("  No heading in EXIF — using fallback heading.")
        heading = FALLBACK_HEADING_DEGREES
        if source == "exif":
            source = "fallback (heading only)"

    return {"latitude": lat, "longitude": lon,
            "heading_degrees": heading, "source": source}


def write_facade_json(scene_name, obj_filename, telemetry, output_dir):
    payload = {
        "device_telemetry": {
            "latitude":        telemetry["latitude"],
            "longitude":       telemetry["longitude"],
            "heading_degrees": telemetry["heading_degrees"],
        },
        "mesh_metadata": {
            "file_name":         obj_filename,
            "front_facing_axis": "+Y",
        },
    }
    out_path = os.path.join(output_dir, f"facade_{scene_name}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ── Pure helpers (no torch / cv2 dependency) ──────────────────────────────────

def compute_pixel_to_meter(image_h_px, known_height_m):
    return known_height_m / image_h_px


def clean_binary_mask(mask, min_area=300):
    mask = mask.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    return mask.astype(np.uint8)


def mask_to_polygons(mask, px2m, min_area_m2=0.05, simplify_tol_px=2.0):
    h_px = mask.shape[0]
    polygons = []
    for contour in measure.find_contours(mask, level=0.5):
        xy = np.flip(contour, axis=1)
        xy[:, 0] *= px2m
        xy[:, 1] = (h_px - xy[:, 1]) * px2m
        if len(xy) < 6:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        poly = poly.simplify(simplify_tol_px * px2m, preserve_topology=True)
        if poly.area < min_area_m2:
            continue
        polygons.append(poly)
    return polygons


def polygon_to_mesh(poly, extrude_m):
    if poly.is_empty or poly.area <= 0:
        return None
    try:
        return trimesh.creation.extrude_polygon(poly, height=extrude_m)
    except Exception as e:
        print(f"   extrude failed: {e}")
        return None


def normalize_phrase(phrase):
    for kc in ["door", "window"]:
        if kc in phrase:
            return kc
    return phrase


def phrase_matches(phrase, keep_classes):
    return any(kc in phrase for kc in keep_classes)


def phrase_to_safe(phrase):
    return str(phrase).strip().lower().replace(" ", "_").replace("/", "_")


def get_extrude(phrase):
    return EXTRUDE_DEPTH_M.get(phrase.lower(), EXTRUDE_DEPTH_M["default"])


def get_sionna_mat(phrase):
    return SIONNA_MATERIAL.get(phrase.lower(), SIONNA_MATERIAL["default"])


def box_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def nms(boxes, phrases, scores, iou_thresh=0.4):
    if len(boxes) == 0:
        return boxes, phrases, scores
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        order = np.array([j for j in order[1:] if box_iou(boxes[i], boxes[j]) <= iou_thresh])
    keep = np.array(keep)
    return boxes[keep], [phrases[k] for k in keep], scores[keep]


def get_tiles(W, H, tile_size, overlap):
    tiles = []
    y = 0
    while y < H:
        y1 = min(y + tile_size, H)
        x = 0
        while x < W:
            x1 = min(x + tile_size, W)
            tiles.append((x, y, x1, y1))
            if x1 == W:
                break
            x += tile_size - overlap
        if y1 == H:
            break
        y += tile_size - overlap
    return tiles


def detect_on_image(image_pil, processor, model, text_labels, device):
    inputs = processor(images=image_pil, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return (results["boxes"].detach().cpu().numpy(),
            results["labels"],
            results["scores"].detach().cpu().numpy())


def is_valid_box(box, W, H, x_min_px, x_max_px, y_min_px, y_max_px,
                 min_area_frac, max_area_frac, min_asp, max_asp):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1
    if cx < x_min_px or cx > x_max_px:
        return False, "outside crop zone"
    if cy < y_min_px or cy > y_max_px:
        return False, "outside crop zone"
    box_area = bw * bh
    if box_area < min_area_frac * W * H:
        return False, "too small"
    if box_area > max_area_frac * W * H:
        return False, "too large"
    aspect = bw / bh if bh > 0 else 999
    if aspect < min_asp or aspect > max_asp:
        return False, "bad aspect"
    return True, ""


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # DEVICE must be resolved here, after torch has been loaded by _load_runtime_deps()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    for sub in ["masks", "meshes", "per_class"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    print(f"\n{'─'*52}")
    print(f" Facade Pipeline  |  device={DEVICE}")
    print(f"{'─'*52}\n")

    # Step 0: Extract telemetry from EXIF
    print("Extracting telemetry from image EXIF …")
    telemetry = extract_telemetry(IMAGE_PATH)
    print(f"  source  : {telemetry['source']}")
    print(f"  lat     : {telemetry['latitude']}")
    print(f"  lon     : {telemetry['longitude']}")
    print(f"  heading : {telemetry['heading_degrees']}°\n")

    # Step 1: Load image
    image_pil    = Image.open(IMAGE_PATH).convert("RGB")
    image_source = np.array(image_pil)
    H_px, W_px   = image_source.shape[:2]

    # Step 2: Auto-resolve wall height from OSM
    wall_height_m, height_source = lookup_building_height(
        lat=telemetry["latitude"],
        lon=telemetry["longitude"],
        fallback_m=KNOWN_WALL_HEIGHT_M,
    )
    print(f"  Wall height: {wall_height_m} m  (source: {height_source})\n")
    px2m = compute_pixel_to_meter(H_px, wall_height_m)

    # Step 3: Load models
    print("Loading Grounding DINO …")
    processor       = AutoProcessor.from_pretrained(MODEL_ID)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

    print("Loading SAM …")
    sam           = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    sam_predictor.set_image(image_source)

    # Step 4: Auto-detect crop zone via DINO facade preflight
    (x_min_px, x_max_px, y_min_px, y_max_px), crop_source = detect_facade_crop(
        image_pil, processor, grounding_model, DEVICE,
        W_px, H_px,
        padding_frac=0.02,
        fallback_top=CROP_TOP_FRACTION,
        fallback_bottom=CROP_BOTTOM_FRACTION,
        fallback_left=CROP_LEFT_FRACTION,
        fallback_right=CROP_RIGHT_FRACTION,
    )
    print(f"  Crop source : {crop_source}")
    print(f"  ROI         : x {x_min_px}–{x_max_px}  y {y_min_px}–{y_max_px} px")
    print(f"  Scale       : {px2m*100:.3f} cm/px  (wall={wall_height_m} m)\n")

    print(f"Filters: aspect {MIN_ASPECT}–{MAX_ASPECT}  |  "
          f"box area {MIN_BOX_AREA_FRACTION*100:.2f}%–{MAX_BOX_AREA_FRACTION*100:.1f}%")
    print(f"Classes: {sorted(KEEP_CLASSES)}\n")

    # Step 5: Detection
    all_boxes, all_phrases, all_scores = [], [], []
    crop_W = x_max_px - x_min_px

    building_crop = image_pil.crop((x_min_px, y_min_px, x_max_px, y_max_px))
    print("Running DINO on building crop (full res) …")
    b, p, s = detect_on_image(building_crop, processor, grounding_model, TEXT_LABELS, DEVICE)
    for box in b:
        all_boxes.append([box[0]+x_min_px, box[1]+y_min_px,
                          box[2]+x_min_px, box[3]+y_min_px])
    all_phrases.extend(p)
    all_scores.extend(s)
    print(f"  → {len(b)} detections")

    if USE_TILING:
        tiles = get_tiles(crop_W, y_max_px - y_min_px, TILE_SIZE, TILE_OVERLAP)
        print(f"\nRunning DINO on {len(tiles)} tiles of building zone …")
        for tx0, ty0, tx1, ty1 in tiles:
            tile_pil = image_pil.crop((tx0+x_min_px, ty0+y_min_px,
                                       tx1+x_min_px, ty1+y_min_px))
            tb, tp, ts = detect_on_image(tile_pil, processor, grounding_model, TEXT_LABELS, DEVICE)
            for box in tb:
                all_boxes.append([box[0]+tx0+x_min_px, box[1]+ty0+y_min_px,
                                   box[2]+tx0+x_min_px, box[3]+ty0+y_min_px])
            all_phrases.extend(tp)
            all_scores.extend(ts)
        print(f"  → {len(all_boxes)} raw detections before filtering")

    from collections import Counter
    label_counts = Counter(str(p).lower().strip() for p in all_phrases)
    print("\nDINO label distribution:")
    for label, count in label_counts.most_common(20):
        print(f"  {label:<30} {count}")

    # Step 6: Filter
    filtered_boxes, filtered_phrases, filtered_scores = [], [], []
    rejected = {"class": 0, "crop": 0, "size": 0, "aspect": 0}

    for box, phrase, score in zip(all_boxes, all_phrases, all_scores):
        phrase = str(phrase).lower().strip()
        if not phrase_matches(phrase, KEEP_CLASSES):
            rejected["class"] += 1
            continue
        phrase = normalize_phrase(phrase)
        ok, reason = is_valid_box(box, W_px, H_px, x_min_px, x_max_px,
                                   y_min_px, y_max_px,
                                   MIN_BOX_AREA_FRACTION, MAX_BOX_AREA_FRACTION,
                                   MIN_ASPECT, MAX_ASPECT)
        if not ok:
            key = "crop" if "crop" in reason else ("aspect" if "aspect" in reason else "size")
            rejected[key] += 1
            continue
        filtered_boxes.append(box)
        filtered_phrases.append(phrase)
        filtered_scores.append(score)

    print(f"\nFiltering: kept {len(filtered_boxes)}  |  "
          f"rejected class={rejected['class']}  crop={rejected['crop']}  "
          f"size={rejected['size']}  aspect={rejected['aspect']}")

    # Step 7: NMS
    if filtered_boxes:
        fb = np.array(filtered_boxes)
        fs = np.array(filtered_scores)
        fb, filtered_phrases, fs = nms(fb, filtered_phrases, fs, NMS_IOU_THRESH)
        print(f"After NMS: {len(fb)} detections\n")
    else:
        fb, fs = np.zeros((0, 4)), np.array([])
        print("  No detections survived filtering. Try lowering BOX_THRESHOLD.\n")

    # Step 8: SAM + mesh export
    scene_meshes, metadata, class_polygons = [], [], {}

    for idx, (box, phrase, score) in enumerate(zip(fb, filtered_phrases, fs)):
        phrase = str(phrase)
        print(f"[{idx:03d}] {phrase:<22}  score={score:.2f}")

        box = np.clip(np.array(box, dtype=np.float32), [0,0,0,0], [W_px,H_px,W_px,H_px])
        masks, mask_scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(mask_scores))
        mask = clean_binary_mask(masks[best], min_area=300)

        mask[y_max_px:, :] = 0
        mask[:y_min_px, :] = 0
        mask[:, :x_min_px] = 0
        mask[:, x_max_px:] = 0

        safe = phrase_to_safe(phrase)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", f"{idx:03d}_{safe}.png"),
                    (mask*255).astype(np.uint8))

        polys = mask_to_polygons(mask, px2m)
        if not polys:
            print(f"  ↳ no valid polygons, skipping")
            continue

        merged = unary_union(polys)
        geoms  = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
        class_polygons.setdefault(phrase, []).extend(geoms)

        extrude_m  = get_extrude(phrase)
        sionna_mat = get_sionna_mat(phrase)

        for j, poly in enumerate(geoms):
            mesh = polygon_to_mesh(poly, extrude_m)
            if mesh is None:
                continue
            obj_path = os.path.join(OUTPUT_DIR, "meshes", f"{idx:03d}_{j:02d}_{safe}.obj")
            mesh.export(obj_path)
            scene_meshes.append((mesh, sionna_mat, phrase))
            metadata.append({
                "id": f"{idx:03d}_{j:02d}", "label": phrase,
                "sionna_material": sionna_mat, "extrude_m": extrude_m,
                "area_m2": round(poly.area, 4),
                "mesh_file": os.path.basename(obj_path),
            })

        print(f"  ↳ {len(geoms)} polygon(s)  mat={sionna_mat}  depth={extrude_m}m")

    # Step 9: Export meshes
    print("\nExporting …")
    obj_filename = f"{SCENE_NAME}.obj"

    if scene_meshes:
        combined = trimesh.util.concatenate([m for m, _, _ in scene_meshes])
        combined.export(os.path.join(OUTPUT_DIR, obj_filename))
        b2 = combined.bounds
        d  = b2[1] - b2[0]
        print(f"  Scene: X={d[0]:.2f}m  Y={d[1]:.2f}m  Z={d[2]:.2f}m")

    for cls, polys in class_polygons.items():
        safe = phrase_to_safe(cls)
        ext  = get_extrude(cls)
        ms   = [m for p in polys if (m := polygon_to_mesh(p, ext)) is not None]
        if not ms:
            continue
        trimesh.util.concatenate(ms).export(
            os.path.join(OUTPUT_DIR, "per_class", f"{safe}.obj"))
        print(f"  → per_class/{safe}.obj  ({len(ms)} meshes)")

    # Sionna scene JSON
    json.dump(
        {
            "scene_name": "ut_campus_facade",
            "px_to_meter": px2m,
            "image_size_px": [W_px, H_px],
            "wall_height_m": wall_height_m,
            "wall_height_source": height_source,
            "crop_source": crop_source,
            "crop_bounds_px": [x_min_px, x_max_px, y_min_px, y_max_px],
            "objects": metadata,
        },
        open(os.path.join(OUTPUT_DIR, "sionna_scene.json"), "w"),
        indent=2,
    )

    # Step 10: Facade anchor JSON
    facade_json_path = write_facade_json(SCENE_NAME, obj_filename, telemetry, OUTPUT_DIR)
    print(f"\n✓ Facade anchor JSON → {facade_json_path}")
    print(json.dumps({
        "device_telemetry": {
            "latitude":        telemetry["latitude"],
            "longitude":       telemetry["longitude"],
            "heading_degrees": telemetry["heading_degrees"],
        },
        "mesh_metadata": {
            "file_name":         obj_filename,
            "front_facing_axis": "+Y",
        },
    }, indent=2))

    print(f"\n✓ done  —  {len(metadata)} mesh regions exported")
    print(f"  height source : {height_source}")
    print(f"  crop source   : {crop_source}\n")


if __name__ == "__main__":
    _load_runtime_deps()
    main()