import os
import cv2
import torch
import numpy as np
from PIL import Image
from skimage import measure, morphology
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# ----------------------------
# Config
# ----------------------------
IMAGE_PATH = "input/building.jpg"
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR = "output"

MODEL_ID = "IDEA-Research/grounding-dino-base"

TEXT_PROMPT = "window . brick wall . concrete wall . glass window . door"
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "meshes"), exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def clean_binary_mask(mask: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Clean noisy mask with morphology + area filtering."""
    mask = mask.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    return mask.astype(np.uint8)


def mask_to_polygons(mask: np.ndarray, min_vertices: int = 6, simplify_tol: float = 2.0):
    """
    Convert binary mask to shapely polygons.
    Returns a list of valid polygons.
    """
    contours = measure.find_contours(mask, level=0.5)
    polygons = []

    for contour in contours:
        # skimage gives (row, col); shapely expects (x, y)
        contour_xy = np.flip(contour, axis=1)

        if len(contour_xy) < min_vertices:
            continue

        poly = Polygon(contour_xy)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            continue

        poly = poly.simplify(simplify_tol, preserve_topology=True)

        if poly.area < 500:
            continue

        polygons.append(poly)

    return polygons


def polygon_to_mesh(poly: Polygon, extrude_height: float = 0.2):
    """
    Extrude a shapely polygon into a trimesh mesh.
    """
    if poly.is_empty or poly.area <= 0:
        return None

    try:
        mesh = trimesh.creation.extrude_polygon(poly, height=extrude_height)
        return mesh
    except Exception as e:
        print(f"failed to extrude polygon: {e}")
        return None


def phrase_to_safe_name(phrase: str) -> str:
    return str(phrase).replace(" ", "_").replace("/", "_")


# ----------------------------
# Load models
# ----------------------------
print(f"Using device: {DEVICE}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# ----------------------------
# Load image
# ----------------------------
image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_source = np.array(image_pil)  # RGB numpy array

sam_predictor.set_image(image_source)

# ----------------------------
# Detect with Hugging Face Grounding DINO
# ----------------------------
text_labels = [["window", "brick wall", "concrete wall", "glass window", "door"]]

inputs = processor(images=image_pil, text=text_labels, return_tensors="pt")
inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs["input_ids"],
    threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    target_sizes=[image_pil.size[::-1]],
)[0]

boxes_xyxy = results["boxes"].detach().cpu().numpy()
phrases = results["labels"]
logits = results["scores"].detach().cpu().numpy()

print(f"Detected {len(boxes_xyxy)} objects")

# ----------------------------
# Segment with SAM
# ----------------------------
for idx, (box, phrase, logit) in enumerate(zip(boxes_xyxy, phrases, logits)):
    box = np.array(box, dtype=np.float32)

    masks, scores, _ = sam_predictor.predict(
        box=box,
        multimask_output=True
    )

    best_mask_idx = int(np.argmax(scores))
    mask = masks[best_mask_idx]
    mask = clean_binary_mask(mask, min_area=300)

    # save mask image
    mask_img = (mask * 255).astype(np.uint8)
    class_name = phrase_to_safe_name(phrase)
    mask_path = os.path.join(OUTPUT_DIR, "masks", f"{idx:03d}_{class_name}.png")
    cv2.imwrite(mask_path, mask_img)

    # convert mask -> polygons
    polygons = mask_to_polygons(mask)

    if not polygons:
        print(f"[{idx}] no valid polygons for {phrase}")
        continue

    # merge if multiple polygons overlap/touch
    merged = unary_union(polygons)

    # unary_union may return Polygon or MultiPolygon
    if merged.geom_type == "Polygon":
        merged_polys = [merged]
    else:
        merged_polys = list(merged.geoms)

    for j, poly in enumerate(merged_polys):
        mesh = polygon_to_mesh(poly, extrude_height=0.15)
        if mesh is None:
            continue

        mesh_path = os.path.join(
            OUTPUT_DIR, "meshes", f"{idx:03d}_{j:02d}_{class_name}.obj"
        )
        mesh.export(mesh_path)
        print(f"saved {mesh_path}")

print("done")