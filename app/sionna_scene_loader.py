import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')
import json
import os

# Sionna imports (requires: pip install sionna)
try:
    import sionna
    from sionna.rt import (
        Scene, Transmitter, Receiver,
        PlanarArray, load_scene,
        RadioMaterial,
    )
    import tensorflow as tf
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("⚠  Sionna not installed — showing scene description only.\n"
          "   pip install sionna tensorflow")

OUTPUT_DIR  = "output"
JSON_PATH   = os.path.join(OUTPUT_DIR, "sionna_scene.json")

# Sionna built-in material names
#   See: sionna.rt.scene.MATERIALS
#   Common options: "itu_concrete", "itu_brick", "itu_glass", "itu_wood",
#                   "itu_marble", "itu_metal", "itu_very_dry_ground"
MATERIAL_MAP = {
    "itu_glass":    "itu_glass",
    "itu_brick":    "itu_brick",
    "itu_concrete": "itu_concrete",
    "itu_wood":     "itu_wood",
}


def load_facade_scene(json_path: str = JSON_PATH):
    with open(json_path) as f:
        desc = json.load(f)

    print(f"Scene: {desc['scene_name']}")
    print(f"  px/m scale : {desc['px_to_meter']*100:.3f} cm/px")
    print(f"  image size : {desc['image_size_px']} px")
    print(f"  wall height: {desc['wall_height_m']} m")
    print(f"  objects    : {len(desc['objects'])}\n")

    # Group per-class obj files
    per_class_dir = os.path.join(OUTPUT_DIR, "per_class")
    class_files = {}
    for fname in os.listdir(per_class_dir):
        if fname.endswith(".obj"):
            cls = fname.replace(".obj", "")
            class_files[cls] = os.path.join(per_class_dir, fname)
    input_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "input")
    for fname in os.listdir(input_dir):
        if fname.endswith(".obj"):
            cls = os.path.splitext(fname)[0]  # filename without extension becomes the class name
            class_files[cls] = os.path.join(input_dir, fname)

    print("Per-class meshes found:")
    for cls, path in class_files.items():
        print(f"  {cls:<20} → {path}")

    if not SIONNA_AVAILABLE:
        print("\nSionna not available, stopping here.")
        return None

    # Build Sionna Scene
    # Sionna expects a Mitsuba XML scene file.
    # We generate one programmatically pointing at our .obj files.
    xml_path = _write_mitsuba_xml(class_files, desc)
    print(f"\nMitsuba XML → {xml_path}")

    scene = load_scene(xml_path)

    # Assign EM materials from our JSON
    for obj in desc["objects"]:
        obj_id       = obj["id"]
        sionna_mat   = obj["sionna_material"]
        # Sionna object names match mesh file names (without extension)
        mesh_name    = obj["mesh_file"].replace(".obj", "")
        try:
            scene.get(mesh_name).radio_material = sionna_mat
        except Exception as e:
            pass  # object may be merged into per-class mesh

    # Assign materials to per-class merged objects
    label_to_mat = {
        "window":        "itu_glass",
        "glass_window":  "itu_glass",
        "door":          "itu_wood",
        "brick_wall":    "itu_brick",
        "concrete_wall": "itu_concrete",
        "pma_building":  "itu_concrete",
    }
    for cls_name, mat in label_to_mat.items():
        try:
            scene.get(cls_name).radio_material = mat
            print(f"  Assigned {mat} → {cls_name}")
        except Exception:
            pass

    print("\nScene ready for ray tracing.")
    return scene


def _write_mitsuba_xml(class_files: dict, desc: dict) -> str:
    label_to_mat = {
        "window":        "glass",
        "glass_window":  "glass",
        "window_pane":   "glass",
        "window_frame":  "concrete",
        "door":          "wood",
        "pma_building":  "concrete",
    }

    # Collect unique ITU material types needed (BSDF id == type name, no mat- prefix)
    used_mats = dict.fromkeys(
        label_to_mat.get(cls, "concrete") for cls in class_files
    )

    lines = ['<scene version="2.1.0">', '', '<!-- Materials -->', '']

    for mat in used_mats:
        lines += [
            f'    <bsdf type="itu-radio-material" id="{mat}">',
            f'        <string name="type" value="{mat}"/>',
            f'        <float name="thickness" value="0.2"/>',
            f'    </bsdf>',
            '',
        ]

    lines += ['<!-- Shapes -->', '']

    for cls_name, obj_path in class_files.items():
        abs_path = os.path.abspath(obj_path)
        mat = label_to_mat.get(cls_name, "concrete")
        lines += [
            f'    <shape type="obj" id="mesh-{cls_name}">',
            f'        <string name="filename" value="{abs_path}"/>',
            f'        <boolean name="face_normals" value="true"/>',
            f'        <ref id="{mat}" name="bsdf"/>',
            f'    </shape>',
            '',
        ]

    lines += ['</scene>']

    xml_path = os.path.join(OUTPUT_DIR, "sionna_scene.xml")
    with open(xml_path, "w") as f:
        f.write("\n".join(lines))
    return xml_path

if __name__ == "__main__":
    scene = load_facade_scene()