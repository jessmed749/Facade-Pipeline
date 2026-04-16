import json
import math
import bpy                      # Don't install this in your venv, Blender makes it available at runtime
from mathutils import Vector    # Don't install this in your venv, Blender makes it available at runtime
import os
import glob
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
import config


INPUT_DIR = config.EXPORTS_DIR
OUTPUT_DIR = config.PLY_DIR
SOURCE_DIR = config.SOURCE_DIR


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def clear_scene():
    """Wipes the default cube, cameras, and lights from the Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def parse_wld3_offset(wld3_path):
    """Reads the 2nd set of coordinates from the wld3 file."""
    if not os.path.exists(wld3_path):
        print(f" -> WARNING: No WLD3 file found at {wld3_path}. Assuming (0,0,0).")
        return 0.0, 0.0, 0.0

    with open(wld3_path, 'r') as f:
        content = f.read().replace('\n', ' ')

    parts = content.split()
    if len(parts) >= 2:
        coords = parts[1].split(',')
        if len(coords) == 3:
            return float(coords[0]), float(coords[1]), float(coords[2])

    return 0.0, 0.0, 0.0


def find_obj_in_category(category_folder):
    """Recursively searches for the .obj file inside the given export folder."""
    target_folder = os.path.join(INPUT_DIR, category_folder)

    if not os.path.exists(target_folder):
        return None

    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.lower().endswith('.obj'):
                return os.path.join(root, file)

    return None


def import_and_position(category_folder):
    """Imports the OBJ and applies the WLD3 offset directly to the location."""
    obj_path = find_obj_in_category(category_folder)
    if not obj_path:
        return None

    wld3_path = obj_path.replace(".obj", ".wld3")
    offset_x, offset_y, offset_z = parse_wld3_offset(wld3_path)

    print(f" -> Importing {category_folder} and shifting by WLD3: X:{offset_x}, Y:{offset_y}, Z:{offset_z}")

    bpy.ops.object.select_all(action='DESELECT')

    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)

    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if not imported_objects:
        return None

    # If ArcGIS split the export, join it back into one object
    if len(imported_objects) > 1:
        bpy.context.view_layer.objects.active = imported_objects[0]
        bpy.ops.object.join()
        main_obj = bpy.context.view_layer.objects.active
    else:
        main_obj = imported_objects[0]

    # Apply the WLD3 offset
    main_obj.location = (offset_x, offset_y, offset_z)

    return main_obj


def import_and_fit_facades():
    """Finds all facade anchoring JSONs, imports them, and fits them from the Center."""
    facade_objects = []

    anchor_jsons = glob.glob(os.path.join(INPUT_DIR, "facade_anchoring_*.json"))

    for anchor_json in anchor_jsons:
        identifier = os.path.basename(anchor_json).replace("facade_anchoring_", "").replace(".json", "")

        with open(anchor_json, 'r') as f:
            data = json.load(f)

        mesh_file = data['mesh_file']
        source_obj_path = os.path.join(SOURCE_DIR, mesh_file)

        if not os.path.exists(source_obj_path):
            print(f" -> ERROR: Facade mesh {source_obj_path} not found for {identifier}.")
            continue

        print(f" -> Importing and anchoring custom facade: {identifier} ({mesh_file})")
        bpy.ops.object.select_all(action='DESELECT')

        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=source_obj_path)
        else:
            bpy.ops.import_scene.obj(filepath=source_obj_path)

        imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if not imported_objects:
            continue

        facade = imported_objects[0]
        bpy.context.view_layer.objects.active = facade

        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        anchors = data['anchors']
        normal = data['wall_normal_2d']

        raw_p1 = Vector(anchors['Bottom_Left'])
        raw_p2 = Vector(anchors['Bottom_Right'])
        nx, ny = normal[0], normal[1]

        # 1. Determine true viewer-left and calculate the wall dimensions
        left_vec = Vector((ny, -nx, 0))
        if raw_p2.dot(left_vec) > raw_p1.dot(left_vec):
            bl_coord = raw_p2
            br_coord = raw_p1
            top_z = anchors.get('Top_Right', anchors.get('Top_Left'))[2]
        else:
            bl_coord = raw_p1
            br_coord = raw_p2
            top_z = anchors.get('Top_Left', anchors.get('Top_Right'))[2]

        target_width = (br_coord - bl_coord).length
        target_height = top_z - bl_coord[2]

        # Calculate the absolute 3D center of the target wall
        wall_center = (bl_coord + br_coord) / 2.0
        wall_center.z = bl_coord.z + (target_height / 2.0)

        # 2. Re-origin the mesh to its exact geometric center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        # 3. Scale mesh based on its native orientation axis
        facing_axis = data.get('front_facing_axis', '+Y')
        if 'Y' in facing_axis:
            current_width = facade.dimensions.x
            if current_width > 0: facade.scale.x = target_width / current_width
        else:
            current_width = facade.dimensions.y
            if current_width > 0: facade.scale.y = target_width / current_width

        current_height = facade.dimensions.z
        if current_height > 0: facade.scale.z = target_height / current_height

        # Bake the scale so the thickness measurement is accurate
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Measure thickness after scaling, but before rotation
        if 'Y' in facing_axis:
            thickness = facade.dimensions.y
        else:
            thickness = facade.dimensions.x

        # 4. Dynamically rotate mesh
        angle = math.atan2(ny, nx)
        if facing_axis == '-Y':
            facade.rotation_euler[2] = angle + (math.pi / 2)
        elif facing_axis == '+Y':
            facade.rotation_euler[2] = angle - (math.pi / 2)
        elif facing_axis == '+X':
            facade.rotation_euler[2] = angle
        elif facing_axis == '-X':
            facade.rotation_euler[2] = angle + math.pi

        # Bake the rotation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # 5. Translate to center + apply the Z-fighting offset
        # We apply a small buffer to try to clear the noise in walls (e.g., if the center of the building wall erroneously sticks further out than it should relative to the edges of the wall)
        buffer_distance = 0.04
        total_push_out = (thickness / 2.0) + buffer_distance

        push_out_vector = Vector((nx * total_push_out, ny * total_push_out, 0.0))

        facade.location = wall_center + push_out_vector

        # 6. Save the final location
        # This locks the vertices in global space so the main pipeline cannot override the push
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Add to our list so we can export it later
        facade_objects.append((identifier, facade))

    return facade_objects


def run_pipeline():
    clear_scene()

    print("\n--- IMPORTING & APPLYING ABSOLUTE POSITIONS ---")
    terrain_grass_obj = import_and_position("terrain_grass")
    terrain_imp_obj = import_and_position("terrain_impervious")
    buildings_obj = import_and_position("buildings")
    basements_obj = import_and_position("basements")

    # Returns a list of tuples: [("PMA", object1), ("Tower", object2)]
    facade_data_list = import_and_fit_facades()
    facade_objs = [f[1] for f in facade_data_list]

    # Group all active objects for easy looping
    all_objs = [obj for obj in [terrain_grass_obj, terrain_imp_obj, buildings_obj, basements_obj] + facade_objs if obj]

    anchor_obj = terrain_grass_obj if terrain_grass_obj else terrain_imp_obj

    if not anchor_obj:
        print("\n -> ERROR: No terrain found. Cannot calculate global center point.")
        return

    print("\n--- CENTERING & BAKING COORDINATES ---")
    print(" -> Snapping 3D Cursor to the mathematical center of the Terrain...")
    bpy.ops.object.select_all(action='DESELECT')
    anchor_obj.select_set(True)
    bpy.context.view_layer.objects.active = anchor_obj

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    bpy.context.scene.cursor.location = anchor_obj.location

    print(" -> Locking all structures (including facades) to the terrain's center origin...")
    for obj in all_objs:
        if obj and obj != anchor_obj:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

    print(" -> Moving all structures to global (0,0,0)...")
    for obj in all_objs:
        if obj:
            obj.location = (0.0, 0.0, 0.0)

    print(" -> Applying all transforms...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if basements_obj:
        print("\n--- PROCESSING BASEMENTS ---")
        bpy.ops.object.select_all(action='DESELECT')
        basements_obj.select_set(True)
        bpy.context.view_layer.objects.active = basements_obj

        mod = basements_obj.modifiers.new(name="Solidify", type='SOLIDIFY')
        mod.thickness = 50.0
        mod.offset = -1.0
        bpy.ops.object.modifier_apply(modifier=mod.name)

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    print("\n--- FINAL PLY EXPORTS ---")

    def export_ply(obj, filename):
        if not obj: return
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        out_path = os.path.join(OUTPUT_DIR, filename)

        print(f" -> Exporting {filename}...")
        if hasattr(bpy.ops.wm, "ply_export"):
            bpy.ops.wm.ply_export(
                filepath=out_path,
                export_selected_objects=True,
                export_normals=True,
                export_triangulated_mesh=True
            )
        else:
            bpy.ops.export_mesh.ply(
                filepath=out_path,
                use_selection=True,
                use_normals=True,
                use_mesh_modifiers=True
            )

    export_ply(terrain_grass_obj, "Terrain_Grass.ply")
    export_ply(terrain_imp_obj, "Terrain_Impervious.ply")
    export_ply(buildings_obj, "Buildings.ply")
    export_ply(basements_obj, "Basements.ply")

    for identifier, f_obj in facade_data_list:
        export_ply(f_obj, f"Facade_{identifier}.ply")

    print("\n==================================")
    print("ALL BLENDER POST-PROCESSING COMPLETE")
    print("==================================")


if __name__ == "__main__":
    run_pipeline()
