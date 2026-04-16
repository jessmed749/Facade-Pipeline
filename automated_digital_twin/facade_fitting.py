import arcpy
import os
import json
import math
import glob
import config

WKID_XY = config.WKID_XY
WKID_Z = config.WKID_Z

SR = arcpy.SpatialReference(WKID_XY, WKID_Z)
WGS84_SR = arcpy.SpatialReference(4326)


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculates the shortest distance from a point to a line segment."""
    line_mag = math.hypot(x2 - x1, y2 - y1)
    if line_mag == 0:
        return math.hypot(px - x1, py - y1)

    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    u = max(min(u, 1.0), 0.0)

    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    return math.hypot(px - ix, py - iy)


def get_all_telemetry_payloads(source_dir):
    """Finds all facade JSONs and extracts their unique identifiers."""
    json_files = glob.glob(os.path.join(source_dir, "facade_*.json"))
    if not json_files:
        print(" -> No facade telemetry files found.")
        return []

    payloads = []
    for file in json_files:
        # Extract the <something> from facade_<something>.json
        identifier = os.path.basename(file).replace("facade_", "").replace(".json", "")
        with open(file, 'r') as f:
            data = json.load(f)
        payloads.append((identifier, data))
        print(f" -> Parsed telemetry for facade: {identifier}")

    return payloads


def execute_facade_raycast(camera_pt, heading, footprints_layer):
    """
    Casts a 2D ray from the camera to find the exact building and wall segment.
    Dynamically expands the wall to ignore fake/collinear vertices.
    """
    print(" -> Casting 2D raycast to identify target wall...")

    ray_dist = 1000.0
    heading_rad = math.radians(heading)
    dx = math.sin(heading_rad) * ray_dist
    dy = math.cos(heading_rad) * ray_dist

    ray_end = arcpy.Point(camera_pt.firstPoint.X + dx, camera_pt.firstPoint.Y + dy)
    ray_line = arcpy.Polyline(arcpy.Array([camera_pt.firstPoint, ray_end]), SR)

    ray_fc = "memory\\Camera_Ray"
    arcpy.management.CopyFeatures([ray_line], ray_fc)

    intersect_pts = "memory\\Ray_Intersections"
    arcpy.analysis.Intersect([ray_fc, footprints_layer], intersect_pts, join_attributes="ALL", output_type="POINT")

    if int(arcpy.management.GetCount(intersect_pts)[0]) == 0:
        raise ValueError("Raycast failed: Camera heading does not intersect any buildings within 1000m.")

    closest_dist = float('inf')
    target_bldg_id = None
    hit_point = None

    with arcpy.da.SearchCursor(intersect_pts, ["SHAPE@", "BUILDING_ID"]) as cursor:
        for geom, bldg_id in cursor:
            dist = camera_pt.distanceTo(geom)
            if dist < closest_dist:
                closest_dist = dist
                target_bldg_id = bldg_id
                hit_point = geom.firstPoint

    print(f" -> Ray hit BUILDING_ID: {target_bldg_id} at distance {closest_dist:.2f}m")

    target_bldg_layer = "memory\\Target_Building"
    arcpy.analysis.Select(footprints_layer, target_bldg_layer, f"BUILDING_ID = {target_bldg_id}")

    best_segment = None
    min_seg_dist = float('inf')
    all_segments = []

    with arcpy.da.SearchCursor(target_bldg_layer, ["SHAPE@"]) as cursor:
        for row in cursor:
            poly = row[0]
            for part in poly:
                for i in range(len(part) - 1):
                    p1, p2 = part[i], part[i + 1]
                    if p1 is None or p2 is None:
                        continue

                    all_segments.append((p1, p2))

                    dist = _point_to_segment_distance(
                        hit_point.X, hit_point.Y,
                        p1.X, p1.Y,
                        p2.X, p2.Y
                    )

                    if dist < min_seg_dist:
                        min_seg_dist = dist
                        best_segment = [p1, p2]

    if not best_segment:
        raise RuntimeError("Failed to resolve the wall segment geometry.")

    # Collinear wall expansion (sometimes walls have redundant vertices in the middle of them)
    p1, p2 = best_segment[0], best_segment[1]

    expanding = True
    while expanding:
        expanding = False
        # Calculate the compass angle of our current wall
        wall_angle = math.degrees(math.atan2(p2.Y - p1.Y, p2.X - p1.X))

        for seg_p1, seg_p2 in all_segments:
            # Calculate angle of the candidate segment
            seg_angle = math.degrees(math.atan2(seg_p2.Y - seg_p1.Y, seg_p2.X - seg_p1.X))
            angle_diff = abs((wall_angle - seg_angle + 180) % 360 - 180)

            # If it's parallel (within 2 degrees of our wall)
            if angle_diff < 2.0:

                # Does it connect exactly to our left corner (p1)?
                if abs(seg_p2.X - p1.X) < 0.01 and abs(seg_p2.Y - p1.Y) < 0.01:
                    p1 = seg_p1  # Expand the wall leftward
                    expanding = True
                    break  # Break the for-loop to recalculate the wall angle

                # Does it connect exactly to our right corner (p2)?
                elif abs(seg_p1.X - p2.X) < 0.01 and abs(seg_p1.Y - p2.Y) < 0.01:
                    p2 = seg_p2  # Expand the wall rightward
                    expanding = True
                    break

    arcpy.management.Delete(ray_fc)
    arcpy.management.Delete(target_bldg_layer)

    return target_bldg_id, p1, p2, hit_point


def extract_roof_heights(bldg_id, p1, p2, lod2_fc):
    """
    Extracts the max Z value for the specific wall corners using 3D multipatch vertices.
    """
    print(f" -> Extracting roof heights for BUILDING_ID: {bldg_id} corners...")

    target_lod2 = "memory\\Target_LOD2_Building"
    arcpy.analysis.Select(lod2_fc, target_lod2, f"BUILDING_ID = {bldg_id}")

    if int(arcpy.management.GetCount(target_lod2)[0]) == 0:
        raise ValueError(f"BUILDING_ID {bldg_id} is missing from the LOD2 layer.")

    temp_3d_points = "memory\\Temp_LOD2_Vertices"
    arcpy.management.FeatureVerticesToPoints(target_lod2, temp_3d_points, "ALL")
    arcpy.management.MakeFeatureLayer(temp_3d_points, "Vertices_Layer")

    def get_max_z_for_xy(corner_pt):
        pt_geom = arcpy.PointGeometry(corner_pt, SR)
        search_radii = [1.0, 3.0, 5.0, 10.0]  # Use search_radii = [5.0, 10.0] if you aren't finding true corner height
        max_z = None

        for radius in search_radii:
            arcpy.management.SelectLayerByLocation(
                in_layer="Vertices_Layer",
                overlap_type="WITHIN_A_DISTANCE",
                select_features=pt_geom,
                search_distance=f"{radius} Meters",
                selection_type="NEW_SELECTION"
            )

            if int(arcpy.management.GetCount("Vertices_Layer")[0]) > 0:
                z_values = []
                with arcpy.da.SearchCursor("Vertices_Layer", ["SHAPE@Z"]) as cursor:
                    for row in cursor:
                        if row[0] is not None:
                            z_values.append(row[0])

                if z_values:
                    max_z = max(z_values)
                    break

        if max_z is None:
            raise ValueError(f"No 3D vertices found within 10m of footprint corner.")

        return max_z

    z1 = get_max_z_for_xy(p1)
    z2 = get_max_z_for_xy(p2)

    arcpy.management.Delete("Vertices_Layer")
    arcpy.management.Delete(temp_3d_points)
    arcpy.management.Delete(target_lod2)

    return z1, z2


def extract_ground_height(pt, dem_raster):
    """
    Samples the underlying DEM raster to find the exact bare-earth elevation at an XY point.
    """
    try:
        # GetCellValue requires the coordinates as a space-separated string
        coord_string = f"{pt.X} {pt.Y}"
        result = arcpy.management.GetCellValue(dem_raster, coord_string)

        # The result object returns a string, which could be 'NoData'
        z_str = result.getOutput(0)

        if not z_str or z_str.strip().lower() == 'nodata':
            print(f"    - Warning: DEM returned NoData at {pt.X:.2f}, {pt.Y:.2f}. Defaulting to 0.0.")
            return 0.0

        return float(z_str)

    except Exception as e:
        print(f"    - Warning: DEM sampling failed at {pt.X:.2f}, {pt.Y:.2f}. Defaulting to 0.0. Error: {e}")
        return 0.0


def calculate_outward_normal(p1, p2, camera_geom):
    """Calculates the outward-facing normal vector of the wall segment."""
    dx = p2.X - p1.X
    dy = p2.Y - p1.Y

    n1 = (-dy, dx)
    n2 = (dy, -dx)

    cx = (p1.X + p2.X) / 2.0
    cy = (p1.Y + p2.Y) / 2.0

    cam_x = camera_geom.firstPoint.X
    cam_y = camera_geom.firstPoint.Y
    v_cam = (cx - cam_x, cy - cam_y)

    dot1 = (n1[0] * v_cam[0]) + (n1[1] * v_cam[1])
    dot2 = (n2[0] * v_cam[0]) + (n2[1] * v_cam[1])

    chosen_n = n1 if dot1 < 0 else n2

    mag = math.hypot(chosen_n[0], chosen_n[1])
    return (chosen_n[0] / mag, chosen_n[1] / mag)


def anchor_single_facade(payload, footprints_fc, lod2_fc, dem_raster):
    """
    Executes the anchoring logic for a single parsed payload.
    Outputs exactly 3 anchor points to prevent mesh warping on stepped walls.
    """
    telemetry = payload['device_telemetry']
    mesh_info = payload['mesh_metadata']

    cam_wgs = arcpy.PointGeometry(arcpy.Point(telemetry['longitude'], telemetry['latitude']), WGS84_SR)
    cam_proj = cam_wgs.projectAs(SR)

    bldg_id, bottom_p1, bottom_p2, hit_pt = execute_facade_raycast(
        camera_pt=cam_proj,
        heading=telemetry['heading_degrees'],
        footprints_layer=footprints_fc
    )

    roof_z1, roof_z2 = extract_roof_heights(bldg_id, bottom_p1, bottom_p2, lod2_fc)

    print(f"    - Extracting ground base heights for BUILDING_ID: {bldg_id} corners...")
    ground_z1 = extract_ground_height(bottom_p1, dem_raster)
    ground_z2 = extract_ground_height(bottom_p2, dem_raster)

    anchor_coords = {
        "Bottom_Left": (bottom_p1.X, bottom_p1.Y, ground_z1),
        "Bottom_Right": (bottom_p2.X, bottom_p2.Y, ground_z2)
    }

    # Only keep the top corner with the minimum Z value
    if roof_z1 <= roof_z2:
        print(f"    - Stepped roof detected. Anchoring to Top_Left min Z: {roof_z1:.2f}")
        anchor_coords["Top_Left"] = (bottom_p1.X, bottom_p1.Y, roof_z1)
    else:
        print(f"    - Stepped roof detected. Anchoring to Top_Right min Z: {roof_z2:.2f}")
        anchor_coords["Top_Right"] = (bottom_p2.X, bottom_p2.Y, roof_z2)

    outward_normal = calculate_outward_normal(bottom_p1, bottom_p2, cam_proj)

    blender_anchoring_data = {
        "building_id": bldg_id,
        "mesh_file": mesh_info['file_name'],
        "front_facing_axis": mesh_info.get('front_facing_axis', '+Y'),
        "anchors": anchor_coords,
        "wall_normal_2d": outward_normal,
        "camera_distance_m": cam_proj.distanceTo(hit_pt)
    }

    return blender_anchoring_data


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = os.path.join(BASE_DIR, "source_data")
    WORKSPACE_GDB = os.path.join(BASE_DIR, "data.gdb")
    EXPORTS_DIR = os.path.join(BASE_DIR, "Exports")

    arcpy.env.workspace = WORKSPACE_GDB
    arcpy.env.overwriteOutput = True

    FOOTPRINTS = "ATX_Buildings_2023_Clipped"
    LOD2_BUILDINGS = "ATX_LiDAR_2021_LOD2_primary"
    TERRAIN_DEM = "DEM_Triangulated_028_Raster_Clipped"

    print("\n--- INITIATING FACADE MESH ANCHORING ---")

    try:
        payloads = get_all_telemetry_payloads(SOURCE_DIR)

        for identifier, payload in payloads:
            print(f"\n -> Processing Facade: {identifier}")
            anchor_data = anchor_single_facade(
                payload=payload,
                footprints_fc=FOOTPRINTS,
                lod2_fc=LOD2_BUILDINGS,
                dem_raster=TERRAIN_DEM
            )

            export_json_path = os.path.join(EXPORTS_DIR, f"facade_anchoring_{identifier}.json")
            with open(export_json_path, 'w') as f:
                json.dump(anchor_data, f, indent=4)

            print(f" -> Saved anchoring instructions to facade_anchoring_{identifier}.json")

    except Exception as e:
        print(f"CRITICAL ERROR in Anchoring Pipeline: {e}")
