"""
Digital Twin Recreation Pipeline
Target Environments: ArcGIS Pro 3.6.1 (arcpy) & Blender 5.1
Coordinate System: NAD 1983 (2011) StatePlane Texas Central FIPS 4203 (Meters) - WKID: 6577
Vertical System: NAVD88 height (Meters) - WKID: 5703
"""
import shutil
import sys
import arcpy
import os
import config
import time
import multiprocessing
import subprocess
import json
from facade_fitting import get_all_telemetry_payloads, anchor_single_facade


SR = arcpy.SpatialReference(config.WKID_XY, config.WKID_Z)


def _lod2_worker(args):
    """
    Isolated worker function for extracting LOD2 geometries.
    Re-imports arcpy and sets strict local environments to prevent schema locks.
    """
    workspace, bldg_chunk_fc, lasd_file, chunk_id, out_suffix, lod_params = args

    # Every worker must re-import arcpy natively
    import arcpy

    # Set own workspace to isolate the environment
    arcpy.env.workspace = workspace
    arcpy.env.overwriteOutput = True

    print(f"[Worker {chunk_id}] Initializing chunk {bldg_chunk_fc}...")

    # Write to completely separate temporary output feature classes to avoid write-locks
    lod2_out = f"Temp_LOD2_{out_suffix}_{chunk_id}"

    # Unique MakeLasDatasetLayer to avoid LAS read-locks
    las_layer = f"Filtered_LAS_Layer_{out_suffix}_{chunk_id}"

    try:
        # Create an isolated LAS layer targeting only necessary classes
        arcpy.management.MakeLasDatasetLayer(
            in_las_dataset=lasd_file,
            out_layer=las_layer,
            class_code=["1", "2", "6", "17", "20"]
        )

        arcpy.env.parallelProcessingFactor = "100%"

        arcpy.ddd.ExtractLOD2Buildings(
            in_height_source=las_layer,
            in_features=bldg_chunk_fc,
            out_feature_class=lod2_out,
            level_of_detail=lod_params['lod'],
            smoothness_level=lod_params['smoothness'],
            extraction_accuracy=lod_params['accuracy']
        )

        print(f"[Worker {chunk_id}] Repairing 3D multipatch topology...")
        arcpy.management.RepairGeometry(
            in_features=lod2_out,
            delete_null="DELETE_NULL",
            validation_method="ESRI"
        )

        # Clean up the memory layer to release the file handle early
        if arcpy.Exists(las_layer):
            arcpy.management.Delete(las_layer)

        print(f"[Worker {chunk_id}] Successfully finished {lod2_out}.")
        return lod2_out

    except Exception as e:
        print(f"[Worker {chunk_id}] CRITICAL ERROR: {e}")
        return None


def _isolated_export_worker(args):
    """
    Spawns a completely fresh ArcPy environment with zero lingering memory
    or schema locks, exactly mimicking a 'restart' of the script.
    """
    workspace, fc_path, target_folder = args
    import arcpy
    arcpy.env.workspace = workspace

    try:
        arcpy.management.Export3DObjects(
            in_features=fc_path,
            target_folder=target_folder,
            formats="FMT3D_OBJ",
            overwrite="OVERWRITE",
            merge="NO_MERGE"
        )
        return True
    except Exception as e:
        print(f"    - Subprocess Export Error: {e}")
        return False


def setup_environment():
    print("Setting up workspace environment...")

    if not arcpy.Exists(config.WORKSPACE_GDB):
        print(f"Creating Geodatabase at {config.WORKSPACE_GDB}")
        arcpy.management.CreateFileGDB(os.path.dirname(config.WORKSPACE_GDB), os.path.basename(config.WORKSPACE_GDB))

    arcpy.env.workspace = config.WORKSPACE_GDB
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = SR


# ==========================================
# STEP 1: AREA OF INTEREST
# ==========================================
def create_aoi_polygon():
    print("Step 1: Generating AOI Polygon...")
    aoi_name = "UT_Campus_AOI"
    aoi_path = os.path.join(config.WORKSPACE_GDB, aoi_name)

    coords = config.AOI_COORDINATE_VERTICES

    # Create empty polygon feature class
    arcpy.management.CreateFeatureclass(
        out_path=config.WORKSPACE_GDB,
        out_name=aoi_name,
        geometry_type="POLYGON",
        has_m="DISABLED",
        has_z="ENABLED",
        spatial_reference=SR
    )

    # Insert the polygon geometry into the feature class
    with arcpy.da.InsertCursor(aoi_path, ["SHAPE@"]) as cursor:
        array = arcpy.Array([arcpy.Point(x, y, z) for x, y, z in coords])
        polygon = arcpy.Polygon(array, SR, has_z=True)
        cursor.insertRow([polygon])

    print(f" -> Created {aoi_name}")
    return aoi_name


# ==============================================
# STEP 2: Download and Clip Building Footprints
# ==============================================
def download_austin_planimetrics(aoi_feature):
    print("Step 2: Downloading & Clipping Planimetrics from REST API...")

    buildings_url = "https://maps.austintexas.gov/arcgis/rest/services/Shared/PlanimetricsSurvey_1/MapServer/0"
    impervious_url = "https://maps.austintexas.gov/arcgis/rest/services/Shared/PlanimetricsSurvey_1/MapServer/1"

    bldg_raw = "ATX_Buildings_2023_Raw"
    cover_raw = "ATX_Impervious_2023_Raw"

    with arcpy.EnvManager(extent=aoi_feature, outputCoordinateSystem=SR):
        print(" -> Fetching Building Footprints from Server...")
        arcpy.management.MakeFeatureLayer(buildings_url, "Buildings_Web_Layer")
        arcpy.management.SelectLayerByLocation("Buildings_Web_Layer", "INTERSECT", aoi_feature)
        arcpy.conversion.ExportFeatures("Buildings_Web_Layer", bldg_raw)

        print(" -> Fetching Impervious Cover from Server...")
        arcpy.management.MakeFeatureLayer(impervious_url, "Impervious_Web_Layer")
        arcpy.management.SelectLayerByLocation("Impervious_Web_Layer", "INTERSECT", aoi_feature)
        arcpy.conversion.ExportFeatures("Impervious_Web_Layer", cover_raw)

    print(" -> Precision clipping to exact AOI boundary...")
    bldg_final = "ATX_Buildings_2023_Clipped"
    cover_final = "Impervious_Cover_2023_Projected_Clipped"

    arcpy.analysis.PairwiseClip(bldg_raw, aoi_feature, bldg_final, precision="MAX_PRECISION")
    arcpy.analysis.PairwiseClip(cover_raw, aoi_feature, cover_final, precision="MAX_PRECISION")

    # Assign each building a non-volatile unique ID, since OBJECTID will sometimes be changed by geoprocessing tools
    arcpy.management.CalculateField(
        in_table=bldg_final,
        field="BUILDING_ID",
        expression="!OBJECTID!",
        expression_type="PYTHON3",
        field_type="LONG"
    )

    print(f" -> Planimetrics downloaded and clipped successfully.")
    return bldg_final, cover_final


# ==========================================
# STEP 3: LAS DATASET
# ==========================================
def create_las_dataset(aoi_feature, skip=False):
    if skip:
        return os.path.join(config.BASE_DIR, "ATX_LIDAR_2021.lasd")

    print("Step 3: Creating LAS Dataset...")
    lasd_out = os.path.join(config.BASE_DIR, "ATX_LIDAR_2021.lasd")

    laz_files = [os.path.join(config.LAS_FOLDER, f) for f in os.listdir(config.LAS_FOLDER) if f.endswith('.laz')]
    if not laz_files:
        print(" -> Warning: No .laz files found in the specified LAS_FOLDER.")
        return None

    arcpy.management.CreateLasDataset(
        input=laz_files,
        out_las_dataset=lasd_out,
        folder_recursion="NO_RECURSION",
        spatial_reference=SR,
        compute_stats="COMPUTE_STATS",
        relative_paths="RELATIVE_PATHS",
        create_las_prj="NO_FILES",
        boundary=aoi_feature,
        add_only_contained_files="INTERSECTED_FILES"
    )
    print(f" -> LAS Dataset created at {lasd_out}")
    return lasd_out


# ==========================================
# STEP 4: CUSTOM DEM & SURFACE GENERATION
# ==========================================
def create_custom_dem(aoi_feature, bldg_clipped, cover_clipped, lasd_file, skip=False):
    if skip:
        return "DEM_Triangulated_028_Raster_Clipped"

    print("Step 4: Creating Custom DEM & Surface...")

    CRS_STRING = 'COMPOUNDCRS["",PROJCRS["NAD_1983_2011_StatePlane_Texas_Central_FIPS_4203",BASEGEOGCRS["GCS_NAD_1983_2011",DYNAMIC[FRAMEEPOCH[2010.0],MODEL["HTDP"]],DATUM["D_NAD_1983_2011",ELLIPSOID["GRS_1980",6378137.0,298.257222101],ANCHOREPOCH[2010.0]],PRIMEM["Greenwich",0.0],CS[ellipsoidal,2],AXIS["Latitude (lat)",north,ORDER[1]],AXIS["Longitude (lon)",east,ORDER[2]],ANGLEUNIT["Degree",0.0174532925199433]],CONVERSION["Lambert_Conformal_Conic",METHOD["Lambert_Conformal_Conic"],PARAMETER["False_Easting",700000.0],PARAMETER["False_Northing",3000000.0],PARAMETER["Central_Meridian",-100.3333333333333],PARAMETER["Standard_Parallel_1",30.11666666666667],PARAMETER["Standard_Parallel_2",31.88333333333333],PARAMETER["Latitude_Of_Origin",29.66666666666667]],CS[Cartesian,2],AXIS["Easting (X)",east,ORDER[1]],AXIS["Northing (Y)",north,ORDER[2]],LENGTHUNIT["Meter",1.0]],VERTCRS["NAVD_1988",VDATUM["North_American_Vertical_Datum_1988"],CS[vertical,1],AXIS["Gravity-related height (H)",up,LENGTHUNIT["Meter",1.0]]]]'

    print(" -> Calculating Minimum Bounding Rectangle for environment extents...")
    desc = arcpy.Describe(aoi_feature)
    pad = 2.0
    extent_str = f"{desc.extent.XMin - pad} {desc.extent.YMin - pad} {desc.extent.XMax + pad} {desc.extent.YMax + pad} {CRS_STRING}"
    dem_sampling_area = config.DEM_SAMPLING_AREA

    # --- Part 1: Impervious and Grass Polygons ---
    grass_clipped = "Ground_Grass_Clipped"
    arcpy.analysis.PairwiseErase(
        in_features=aoi_feature,
        erase_features=cover_clipped,
        out_feature_class=grass_clipped,
        cluster_tolerance=None
    )

    grass_dissolved = "Grass_Ground_Clipped_Dissolved"
    arcpy.analysis.PairwiseDissolve(
        in_features=grass_clipped,
        out_feature_class=grass_dissolved,
        dissolve_field=None,
        statistics_fields=None,
        multi_part="MULTI_PART",
        concatenation_separator=""
    )

    cover_dissolved = "Impervious_Cover_Clipped_Dissolved"
    arcpy.analysis.PairwiseDissolve(
        in_features=cover_clipped,
        out_feature_class=cover_dissolved,
        dissolve_field=None,
        statistics_fields=None,
        multi_part="MULTI_PART",
        concatenation_separator=""
    )

    surface_union = "Impervious_and_Grass_Clipped_Dissolved_Union"
    arcpy.analysis.Union(
        in_features=f"{cover_dissolved} 1;{grass_dissolved} 2",
        out_feature_class=surface_union,
        join_attributes="ALL",
        cluster_tolerance=None,
        gaps="GAPS"
    )

    # --- Part 2: Filter LAS & Rasterize ---
    filtered_las_dem = "Filtered_LAS_DEM_Layer"
    arcpy.management.MakeLasDatasetLayer(
        in_las_dataset=lasd_file,
        out_layer=filtered_las_dem,
        class_code=["2", "17", "20"]
    )

    dem_raster = "DEM_Triangulated_028_Raster"
    with arcpy.EnvManager(extent=extent_str):
        arcpy.conversion.LasDatasetToRaster(
            in_las_dataset=filtered_las_dem,
            out_raster=dem_raster,
            value_field="ELEVATION",
            interpolation_type="TRIANGULATION LINEAR NO_THINNING",
            data_type="FLOAT",
            sampling_type="CELLSIZE",
            sampling_value=dem_sampling_area,
            z_factor=1
        )

    # --- Part 3: Clip Raster ---
    dem_raster_clipped = "DEM_Triangulated_028_Raster_Clipped"
    arcpy.management.Clip(
        in_raster=dem_raster,
        rectangle=extent_str,
        out_raster=dem_raster_clipped,
        in_template_dataset=surface_union,
        nodata_value="3.4e+38",
        clipping_geometry="ClippingGeometry",
        maintain_clipping_extent="MAINTAIN_EXTENT"
    )

    print(" -> Creating 2D Fishnet Grid for high-res terrain sampling...")
    desc = arcpy.Describe(aoi_feature)
    origin = f"{desc.extent.XMin} {desc.extent.YMin}"
    y_axis = f"{desc.extent.XMin} {desc.extent.YMax}"

    fishnet = "Terrain_Fishnet"
    arcpy.management.CreateFishnet(
        out_feature_class=fishnet,
        origin_coord=origin,
        y_axis_coord=y_axis,
        cell_width=1,
        cell_height=1,
        labels="NO_LABELS",
        template=aoi_feature,
        geometry_type="POLYGON"
    )

    print(" -> Slicing Fishnet with Surface Boundaries...")
    fishnet_intersected = "Fishnet_Intersected"
    arcpy.analysis.PairwiseIntersect(
        in_features=[fishnet, surface_union],
        out_feature_class=fishnet_intersected,
        join_attributes="ALL"
    )

    print(" -> Classifying grid cells...")
    arcpy.management.AddField(fishnet_intersected, "Surface_Type", "TEXT")
    code_block = """def get_type(imp_id):
            return 'Impervious' if imp_id != -1 else 'Grass'"""
    arcpy.management.CalculateField(
        in_table=fishnet_intersected,
        field="Surface_Type",
        expression="get_type(!FID_Impervious_Cover_Clipped_Dissolved!)",
        expression_type="PYTHON3",
        code_block=code_block
    )

    print(" -> Draping dense grid over DEM Raster...")
    fishnet_3d = "Fishnet_3D"
    arcpy.ddd.InterpolateShape(
        in_surface=dem_raster_clipped,
        in_feature_class=fishnet_intersected,
        out_feature_class=fishnet_3d,
        z_factor=1,
        method="BILINEAR"
    )

    print(" -> Merging 3D grid into solid Multipatches...")
    dem_unioned_multipatch = "DEM_Unioned_Multipatch"
    arcpy.management.MakeFeatureLayer(fishnet_3d, "Fishnet_3D_Layer")
    arcpy.ddd.Layer3DToFeatureClass(
        in_feature_layer="Fishnet_3D_Layer",
        out_feature_class=dem_unioned_multipatch,
        group_field="Surface_Type",
        disable_materials="ENABLE_COLORS_AND_TEXTURES"
    )

    print(" -> Exporting High-Res 3D surfaces to OBJ for Blender...")
    for subfolder in ['terrain_grass', 'terrain_impervious']:
        folder_path = os.path.join(config.EXPORTS_DIR, subfolder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)

    arcpy.management.MakeFeatureLayer(dem_unioned_multipatch, "Multipatch_Layer")

    def export_with_subprocess(fc_name, surface_type, target_folder, max_retries=3):
        print(f" -> Preparing isolated export for {surface_type}...")

        # 1. Create the physical Feature Class
        arcpy.management.SelectLayerByAttribute("Multipatch_Layer", "NEW_SELECTION", f"Surface_Type = '{surface_type}'")
        arcpy.conversion.ExportFeatures("Multipatch_Layer", fc_name)
        arcpy.management.Add3DFormats(fc_name, "MULTIPATCH_WITH_MATERIALS", "FMT3D_OBJ")

        # 2. Clear locks
        arcpy.ClearWorkspaceCache_management()

        fc_absolute_path = os.path.join(arcpy.env.workspace, fc_name)

        for attempt in range(1, max_retries + 1):
            print(f"    - Spawning fresh process to export (Attempt {attempt}/{max_retries})...")

            # 3. Fire the export in a completely isolated process
            with multiprocessing.Pool(processes=1) as pool:
                pool.map(_isolated_export_worker, [(arcpy.env.workspace, fc_absolute_path, target_folder)])

            # 4. Search recursively for the .obj files
            print(f"    - Process finished. Scanning directory and subdirectories for output...")
            flush_successful = False

            # Sometimes it takes some time for the files to properly appear on disk
            for poll in range(20):
                for root, dirs, files in os.walk(target_folder):
                    for file in files:
                        if file.lower().endswith('.obj'):
                            test_file = os.path.join(root, file)
                            if os.path.getsize(test_file) > 0:
                                flush_successful = True
                                break
                    if flush_successful:
                        break

                if flush_successful:
                    break

                time.sleep(1.0)

            if flush_successful:
                print(f"    - Success! {surface_type} safely written to disk.")
                return
            else:
                print(f"    - Warning: Process completed but no files appeared after {poll + 1} seconds.")

            if attempt < max_retries:
                print("    - Pausing before retry to ensure completely severed locks...")
                time.sleep(3.0)

        raise RuntimeError(f"CRITICAL: Failed to export {surface_type} after {max_retries} attempts and buffer waits.")

    export_with_subprocess(
        fc_name="DEM_Grass_3D_Final",
        surface_type="Grass",
        target_folder=os.path.join(config.EXPORTS_DIR, 'terrain_grass')
    )

    export_with_subprocess(
        fc_name="DEM_Impervious_3D_Final",
        surface_type="Impervious",
        target_folder=os.path.join(config.EXPORTS_DIR, 'terrain_impervious')
    )

    if arcpy.Exists("Multipatch_Layer"):
        arcpy.management.Delete("Multipatch_Layer")

    return dem_raster_clipped


def shatter_by_height_tiers(lasd_file, bldg_clipped):
    print("Step 4.5: Shattering building footprints by height tiers...")

    height_tier_cutoff = config.HEIGHT_TIER_CUTOFF
    height_sampling_area = config.HEIGHT_SAMPLING_AREA

    filtered_las = "Bldg_LAS_Tiered"
    arcpy.management.MakeLasDatasetLayer(
        in_las_dataset=lasd_file,
        out_layer=filtered_las,
        class_code=["1", "2", "6", "20"]
    )

    print(" -> Generating Roof DSM (5m Natural Neighbor)...")
    roof_dsm = "Roof_DSM_Tiers"
    desc = arcpy.Describe(bldg_clipped)

    with arcpy.EnvManager(extent=desc.extent, outputCoordinateSystem=desc.spatialReference):
        arcpy.conversion.LasDatasetToRaster(
            in_las_dataset=filtered_las,
            out_raster=roof_dsm,
            value_field="ELEVATION",
            interpolation_type="BINNING MAXIMUM NATURAL_NEIGHBOR",
            data_type="FLOAT",
            sampling_type="CELLSIZE",
            sampling_value=height_sampling_area,
            z_factor=1
        )

    print(" -> Calculating height contours via native Map Algebra...")
    tiered_raster_path = "Tiered_Math_Raster"
    from arcpy.sa import Raster, Int

    with arcpy.EnvManager(scratchWorkspace=arcpy.env.workspace):
        tiered_math = Int(Raster(roof_dsm) / height_tier_cutoff)
        tiered_math.save(tiered_raster_path)

    print(" -> Converting tiers to cutting polygons...")
    tier_polys = "Building_Tier_Polygons"
    arcpy.conversion.RasterToPolygon(
        in_raster=tiered_raster_path,
        out_polygon_features=tier_polys,
        simplify="SIMPLIFY",
        raster_field="Value",
        create_multipart_features="SINGLE_OUTER_PART"
    )

    print(" -> Shattering footprints...")
    bldg_split_final = "ATX_Buildings_Tier_Segmented"
    arcpy.analysis.PairwiseIntersect(
        in_features=[tier_polys, bldg_clipped],
        out_feature_class=bldg_split_final,
        join_attributes="ALL"
    )

    print(f" -> Complex footprints successfully shattered into {bldg_split_final}")
    return bldg_split_final


# ==========================================
# STEP 5: PARALLEL LOD2 BUILDINGS
# ==========================================
def generate_lod2_parallel(lasd_file, bldg_clipped, final_lod2_out, out_suffix):
    """
    Manager function to chop footprints, spawn process pool, and merge results.
    """
    cores = config.PROCESSING_CORES
    lod_params = {
        'lod': config.LOD2_LEVEL_OF_DETAIL,
        'smoothness': config.LOD2_SMOOTHNESS,
        'accuracy': config.LOD2_ACCURACY
    }

    print(f" -> Preparing isolated parallel LOD2 extraction across {cores} physical cores...")

    # Dataset must be split into temporary footprint chunks using BUILDING_ID % N
    chunk_fcs = []
    print(" -> Chunking footprint datasets...")
    for i in range(cores):
        chunk_name = f"Temp_Bldg_Chunk_{out_suffix}_{i}"

        # SQL clause for File GDB Modulo
        where_clause = f"MOD(BUILDING_ID, {cores}) = {i}"

        arcpy.analysis.Select(bldg_clipped, chunk_name, where_clause)

        if int(arcpy.management.GetCount(chunk_name)[0]) > 0:
            # Package the args payload for the isolated worker
            chunk_fcs.append((arcpy.env.workspace, chunk_name, lasd_file, i, out_suffix, lod_params))
        else:
            arcpy.management.Delete(chunk_name)

    if not chunk_fcs:
        print(" -> No buildings to process.")
        return None

    print(f" -> Spawning native multiprocessing Pool ({len(chunk_fcs)} Active Chunks)...")
    completed_fcs = []
    with multiprocessing.Pool(processes=cores) as pool:
        results = pool.map(_lod2_worker, chunk_fcs)

    # Filter out crash results
    completed_fcs = [res for res in results if res is not None]

    if not completed_fcs:
        print(" -> CRITICAL ERROR: ALL POOL WORKERS FAILED.")
        return None

    # Merge outputs and delete temp chunk files
    print(f" -> Merging {len(completed_fcs)} isolated shards into {final_lod2_out}...")
    arcpy.management.Merge(completed_fcs, final_lod2_out, field_match_mode="AUTOMATIC")

    print(" -> Purging chunk footprints and temporary LOD2 generations...")
    for args in chunk_fcs:
        if arcpy.Exists(args[1]):
            arcpy.management.Delete(args[1])
    for fc in completed_fcs:
        if arcpy.Exists(fc):
            arcpy.management.Delete(fc)

    return final_lod2_out


def extract_lod2_and_basements(lasd_file, bldg_clipped, out_suffix='primary'):
    print(f"Step 5: Extracting LOD2 Buildings ({out_suffix})...")

    lod2_out = f"ATX_LiDAR_2021_LOD2_{out_suffix}"
    basements_multipatch = f"Basements_Multipatch_{out_suffix}"

    result_fc = generate_lod2_parallel(
        lasd_file=lasd_file,
        bldg_clipped=bldg_clipped,
        final_lod2_out=lod2_out,
        out_suffix=out_suffix
    )

    if not result_fc or not arcpy.Exists(lod2_out):
        print(f" -> ERROR during Parallel LOD2 Extraction.")
        return None, None

    print("Step 6: Prepping Basement Data...")

    arcpy.ddd.AddZInformation(
        in_feature_class=lod2_out,
        out_property="Z_MIN"
    )

    footprints_out = f"LOD2_Footprints_{out_suffix}"
    footprints_3d_out = f"LOD2_Footprints_3D_{out_suffix}"
    temp_layer = f"LOD2_Footprints_Layer_{out_suffix}"

    arcpy.ddd.MultiPatchFootprint(
        in_feature_class=lod2_out,
        out_feature_class=footprints_out
    )

    # Join Attributes (so footprint gets the Z_Min from the multipatch)
    arcpy.management.JoinField(
        in_data=footprints_out,
        in_field="OBJECTID",
        join_table=lod2_out,
        join_field="OBJECTID",
        fields=["Z_Min"]
    )

    arcpy.ddd.FeatureTo3DByAttribute(
        in_features=footprints_out,
        out_feature_class=footprints_3d_out,
        height_field="Z_Min"
    )

    arcpy.management.MakeFeatureLayer(
        in_features=footprints_3d_out,
        out_layer=temp_layer
    )

    arcpy.ddd.Layer3DToFeatureClass(
        in_feature_layer=temp_layer,
        out_feature_class=basements_multipatch,
        group_field=None,
        disable_materials="ENABLE_COLORS_AND_TEXTURES"
    )

    print(f" -> 3D Footprints ready at {basements_multipatch}.")

    return lod2_out, basements_multipatch


def find_problematic_buildings(lod2_fc, roof_dsm, bare_earth_dem, bldg_footprints, cell_size=0.5):
    print("Step 5.5: Evaluating LOD2 Volume Accuracy...")

    worst_percent = config.WORST_PERCENT

    lod2_raster = "LOD2_Surface_Raster"
    arcpy.conversion.MultipatchToRaster(
        in_multipatch_features=lod2_fc,
        out_raster=lod2_raster,
        cell_size=cell_size,
        cell_assignment_method="MAXIMUM_HEIGHT"
    )

    from arcpy.sa import Abs, Minus, ZonalStatisticsAsTable, Raster
    print(" -> Calculating height deviations...")
    error_raster = Abs(Raster(lod2_raster) - Raster(roof_dsm))
    actual_height_raster = Minus(Raster(roof_dsm), Raster(bare_earth_dem))

    print(" -> Aggregating volumetric data per building...")
    error_table = "Zonal_Error_Sum"
    actual_table = "Zonal_Actual_Sum"
    uid_field = "BUILDING_ID"

    ZonalStatisticsAsTable(bldg_footprints, uid_field, error_raster, error_table, "DATA", "SUM")
    ZonalStatisticsAsTable(bldg_footprints, uid_field, actual_height_raster, actual_table, "DATA", "SUM")

    arcpy.management.JoinField(bldg_footprints, uid_field, error_table, uid_field, ["SUM"])
    arcpy.management.AlterField(bldg_footprints, "SUM", "ERROR_SUM", "ERROR_SUM")

    arcpy.management.JoinField(bldg_footprints, uid_field, actual_table, uid_field, ["SUM"])
    arcpy.management.AlterField(bldg_footprints, "SUM", "ACTUAL_SUM", "ACTUAL_SUM")

    print(" -> Calculating final error thresholds...")
    code_block = """def calc_pct(err_sum, act_sum):
        try:
            if act_sum and act_sum > 0:
                return (float(err_sum) / float(act_sum)) * 100
            return 0
        except:
            return 0"""

    arcpy.management.CalculateField(
        in_table=bldg_footprints,
        field="PCT_VOLUME_ERROR",
        expression="calc_pct(!ERROR_SUM!, !ACTUAL_SUM!)",
        expression_type="PYTHON3",
        code_block=code_block,
        field_type="DOUBLE"
    )

    bad_footprints = "ATX_Buildings_Requires_Shatter"

    total_buildings = int(arcpy.management.GetCount(bldg_footprints)[0])
    n_worst = int(total_buildings * (worst_percent / 100.0))

    # This forces it to grab at least 1 building if you asked for >0%.
    if n_worst == 0 and worst_percent > 0 and total_buildings > 0:
        n_worst = 1

    print(f" -> Targeting the worst {worst_percent}% of {total_buildings} buildings ({n_worst} buildings)...")

    worst_ids = []

    sql_postfix = "ORDER BY PCT_VOLUME_ERROR DESC"
    with arcpy.da.SearchCursor(bldg_footprints, ["BUILDING_ID"], sql_clause=(None, sql_postfix)) as cursor:
        for i, row in enumerate(cursor):
            if i >= n_worst:
                break
            worst_ids.append(str(row[0]))

    if worst_ids:
        id_list = ",".join(worst_ids)
        where_clause = f"BUILDING_ID IN ({id_list})"
        arcpy.analysis.Select(bldg_footprints, bad_footprints, where_clause)
    else:
        arcpy.management.CopyFeatures(bldg_footprints, bad_footprints)
        arcpy.management.DeleteRows(bad_footprints)

    count = int(arcpy.management.GetCount(bad_footprints)[0])
    print(f" -> Triage complete: Found {count} buildings requiring the Shatter pipeline.")

    return bad_footprints, count


# ==========================================
# STEP 6: LOD2 SECOND PASS (TRIAGE & SHATTER)
# ==========================================
def lod2_second_pass(primary_lod2, primary_basements, bldg_clip, lasd_dataset, bare_earth_dem):
    print("\nStarting Second Pass for Complex Buildings...")

    bad_footprints = "ATX_Buildings_Requires_Shatter"
    force_shatter_ids = config.FORCE_SHATTER_IDs
    skip_triage = config.SKIP_TRIAGE

    # PATH A: Skip triage and only use forced IDs
    if skip_triage and force_shatter_ids:
        print(" -> Skipping volumetric triage. Using explicit building IDs only...")
        id_list = ",".join([str(i) for i in force_shatter_ids])
        where_clause = f"BUILDING_ID IN ({id_list})"

        arcpy.analysis.Select(bldg_clip, bad_footprints, where_clause)
        bad_count = int(arcpy.management.GetCount(bad_footprints)[0])

    # PATH B: Standard triage (+ append forced IDs)
    else:
        print(" -> Generating Campus Roof DSM for volumetric triage...")
        roof_dsm = "Campus_Roof_DSM_Triage"
        filtered_las_triage = "LAS_Triage_Layer"

        arcpy.management.MakeLasDatasetLayer(lasd_dataset, filtered_las_triage, class_code=["1", "2", "6", "20"])
        desc = arcpy.Describe(bldg_clip)

        with arcpy.EnvManager(extent=desc.extent, outputCoordinateSystem=desc.spatialReference):
            arcpy.conversion.LasDatasetToRaster(
                in_las_dataset=filtered_las_triage,
                out_raster=roof_dsm,
                value_field="ELEVATION",
                interpolation_type="BINNING MAXIMUM NATURAL_NEIGHBOR",
                data_type="FLOAT",
                sampling_type="CELLSIZE",
                sampling_value=1,
                z_factor=1
            )

        # Run normal triage
        bad_footprints, bad_count = find_problematic_buildings(
            lod2_fc=primary_lod2,
            roof_dsm=roof_dsm,
            bare_earth_dem=bare_earth_dem,
            bldg_footprints=bldg_clip
        )

        # If user passed extra IDs, select them and append them to the triage list
        if force_shatter_ids:
            print(f" -> Forcing explicit building IDs into the shatter pipeline: {force_shatter_ids}")
            forced_fc = "Forced_Shatter_Footprints"
            id_list = ",".join([str(i) for i in force_shatter_ids])
            where_clause = f"BUILDING_ID IN ({id_list})"

            arcpy.analysis.Select(bldg_clip, forced_fc, where_clause)
            arcpy.management.Append(inputs=forced_fc, target=bad_footprints, schema_type="NO_TEST")

            # Delete Identical ensures we don't process a building twice if triage already caught it
            arcpy.management.DeleteIdentical(in_dataset=bad_footprints, fields=["BUILDING_ID"])
            bad_count = int(arcpy.management.GetCount(bad_footprints)[0])

    # Shatter pipeline
    if bad_count == 0:
        print(" -> All buildings passed volume check. Skipping second pass.")
        return primary_lod2, primary_basements

    shattered_footprints = shatter_by_height_tiers(lasd_dataset, bad_footprints)

    shattered_lod2, shattered_basements = extract_lod2_and_basements(
        lasd_file=lasd_dataset,
        bldg_clipped=shattered_footprints,
        out_suffix="Shattered"
    )

    print(" -> Purging corrupted flat shards (Z = 0) from the shattered dataset...")
    with arcpy.da.UpdateCursor(shattered_lod2, ["Z_MIN"]) as cursor:
        for row in cursor:
            if row[0] is None or row[0] <= 1.0:
                cursor.deleteRow()

    print(" -> Removing bad generations from primary layers...")
    bad_ids = [row[0] for row in arcpy.da.SearchCursor(bad_footprints, ["BUILDING_ID"])]

    with arcpy.da.UpdateCursor(primary_lod2, ["BUILDING_ID"]) as cursor:
        for row in cursor:
            if row[0] in bad_ids:
                cursor.deleteRow()

    with arcpy.da.UpdateCursor(primary_basements, ["BUILDING_ID"]) as cursor:
        for row in cursor:
            if row[0] in bad_ids:
                cursor.deleteRow()

    print(" -> Merging raw corrected geometry into primary layers...")
    arcpy.management.Append(inputs=shattered_lod2, target=primary_lod2, schema_type="NO_TEST")
    arcpy.management.Append(inputs=shattered_basements, target=primary_basements, schema_type="NO_TEST")

    print(" -> Second Pass Complete!")

    return primary_lod2, primary_basements


def run_blender_postprocessing():
    print("Handing off to Blender for centering and PLY export...")

    command = [config.blender_exe, "--background", "--python", config.blender_script]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Blender processing complete!")
        print("\n--- BLENDER LOGS ---")
        print(result.stdout)
    else:
        print("Blender hit an error:")
        print(result.stderr)


def run_standalone_exporter():
    print("\nHanding off to standalone Python process for geometry export...")
    exporter_script = os.path.join(config.BASE_DIR, "arcpy_exporter.py")

    # Use sys.executable to ensure we use ArcGIS Pro's Python interpreter
    command = [sys.executable, exporter_script]

    # Force the main process to drop all Geodatabase locks before spawning the child
    arcpy.ClearWorkspaceCache_management()

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode == 0:
        print("Standalone export completed successfully.")
        return True
    else:
        print("Standalone export failed! See errors below:")
        print(result.stderr)
        return False


def anchor_facades():
    multiprocessing.freeze_support()

    setup_environment()
    aoi_fc = create_aoi_polygon()
    download_austin_planimetrics(aoi_fc)

    print("\n--- Initiating Facade Mesh Anchoring ---")
    export_success = True

    try:
        payloads = get_all_telemetry_payloads(config.SOURCE_DIR)

        if not payloads:
            print("No payloads found to process. Exiting.")
            return

        for identifier, payload in payloads:
            print(f"\n -> Processing Facade: {identifier}")
            anchor_data = anchor_single_facade(
                payload=payload,
                footprints_fc="ATX_Buildings_2023_Clipped",
                lod2_fc="LOD2_Export_Merge_Clean",
                dem_raster="DEM_Triangulated_028_Raster_Clipped"
            )

            export_json_path = os.path.join(config.EXPORTS_DIR, f"facade_anchoring_{identifier}.json")
            with open(export_json_path, 'w') as f:
                json.dump(anchor_data, f, indent=4)

            print(f" -> Saved anchoring instructions to facade_anchoring_{identifier}.json")

    except Exception as e:
        print(f"CRITICAL ERROR in Anchoring Pipeline: {e}")
        export_success = False

    if export_success:
        print("\n--- HANDING OFF TO BLENDER ---")

        command = [
            config.blender_exe,
            "--background",
            "--python-exit-code", "1",
            "--python", config.blender_script
        ]

        print(" -> Blender is currently processing the geometry. Please wait...")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print("\nBlender processing complete!")
            print("\n--- BLENDER LOGS ---")
            print(result.stdout)
        else:
            print("\nBlender hit an error:")
            print(result.stderr)
    else:
        print("Pipeline halted due to anchoring failure. Facade anchoring Blender step skipped.")


def main():
    setup_environment()

    aoi_fc = create_aoi_polygon()

    bldg_clip, cover_clip = download_austin_planimetrics(aoi_fc)

    lasd_dataset = create_las_dataset(aoi_fc, skip=False)

    bare_earth_dem = create_custom_dem(aoi_fc, bldg_clip, cover_clip, lasd_dataset, skip=False)

    primary_lod2, primary_basements = extract_lod2_and_basements(lasd_dataset, bldg_clip, out_suffix="primary")

    final_lod2, final_basements = lod2_second_pass(
        primary_lod2=primary_lod2,
        primary_basements=primary_basements,
        bldg_clip=bldg_clip,
        lasd_dataset=lasd_dataset,
        bare_earth_dem=bare_earth_dem
    )

    print("Prepping unified building schemas for export...")
    arcpy.management.Merge(inputs=final_lod2, output="LOD2_Export_Merge_Clean", field_match_mode="AUTOMATIC")
    arcpy.management.Merge(inputs=final_basements, output="Basements_Export_Merge_Clean", field_match_mode="AUTOMATIC")

    print("ArcPy Generation Pipeline Complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    main()

    time.sleep(5)

    run_standalone_exporter()
    run_blender_postprocessing()
    anchor_facades()
