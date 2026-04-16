import os
import time
import shutil
import arcpy
import config


WORKSPACE_GDB = config.WORKSPACE_GDB
EXPORTS_DIR = config.EXPORTS_DIR


def setup_export_folders():
    print("[Exporter] Cleaning export directories...")
    folders = ['buildings', 'basements', 'terrain_grass', 'terrain_impervious']
    for subfolder in folders:
        folder_path = os.path.join(EXPORTS_DIR, subfolder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"[Exporter] Warning: Could not delete {folder_path}. {e}")
        os.makedirs(folder_path, exist_ok=True)


def robust_export(fc_name, target_folder, max_retries=3):
    """Executes Export3DObjects with an active polling loop to guarantee file creation."""
    fc_path = os.path.join(WORKSPACE_GDB, fc_name)

    if not arcpy.Exists(fc_path):
        raise FileNotFoundError(f"CRITICAL: {fc_name} does not exist in the Geodatabase.")

    # Ensure 3D formats are applied (redundant, but safe)
    arcpy.management.Add3DFormats(fc_path, "MULTIPATCH_WITH_MATERIALS", "FMT3D_OBJ")

    for attempt in range(1, max_retries + 1):
        print(f"[Exporter] Exporting {fc_name} (Attempt {attempt}/{max_retries})...")
        try:
            arcpy.management.Export3DObjects(
                in_features=fc_path,
                target_folder=target_folder,
                formats="FMT3D_OBJ",
                overwrite="OVERWRITE",
                merge="MERGE"
            )
        except Exception as e:
            print(f"[Exporter] ArcPy Exception: {e}")

        # Active Polling Loop: Recursively check for the .obj file
        flush_successful = False
        print(f"[Exporter] Scanning directory for output files...")

        for poll in range(20):  # Poll for up to 20 seconds
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
            print(f"[Exporter] -> Success! {fc_name} is physically on disk.\n")
            return
        else:
            print(f"[Exporter] -> Warning: Files not found after attempt {attempt}.")
            time.sleep(3.0)

    raise RuntimeError(f"[Exporter] CRITICAL FAILURE: Could not export {fc_name}.")


def main():
    print("\n--- STARTING STANDALONE ARCPY EXPORT ---")
    arcpy.env.workspace = WORKSPACE_GDB
    arcpy.env.overwriteOutput = True

    setup_export_folders()

    try:
        # 1. Export Terrain
        robust_export("DEM_Grass_3D_Final", os.path.join(EXPORTS_DIR, 'terrain_grass'))
        robust_export("DEM_Impervious_3D_Final", os.path.join(EXPORTS_DIR, 'terrain_impervious'))

        # 2. Export Buildings & Basements
        robust_export("LOD2_Export_Merge_Clean", os.path.join(EXPORTS_DIR, 'buildings'))
        robust_export("Basements_Export_Merge_Clean", os.path.join(EXPORTS_DIR, 'basements'))

        print("--- ALL EXPORTS COMPLETED SUCCESSFULLY ---\n")

    except Exception as e:
        print(f"\n[Exporter] FATAL ERROR: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()