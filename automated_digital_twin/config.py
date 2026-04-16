import os

# Directory and File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WORKSPACE_GDB = os.path.join(BASE_DIR, "data.gdb")
EXPORTS_DIR = os.path.join(BASE_DIR, "Exports")
PLY_DIR = os.path.join(BASE_DIR, "PLY_Exports")

SOURCE_DIR = os.path.join(BASE_DIR, "source_data")
LAS_FOLDER = os.path.join(SOURCE_DIR, "stratmap21-28cm-50cm-bexar-travis_3097433_lpc")

blender_exe = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
blender_script = os.path.join(BASE_DIR, "blender_postprocess.py")


# Arcpy Spatial Reference
WKID_XY = 6577
WKID_Z = 5703


# LOD2 Buildings
LOD2_LEVEL_OF_DETAIL = "LOD2.0"     # One of ["LOD1.2", "LOD1.3", "LOD2.0"]
LOD2_SMOOTHNESS = 0                 # 0.0 to 1.0
LOD2_ACCURACY = "HIGH"              # One of ["LOW", "MEDIUM", "HIGH"]
PROCESSING_CORES = 8                # Default is 4, lower if CPU or memory is maxing out during LOD2 generation


# Second Pass Buildings
FORCE_SHATTER_IDs = None            # Specify specific building IDs to shatter for LOD2 second pass. List of integers (e.g., [3, 37]), otherwise None. Default is None.
SKIP_TRIAGE = False                 # Set to True if you only want to re-generate buildings in FORCE_SHATTER_IDs and nothing else. Default is False.
WORST_PERCENT = 25.0                # Top N% worst building volume error, used to select which buildings are re-generated. Default is 25.0. Setting it to 0.0 will disable the second pass.
HEIGHT_TIER_CUTOFF = 5.0            # Decrease if second pass is missing vertical details, increase if holes are appearing in the LOD2 second pass buildings or there are significant exterior wall artifacts. Default is 5.0.
HEIGHT_SAMPLING_AREA = 10           # Area to sample height for deciding where to shatter building footprints into height tiers. Default is 10.


# Digital Elevation Model (DEM)
DEM_SAMPLING_AREA = 0.28            # Suggested minimum is the density of your LAS point cloud. Higher values may decrease DEM processing time. Default is 0.28.


# Area of Interest Coordinates
# Points along an AOI must be strictly counter-clockwise or clockwise
# Coordinate System: NAD 1983 (2011) StatePlane Texas Central FIPS 4203 (Meters) - WKID: 6577

AOI_COORDINATE_VERTICES = [
    (949777.46, 3071933.97, 0.0),
    (949853.50, 3071929.12, 0.0),
    (949847.49, 3071844.53, 0.0),
    (949775.61, 3071847.07, 0.0)
]

# Example coordinates for the UT Austin Physics, Math, and Astronomy Building:
#
# AOI_COORDINATE_VERTICES = [
#     (949777.46, 3071933.97, 0.0),
#     (949853.50, 3071929.12, 0.0),
#     (949847.49, 3071844.53, 0.0),
#     (949775.61, 3071847.07, 0.0)
# ]
#
# The area of interest can generally be any n-sided polygon
# For example an area with a dozen buildings:
#
# AOI_COORDINATE_VERTICES = [
#     (949454.22, 3072383.29, 0.0),
#     (949845.76, 3072335.22, 0.0),
#     (949992.39, 3072224.17, 0.0),
#     (949939.71, 3071499.04, 0.0),
#     (949397.50, 3071529.57, 0.0)
# ]
#
# Or even all of UT Austin campus (however AOIs this large may fail to generate all at once):
#
# AOI_COORDINATE_VERTICES = [
#     (949264.6865, 3072557.328, 0.0),
#     (949544.2853, 3072428.325, 0.0),
#     (949605.3712, 3072536.843, 0.0),
#     (949689.9027, 3072497.041, 0.0),
#     (949748.5309, 3072604.938, 0.0),
#     (950065.7984, 3072316.631, 0.0),
#     (950242.1321, 3072227.176, 0.0),
#     (950223.2516, 3072190.495, 0.0),
#     (950595.8011, 3072003.004, 0.0),
#     (950574.9609, 3071963.073, 0.0),
#     (950733.0857, 3071891.756, 0.0),
#     (950760.8559, 3071929.895, 0.0),
#     (950969.6561, 3071821.566, 0.0),
#     (951063.1828, 3071629.077, 0.0),
#     (951343.2218, 3071480.453, 0.0),
#     (951314.429, 3071389.421, 0.0),
#     (951289.8595, 3071386.659, 0.0),
#     (951313.005, 3071275.76, 0.0),
#     (951032.919, 3071207.889, 0.0),
#     (951154.3959, 3070635.545, 0.0),
#     (950962.0959, 3070592.24, 0.0),
#     (950479.7039, 3070502.012, 0.0),
#     (950529.4292, 3070258.968, 0.0),
#     (950412.0329, 3070233.881, 0.0),
#     (950449.1401, 3070051.178, 0.0),
#     (950302.3575, 3070020.641, 0.0),
#     (950155.2399, 3070015.606, 0.0),
#     (949671.2528, 3070159.907, 0.0),
#     (949831.2224, 3070695.478, 0.0),
#     (949055.6722, 3070928.94, 0.0),
#     (949083.5201, 3071026.481, 0.0),
#     (949122.7895, 3071138.746, 0.0),
#     (949160.3303, 3071759.825, 0.0),
#     (949167.5909, 3071761.121, 0.0),
#     (949195.1412, 3072369.228, 0.0)
# ]
