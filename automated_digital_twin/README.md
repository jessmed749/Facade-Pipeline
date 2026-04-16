# Quick Start

1. Download and Install [ArcGIS Pro 3.6.1](https://pro.arcgis.com/en/pro-app/3.6/get-started/download-arcgis-pro.htm)
2. Download and Install [Blender 5.1](https://www.blender.org/download/releases/5-1/)
3. Open `config.py` and update the `blender_exe` variable to point to `blender.exe`
4. [Set up your Python environment](#python-environment-setup)
5. [Download LiDAR data for your area](#required-inputs-aerial-lidar-data)
6. Open `config.py` and edit `AOI_COORDINATE_VERTICES` to define the area of interest where you want to generate a digital twin

---

## Python Environment Setup

This project relies on `arcpy`. Because ArcGIS Pro's default Python environment is read-only, you must clone it before installing any additional third-party packages required for processing the campus digital twin data. ArcGIS Pro uses `conda` rather than standard virtual environments (`venv`).

### Instructions to Clone and Activate:

1. Open Windows search, search for Python Command Prompt. Make sure the terminal that opens has (`arcgispro-py3`) on the left

2. Run the following command to clone the default environment:

    `conda create --clone arcgispro-py3 --name campus_twin_env`

3. In your IDE, select the campus_twin_env conda environment (by default, you can find the conda environment at 
`C:\Users\<YourUsername>\AppData\Local\ESRI\conda\envs\campus_twin_env)`.

---

### Required Inputs: Aerial LiDAR Data
1. Go to the [Texas Geographic Data Hub](https://data.geographic.texas.gov/collection/?c=447db89a-58ee-4a1b-a61f-b918af2fb0bb).
2. Open the **Downloads** tab and click **Select download areas**.
3. Choose **Austin East | SW**.
4. Download the **Compressed Lidar Point Cloud**.
5. Extract the folder (e.g., `stratmap21-28cm-50cm-bexar-travis_3097433_lpc`) into your `source_data` directory.
6. Open `config.py` and update the `LAS_FOLDER` variable to match the extracted folder's name.


### Optional Inputs: Building Facades
To add 3D facades, place a matched pair of files (a JSON config and an OBJ mesh) into `./source_data/`.

**1. Config File** (`facade_<name>.json`)
Sets the map location and orientation for the mesh. 
* **Example:** `facade_PMA.json`
```json
{
  "device_telemetry": {
    "latitude": 30.288753,
    "longitude": -97.736348,
    "heading_degrees": 94.67
  },
  "mesh_metadata": {
    "file_name": "combined_scene.obj",
    "front_facing_axis": "+Y"
  }
}
```

**2. Geometry File** (`.obj`)
The 3D model itself. The filename must perfectly match the `file_name` string inside your JSON config.
* **Example:** `combined_scene.obj`

---

### Tuning `config.py` Parameters

| Problem                                           | Things to try                                                                                                                                                                                                                                                                                                                                       |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Building walls are too jagged                     | * Increase `LOD2_SMOOTHNESS` (try 0.1 first) <br/> * Try adding the building ID to `FORCE_SHATTER_IDs` (to find the building ID, open data.gdb in ArcGIS Pro) <br/> * Increase `HEIGHT_TIER_CUTOFF` by a few meters <br/> * Change `LOD2_LEVEL_OF_DETAIL` to LOD1.3 (warning, this may affect facade accuracy, and it will affect rooftop accuracy) |
| Out of memory crashes                             | * Decrease `PROCESSING_CORES` <br/> * If happening specifically during the DEM creation, try increasing `DEM_SAMPLING_AREA`                                                                                                                                                                                                                         |
| Custom DEM creation not finishing after 24+ hours | * Area of Interest may be too big, try shrinking it                                                                                                                                                                                                                                                                                                 |


---

### Troubleshooting

| Problem                           | Things to try                                                                                                                                                      |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Facades are on the wrong building | Check the coordinates in the facade_<name>.json file in the source_data folder. There can't be any buildings between the coordinates and the target wall/building. |
