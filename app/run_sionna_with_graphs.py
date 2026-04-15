import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mitsuba as mi

mi.set_variant('llvm_ad_mono_polarized')

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SCENE_XML = os.path.join(OUTPUT_DIR, "sionna_scene.xml")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

USE_BLOCKER = True
BLOCKER_OBJ = os.path.join(OUTPUT_DIR, "blocker_wall.obj")

TX_POS = [-20.0, -5.0, 5.0]
RX_POS = [20.0, -5.0, 2.0]

TX_HEIGHTS = [1.5, 5.0, 10.0, 20.0, 35.0]
DEPTHS = [0, 1, 2, 3, 4, 5]

solver = PathSolver()

def write_blocker_obj(path):
    verts = [
        (-0.1, -1,   0),
        ( 0.1, -1,   0),
        ( 0.1, -20,  0),
        (-0.1, -20,  0),
        (-0.1, -1,  15),
        ( 0.1, -1,  15),
        ( 0.1, -20, 15),
        (-0.1, -20, 15),
    ]
    faces = [
        (1,2,3,4),
        (5,6,7,8),
        (1,2,6,5),
        (3,4,8,7),
        (1,4,8,5),
        (2,3,7,6),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# blocker wall\no blocker_wall\n")
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")

def patch_scene_xml_with_blocker(xml_path, blocker_obj_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    if "blocker_wall" in xml:
        return

    snippet = f"""
    <shape type="obj" id="blocker_wall">
        <string name="filename" value="{os.path.abspath(blocker_obj_path)}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="concrete" name="bsdf"/>
    </shape>
"""
    xml = xml.replace("</scene>", snippet + "\n</scene>")

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)

def setup_scene():
    scene = load_scene(SCENE_XML)
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )
    scene.frequency = 3.5e9
    return scene

def compute_paths(scene, max_depth):
    return solver(scene, max_depth=max_depth)

def path_power(paths):
    a_real, a_imag = paths.a
    return float(np.sum(np.array(a_real)**2 + np.array(a_imag)**2))

def path_count(paths):
    a_real, _ = paths.a
    return int(a_real.shape[-1])
    

def save_line(df, x, y, title, fname):
    plt.figure(figsize=(7,5))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, fname), dpi=200)
    plt.close()

def save_bar(df, x, y, title, fname):
    plt.figure(figsize=(7,5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, fname), dpi=200)
    plt.close()

def main():
    if not os.path.exists(SCENE_XML):
        raise FileNotFoundError(f"Missing scene XML: {SCENE_XML}")

    if USE_BLOCKER:
        write_blocker_obj(BLOCKER_OBJ)
        patch_scene_xml_with_blocker(SCENE_XML, BLOCKER_OBJ)

    # Experiment 1
    scene = setup_scene()
    scene.add(Transmitter("tx", position=TX_POS))
    scene.add(Receiver("rx", position=RX_POS))
    p1 = compute_paths(scene, max_depth=4)

    exp1 = pd.DataFrame([{
        "scenario": "PMA_blocked_reflection",
        "num_paths": path_count(p1),
        "power": path_power(p1)
    }])
    exp1.to_csv(os.path.join(GRAPH_DIR, "exp1.csv"), index=False)

    # Experiment 2
    rows = []
    for h in TX_HEIGHTS:
        scene = setup_scene()
        scene.add(Transmitter("tx", position=[TX_POS[0], TX_POS[1], h]))
        scene.add(Receiver("rx", position=RX_POS))
        p = compute_paths(scene, max_depth=4)
        rows.append({
            "height": h,
            "num_paths": path_count(p),
            "power": path_power(p)
        })
    exp2 = pd.DataFrame(rows)
    exp2.to_csv(os.path.join(GRAPH_DIR, "exp2.csv"), index=False)

    # Experiment 3
    scene = setup_scene()
    scene.add(Transmitter("tx", position=TX_POS))
    scene.add(Receiver("rx", position=RX_POS))
    p_on = compute_paths(scene, max_depth=4)
    p_off = compute_paths(scene, max_depth=0)

    exp3 = pd.DataFrame([
        {"mode": "ON", "num_paths": path_count(p_on), "power": path_power(p_on)},
        {"mode": "OFF", "num_paths": path_count(p_off), "power": path_power(p_off)}
    ])
    exp3.to_csv(os.path.join(GRAPH_DIR, "exp3.csv"), index=False)

    # Experiment 4
    rows = []
    for d in DEPTHS:
        scene = setup_scene()
        scene.add(Transmitter("tx", position=TX_POS))
        scene.add(Receiver("rx", position=RX_POS))
        p = compute_paths(scene, max_depth=d)
        rows.append({
            "depth": d,
            "num_paths": path_count(p),
            "power": path_power(p)
        })
    exp4 = pd.DataFrame(rows)
    exp4.to_csv(os.path.join(GRAPH_DIR, "exp4.csv"), index=False)

    # Plots
    save_bar(exp1, "scenario", "num_paths", "Experiment 1: PMA NLoS Path Count", "exp1_paths.png")
    save_bar(exp1, "scenario", "power", "Experiment 1: PMA NLoS Power", "exp1_power.png")

    save_line(exp2, "height", "num_paths", "Experiment 2: TX Height vs Path Count", "exp2_height_paths.png")
    save_line(exp2, "height", "power", "Experiment 2: TX Height vs Power", "exp2_height_power.png")

    save_bar(exp3, "mode", "num_paths", "Experiment 3: Reflections OFF vs ON", "exp3_reflection_paths.png")
    save_bar(exp3, "mode", "power", "Experiment 3: Reflections OFF vs ON Power", "exp3_reflection_power.png")

    save_line(exp4, "depth", "num_paths", "Experiment 4: Max Depth vs Path Count", "exp4_depth_paths.png")
    save_line(exp4, "depth", "power", "Experiment 4: Max Depth vs Power", "exp4_depth_power.png")

    summary = pd.concat([
        exp1.assign(experiment="exp1_nlos"),
        exp2.assign(experiment="exp2_height_sweep"),
        exp3.assign(experiment="exp3_reflections_toggle"),
        exp4.assign(experiment="exp4_depth_analysis"),
    ], ignore_index=True, sort=False)

    summary.to_csv(os.path.join(GRAPH_DIR, "summary_all_experiments.csv"), index=False)
    print("Saved graphs and CSVs to:", GRAPH_DIR)

if __name__ == "__main__":
    main()
