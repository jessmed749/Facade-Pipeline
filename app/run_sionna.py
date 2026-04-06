import mitsuba as mi
mi.set_variant('llvm_ad_mono_polarized')

import os
import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
SCENE_XML   = os.path.join(_OUTPUT_DIR, "sionna_scene.xml")

# ── Geometry notes ────────────────────────────────────────────────────────────
#
# OLD geometry :
#   TX [10, -20, 5]  →  RX [0, 20, 2]
#   TX is in front of the facade, RX is BEHIND it.
#   The direct TX→RX ray passes clean through the scene with no obstruction,
#   so path 0 was always a near-LoS path dominating everything at -75.7 dBW.
#   The facade only added +0.2 dB because reflections were irrelevant vs direct.
#
# NEW geometry — both TX and RX on the FRONT side (Y negative), far apart in X:
#   TX [-20, -5, 5]     RX [20, -5, 2]
#   The direct TX→RX ray now has to travel horizontally across the facade face.
#   With USE_BLOCKER=True a concrete slab at X=0 completely cuts the direct path,
#   so ALL received power must come from facade reflections.
#
#         facade (Y ≈ 0, extends in X and Z)
#         ┌──────────────────────────────┐
#         │                              │
#         └──────────────────────────────┘
#              blocker wall (X=0)
#                    │
#   TX (-20,-5,5)    │         RX (20,-5,2)
#         ←──────────┼──────────→  (direct path blocked)
#          ↘         │         ↗
#            reflects off facade surface
#
# 

USE_BLOCKER = True   # Set False to remove the wall and allow soft NLoS

BLOCKER_OBJ = os.path.join(_OUTPUT_DIR, "blocker_wall.obj")

TX_POS = [-20.0, -5.0,  5.0]   # far left,  5 m in front of facade
RX_POS = [ 20.0, -5.0,  2.0]   # far right, 5 m in front of facade


def write_blocker_obj(path):
    """
    Thin vertical concrete slab at X=0, Y=-1 to -20, Z=0 to 15.
    Completely blocks the direct TX→RX path.
    """
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
    with open(path, "w") as f:
        f.write("# Blocker wall — LoS obstruction\no blocker_wall\n")
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")
    print(f"  Blocker OBJ written → {path}")


def patch_scene_xml_with_blocker(xml_path, blocker_obj_path):
    """Insert blocker wall into the Mitsuba XML scene if not already present."""
    with open(xml_path, "r") as f:
        xml = f.read()

    if "mesh-blocker_wall" in xml:
        print(f"  Blocker already in XML — skipping injection")
        return

    snippet = f"""
    <!-- Blocker wall — forces NLoS -->
    <shape type="obj" id="mesh-blocker_wall">
        <string name="filename" value="{os.path.abspath(blocker_obj_path)}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="concrete" name="bsdf"/>
    </shape>
"""
    xml = xml.replace("</scene>", snippet + "\n</scene>")
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"  Blocker wall injected into scene XML")


def setup_scene():
    scene = load_scene(SCENE_XML)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                  horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                  horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.frequency = 3.5e9
    return scene


solver = PathSolver()


def compute_paths(scene, max_depth):
    return solver(scene, max_depth=max_depth)


def path_power(paths):
    a_real, a_imag = paths.a
    return float(np.sum(np.array(a_real)**2 + np.array(a_imag)**2))


def fmt_power(p):
    if p <= 0:
        return "0.0  (no paths)"
    return f"{p:.4e}  ({10*np.log10(p):+.1f} dBW)"


# ── Scene preparation ─────────────────────────────────────────────────────────
print("\nPreparing scene …")
if USE_BLOCKER:
    write_blocker_obj(BLOCKER_OBJ)
    patch_scene_xml_with_blocker(SCENE_XML, BLOCKER_OBJ)
    print(f"  Mode : TRUE NLoS — blocker wall at X=0 cuts direct path")
else:
    print(f"  Mode : Soft NLoS — TX/RX separated by facade geometry only")

print(f"  TX   : {TX_POS}")
print(f"  RX   : {RX_POS}")

# ── Experiment 1 — NLoS path validation ──────────────────────────────────────
print("\n=== Experiment 1: NLoS Path Validation ===")
scene = setup_scene()
tx = Transmitter("tx", position=TX_POS)
rx = Receiver("rx",    position=RX_POS)
scene.add(tx); scene.add(rx)
paths = compute_paths(scene, max_depth=4)
a_real, a_imag = paths.a
p_total = path_power(paths)
print(f"  Paths found            : {a_real.shape[-1]}")
print(f"  Path coefficients shape: {a_real.shape}")
print(f"  Total power            : {fmt_power(p_total)}")

coeffs = np.array(a_real)[0,0,0,0] + 1j * np.array(a_imag)[0,0,0,0]
print(f"  Per-path |a|² breakdown:")
for i, c in enumerate(coeffs):
    pp = abs(c)**2
    dbw = 10 * np.log10(pp) if pp > 0 else float('nan')
    print(f"    path {i:2d}: {pp:.4e}  ({dbw:+.1f} dBW)")

# ── Experiment 2 — TX height sweep ───────────────────────────────────────────
print("\n=== Experiment 2: TX Height Sweep ===")
print(f"  {'Height':>8}  {'Paths':>6}  {'Power (lin)':>14}  {'Power (dBW)':>12}")
print(f"  {'─'*8}  {'─'*6}  {'─'*14}  {'─'*12}")
for height in [1.5, 5.0, 10.0, 20.0, 35.0]:
    scene = setup_scene()
    tx = Transmitter("tx", position=[TX_POS[0], TX_POS[1], height])
    rx = Receiver("rx",    position=RX_POS)
    scene.add(tx); scene.add(rx)
    paths = compute_paths(scene, max_depth=4)
    a_real, _ = paths.a
    p = path_power(paths)
    dbw = 10 * np.log10(p) if p > 0 else float('nan')
    print(f"  {height:>7.1f}m  {a_real.shape[-1]:>6}  {p:>14.4e}  {dbw:>+11.1f} dBW")

# ── Experiment 3 — Reflections on vs off ─────────────────────────────────────
print("\n=== Experiment 3: Reflections On vs Off ===")
scene = setup_scene()
tx = Transmitter("tx", position=TX_POS)
rx = Receiver("rx",    position=RX_POS)
scene.add(tx); scene.add(rx)

paths_on  = compute_paths(scene, max_depth=4)
paths_off = compute_paths(scene, max_depth=0)
p_on  = path_power(paths_on)
p_off = path_power(paths_off)
ar_on,  _ = paths_on.a
ar_off, _ = paths_off.a

print(f"  Reflections ON  — paths={ar_on.shape[-1]:3d}  power={fmt_power(p_on)}")
print(f"  Reflections OFF — paths={ar_off.shape[-1]:3d}  power={fmt_power(p_off)}")

if USE_BLOCKER:
    if p_off == 0:
        print(f"   Blocker confirmed: depth=0 finds no LoS path")
        print(f"   All {ar_on.shape[-1]} received paths are pure facade reflections")
    else:
        print(f"   LoS still leaking — blocker geometry may need to be wider/taller")
else:
    if p_on > 0 and p_off > 0:
        print(f"  Facade multipath gain: {10*np.log10(p_on/p_off):+.1f} dB over LoS-only")

# ── Experiment 4 — Reflection depth 0 → 5 ────────────────────────────────────
print("\n=== Experiment 4: Reflection Depth Analysis ===")
print(f"  {'Depth':>6}  {'Paths':>6}  {'Cumul. power':>16}  {'dBW':>10}  {'ΔdB vs prev':>12}")
print(f"  {'─'*6}  {'─'*6}  {'─'*16}  {'─'*10}  {'─'*12}")
scene = setup_scene()
tx = Transmitter("tx", position=TX_POS)
rx = Receiver("rx",    position=RX_POS)
scene.add(tx); scene.add(rx)

prev_dbw = None
for depth in range(6):
    paths = compute_paths(scene, max_depth=depth)
    a_real, _ = paths.a
    p = path_power(paths)
    dbw = 10 * np.log10(p) if p > 0 else float('nan')
    if prev_dbw is not None and not np.isnan(dbw) and not np.isnan(prev_dbw):
        delta = f"{dbw - prev_dbw:+.2f} dB"
    else:
        delta = "  —"
    print(f"  {depth:>6}  {a_real.shape[-1]:>6}  {p:>16.4e}  {dbw:>+9.1f}  {delta:>12}")
    prev_dbw = dbw

if USE_BLOCKER:
    print(f"\n  With the blocker in place you should see:")
    print(f"    depth=0 → 0 paths, 0 power  (no LoS)")
    print(f"    depth=1 → first facade reflections appear")
    print(f"    depth=2+ → additional bounces, diminishing returns")
    print(f"  If depth=0 still shows power, widen/raise the blocker wall.")

print("\n=== All experiments complete ===")