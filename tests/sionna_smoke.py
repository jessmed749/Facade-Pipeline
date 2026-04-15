"""
tests/sionna_smoke.py

Quick sanity check: load the output Sionna XML scene, run PathSolver at
max_depth=4, and assert that:
  1. At least 1 path is found.
  2. Total received power > 1e-12 (above numerical noise floor).

Fails with exit code 1 if either condition is not met.
"""

import os
import sys
import numpy as np

SCENE_XML = os.path.join("output", "sionna_scene.xml")


def path_power(paths):
    a_real, a_imag = paths.a
    return float(np.sum(np.array(a_real)**2 + np.array(a_imag)**2))


def main():
    if not os.path.exists(SCENE_XML):
        print(f"✗ Sionna scene XML not found: {SCENE_XML}")
        print("  Run the pipeline first: make run")
        sys.exit(1)

    try:
        import mitsuba as mi
        mi.set_variant("llvm_ad_mono_polarized")
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    except ImportError as e:
        print(f"⚠  Sionna not installed ({e}) — skipping smoke test.")
        sys.exit(0)

    print("Sionna smoke test …")
    print(f"  Loading scene: {SCENE_XML}")

    scene = load_scene(SCENE_XML)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                  horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                  horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.frequency = 3.5e9

    from sionna.rt import Transmitter, Receiver
    tx = Transmitter("tx", position=[-20.0, -5.0, 5.0])
    rx = Receiver("rx",    position=[ 20.0, -5.0, 2.0])
    scene.add(tx)
    scene.add(rx)

    solver = PathSolver()
    paths  = solver(scene, max_depth=4)
    a_real, _ = paths.a
    n_paths = a_real.shape[-1]
    power   = path_power(paths)

    print(f"  Paths found : {n_paths}")
    print(f"  Total power : {power:.4e}")

    ok = True
    if n_paths < 1:
        print("✗ FAIL: no paths found — mesh geometry may be invalid or empty.")
        ok = False
    else:
        print(f"  ✓ Path count: {n_paths} >= 1")

    if power <= 1e-12:
        print("✗ FAIL: total power at noise floor — materials or geometry broken.")
        ok = False
    else:
        print(f"  ✓ Power: {power:.4e} > 1e-12")

    if not ok:
        sys.exit(1)

    print("Sionna smoke test passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()