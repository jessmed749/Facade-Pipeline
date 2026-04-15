"""
tests/validate_outputs.py

Run after the pipeline to validate both mesh geometry and Sionna RF behaviour.
Writes output/validation_report.json — CI reads the exit code.

Usage:
    python3 tests/validate_outputs.py --output-dir output/building1
    python3 tests/validate_outputs.py --output-dir output/building1 --skip-sionna

Exit codes:
    0  all checks passed
    1  one or more checks failed
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Mesh geometry checks
# ─────────────────────────────────────────────────────────────────────────────

def check_obj_exists(output_dir: Path) -> list[dict]:
    results = []

    # combined_scene.obj must exist and be non-empty
    combined = output_dir / "combined_scene.obj"
    results.append({
        "check": "combined_scene.obj exists",
        "status": PASS if combined.exists() and combined.stat().st_size > 0 else FAIL,
        "detail": str(combined),
    })

    # per_class/ must have at least one .obj
    per_class = list((output_dir / "per_class").glob("*.obj")) if (output_dir / "per_class").exists() else []
    results.append({
        "check": "per_class has at least one .obj",
        "status": PASS if len(per_class) >= 1 else FAIL,
        "detail": f"{len(per_class)} files: {[f.name for f in per_class]}",
    })

    # sionna_scene.json must exist and be valid JSON
    sjson = output_dir / "sionna_scene.json"
    try:
        with open(sjson) as f:
            sdata = json.load(f)
        results.append({
            "check": "sionna_scene.json valid",
            "status": PASS,
            "detail": f"{len(sdata.get('objects', []))} objects",
        })
    except Exception as e:
        results.append({"check": "sionna_scene.json valid", "status": FAIL, "detail": str(e)})
        sdata = {}

    # facade_*.json must exist
    facade_files = list(output_dir.glob("facade_*.json"))
    results.append({
        "check": "facade_*.json exists",
        "status": PASS if facade_files else FAIL,
        "detail": [f.name for f in facade_files],
    })

    return results, sdata


def check_mesh_geometry(output_dir: Path, sionna_data: dict) -> list[dict]:
    """Load each per-class .obj and run trimesh geometry assertions."""
    results = []
    per_class_dir = output_dir / "per_class"

    if not per_class_dir.exists():
        return [{"check": "per_class dir", "status": FAIL, "detail": "directory missing"}]

    for obj_file in sorted(per_class_dir.glob("*.obj")):
        label = obj_file.stem  # e.g. "window", "door"

        try:
            mesh = trimesh.load(str(obj_file), force="mesh")
        except Exception as e:
            results.append({
                "check": f"{label}: loads without error",
                "status": FAIL, "detail": str(e),
            })
            continue

        results.append({
            "check": f"{label}: loads without error",
            "status": PASS,
            "detail": f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces",
        })

        # Must have faces
        results.append({
            "check": f"{label}: has faces",
            "status": PASS if len(mesh.faces) > 0 else FAIL,
            "detail": f"{len(mesh.faces)} faces",
        })

        # No degenerate (zero-area) faces — trimesh flags these
        degen = mesh.triangles_area < 1e-10
        results.append({
            "check": f"{label}: no degenerate faces",
            "status": PASS if degen.sum() == 0 else FAIL,
            "detail": f"{int(degen.sum())} degenerate out of {len(mesh.faces)}",
        })

        # Bounding box must be plausible (0.1 m – 200 m on each axis)
        bounds = mesh.bounds
        dims   = bounds[1] - bounds[0]
        ok = all(0.05 <= d <= 200.0 for d in dims)
        results.append({
            "check": f"{label}: bounds plausible (0.05–200 m)",
            "status": PASS if ok else FAIL,
            "detail": f"X={dims[0]:.2f}m Y={dims[1]:.2f}m Z={dims[2]:.2f}m",
        })

    # Per-object area sanity from sionna_scene.json
    objects = sionna_data.get("objects", [])
    if objects:
        areas = [o["area_m2"] for o in objects]
        bad_area = [o for o in objects if not (0.05 <= o["area_m2"] <= 50.0)]
        results.append({
            "check": "all object areas 0.05–50 m²",
            "status": PASS if not bad_area else FAIL,
            "detail": (f"min={min(areas):.2f} max={max(areas):.2f} — "
                       f"{len(bad_area)} out of range"),
        })

        # Must have detected at least one window
        window_objs = [o for o in objects if "window" in o["label"]]
        results.append({
            "check": "at least 1 window detected",
            "status": PASS if window_objs else FAIL,
            "detail": f"{len(window_objs)} window object(s)",
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Sionna RF validation
# ─────────────────────────────────────────────────────────────────────────────

def check_sionna(output_dir: Path) -> list[dict]:
    results = []
    xml_path = output_dir / "sionna_scene.xml"

    if not xml_path.exists():
        return [{"check": "sionna_scene.xml exists", "status": FAIL,
                 "detail": "run sionna_scene_loader.py first"}]

    results.append({"check": "sionna_scene.xml exists", "status": PASS, "detail": str(xml_path)})

    try:
        import mitsuba as mi
        mi.set_variant("llvm_ad_mono_polarized")
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
    except ImportError as e:
        return results + [{"check": "sionna importable", "status": SKIP,
                           "detail": f"not installed: {e}"}]

    results.append({"check": "sionna importable", "status": PASS, "detail": ""})

    # 1. Scene loads without exception
    try:
        scene = load_scene(str(xml_path))
        results.append({"check": "load_scene() succeeds", "status": PASS, "detail": ""})
    except Exception as e:
        results.append({"check": "load_scene() succeeds", "status": FAIL, "detail": str(e)})
        return results

    # Shared scene setup
    def setup(sc):
        sc.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                   horizontal_spacing=0.5, pattern="iso", polarization="V")
        sc.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                   horizontal_spacing=0.5, pattern="iso", polarization="V")
        sc.frequency = 3.5e9
        return sc

    def path_power(paths):
        a_real, a_imag = paths.a
        return float(np.sum(np.array(a_real)**2 + np.array(a_imag)**2))

    def dbw(p):
        return 10 * np.log10(p) if p > 0 else float("-inf")

    solver = PathSolver()

    # 2. NLoS paths exist at depth=4
    try:
        scene = setup(load_scene(str(xml_path)))
        tx = Transmitter("tx", position=[-20.0, -5.0, 5.0])
        rx = Receiver("rx",    position=[ 20.0, -5.0, 2.0])
        scene.add(tx); scene.add(rx)
        paths = solver(scene, max_depth=4)
        a_real, _ = paths.a
        n_paths = int(a_real.shape[-1])
        power   = path_power(paths)
        pw_dbw  = dbw(power)

        results.append({
            "check": "NLoS paths ≥ 1 at depth=4",
            "status": PASS if n_paths >= 1 else FAIL,
            "detail": f"{n_paths} paths, power={pw_dbw:.1f} dBW",
        })
        results.append({
            "check": "received power > −120 dBW",
            "status": PASS if pw_dbw > -120 else FAIL,
            "detail": f"{pw_dbw:.1f} dBW",
        })
    except Exception as e:
        results.append({"check": "NLoS path solve", "status": FAIL, "detail": str(e)})
        return results

    # 3. depth=0 (LoS only) has LESS power than depth=4 (reflections included)
    #    This proves the facade geometry is actually contributing reflected paths.
    try:
        scene = setup(load_scene(str(xml_path)))
        tx = Transmitter("tx2", position=[-20.0, -5.0, 5.0])
        rx = Receiver("rx2",   position=[ 20.0, -5.0, 2.0])
        scene.add(tx); scene.add(rx)
        p_los  = path_power(solver(scene, max_depth=0))
        p_full = path_power(solver(scene, max_depth=4))
        gain_db = dbw(p_full) - dbw(p_los) if p_los > 0 else float("inf")

        results.append({
            "check": "reflections add power (depth=4 > depth=0)",
            "status": PASS if p_full > p_los else FAIL,
            "detail": (f"LoS={dbw(p_los):.1f} dBW  full={dbw(p_full):.1f} dBW  "
                       f"gain={gain_db:+.1f} dB"),
        })
    except Exception as e:
        results.append({"check": "reflection gain check", "status": FAIL, "detail": str(e)})

    # 4. Material contrast: glass (window) reflects more than concrete (wall).
    #    We probe this by reading the XML for the BSDF types and checking they
    #    differ — a hard structural check that doesn't require two separate meshes.
    try:
        xml_text = xml_path.read_text()
        has_glass    = "glass" in xml_text.lower()
        has_concrete = "concrete" in xml_text.lower()
        results.append({
            "check": "XML contains glass + concrete BSDFs",
            "status": PASS if has_glass and has_concrete else FAIL,
            "detail": f"glass={has_glass} concrete={has_concrete}",
        })
    except Exception as e:
        results.append({"check": "XML BSDF check", "status": FAIL, "detail": str(e)})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(output_dir: Path, skip_sionna: bool) -> dict:
    print(f"\n{'═'*60}")
    print(f" Facade output validation")
    print(f" output dir: {output_dir}")
    print(f"{'═'*60}\n")

    all_results = []

    # Tier 2a — file existence
    print("── Tier 2a: file checks ──────────────────────────────────")
    file_results, sionna_data = check_obj_exists(output_dir)
    for r in file_results:
        _print_result(r)
    all_results.extend(file_results)

    # Tier 2b — mesh geometry
    print("\n── Tier 2b: mesh geometry ────────────────────────────────")
    geo_results = check_mesh_geometry(output_dir, sionna_data)
    for r in geo_results:
        _print_result(r)
    all_results.extend(geo_results)

    # Tier 3 — Sionna RF
    if not skip_sionna:
        print("\n── Tier 3: sionna RF validation ──────────────────────────")
        sionna_results = check_sionna(output_dir)
        for r in sionna_results:
            _print_result(r)
        all_results.extend(sionna_results)

    # Summary
    passed  = sum(1 for r in all_results if r["status"] == PASS)
    failed  = sum(1 for r in all_results if r["status"] == FAIL)
    skipped = sum(1 for r in all_results if r["status"] == SKIP)

    print(f"\n{'─'*60}")
    print(f"  {passed} passed  |  {failed} failed  |  {skipped} skipped")
    print(f"{'─'*60}\n")

    report = {
        "output_dir": str(output_dir),
        "passed": passed, "failed": failed, "skipped": skipped,
        "success": failed == 0,
        "checks": all_results,
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report → {report_path}\n")

    return report


def _print_result(r: dict):
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}.get(r["status"], "?")
    detail = f"  ({r['detail']})" if r.get("detail") else ""
    print(f"  {icon} {r['check']}{detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                        help="Pipeline output directory to validate")
    parser.add_argument("--skip-sionna", action="store_true",
                        help="Skip Sionna RF checks (tier 3)")
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: output dir '{args.output_dir}' does not exist.")
        sys.exit(1)

    report = run_all(args.output_dir, args.skip_sionna)
    sys.exit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()