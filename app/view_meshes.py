import os
import trimesh

mesh_dir = "output/meshes"
meshes = []

for mesh_file in os.listdir(mesh_dir):
    if mesh_file.endswith(".obj"):
        mesh_path = os.path.join(mesh_dir, mesh_file)
        mesh = trimesh.load(mesh_path, force="mesh")
        meshes.append(mesh)

if not meshes:
    print("no meshes found in output/meshes")
    raise SystemExit(1)

combined = trimesh.util.concatenate(meshes)
out_path = "output/combined_mesh.obj"
combined.export(out_path)

print(f"saved {out_path}")
print(combined.bounds)