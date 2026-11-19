"""
Phase 3 - Step 1: Load and reconstruct a full dental arch
from Person 2's per-tooth OBJ output.

Folder structure (from Person 2's README):
    processed_part1_upper/CASE_ID/tooth_11.obj, tooth_12.obj, ...
    processed_part1_lower/CASE_ID/tooth_31.obj, tooth_32.obj, ...

Since teeth are NOT individually centered (original coordinates preserved),
simply loading and merging them recreates the full arch shape - no manual
positioning needed at this stage.
"""

import os
import glob
import json
import numpy as np
import trimesh
import open3d as o3d


def trimesh_to_open3d(tm_mesh):
    """Convert a trimesh.Trimesh into an o3d.geometry.TriangleMesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tm_mesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm_mesh.faces))
    return mesh


class ArchLoader:
    def __init__(self, case_dir):
        """
        case_dir: path to a single case folder, e.g.
                  'processed_part1_upper/BUGB9DY1'
        """
        self.case_dir = case_dir
        self.tooth_meshes = {}   # {fdi_number: o3d.geometry.TriangleMesh}
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        meta_path = os.path.join(self.case_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}

    def load_teeth(self):
        """Load every tooth_*.obj file in the case folder individually.

        Uses trimesh to read the OBJ (more tolerant of format quirks than
        Open3D's own reader, which was returning empty meshes on these
        files), then converts to Open3D for merging/visualization.
        """
        obj_files = sorted(glob.glob(os.path.join(self.case_dir, "tooth_*.obj")))
        if not obj_files:
            raise FileNotFoundError(f"No tooth_*.obj files found in {self.case_dir}")

        for path in obj_files:
            fname = os.path.basename(path)
            # tooth_11.obj -> 11
            fdi = int(fname.replace("tooth_", "").replace(".obj", ""))

            tm_mesh = trimesh.load(path, force="mesh", process=False)
            if tm_mesh.vertices.shape[0] == 0:
                print(f"WARNING: {fname} loaded with 0 vertices even via trimesh")
                continue

            mesh = trimesh_to_open3d(tm_mesh)
            mesh.compute_vertex_normals()
            self.tooth_meshes[fdi] = mesh

        print(f"Loaded {len(self.tooth_meshes)} teeth from {self.case_dir}")
        return self.tooth_meshes

    def get_combined_arch(self, color_by_tooth=True):
        """
        Merge all loaded teeth into a single mesh, preserving their
        original relative positions (this reconstructs the arch shape).
        """
        if not self.tooth_meshes:
            self.load_teeth()

        combined = o3d.geometry.TriangleMesh()
        rng = np.random.default_rng(42)

        for fdi, mesh in self.tooth_meshes.items():
            m = mesh  # already in original arch coordinates - just merge
            if color_by_tooth:
                color = rng.uniform(0.3, 1.0, size=3)
                m.paint_uniform_color(color)
            combined += m

        combined.compute_vertex_normals()
        return combined

    def get_arch_center(self):
        """Center point of the reconstructed arch (its bounding-box center)."""
        combined = self.get_combined_arch(color_by_tooth=False)
        return combined.get_center()


def visualize_arch(case_dir):
    """Sanity check: does the arch look right when just loaded and merged?"""
    loader = ArchLoader(case_dir)
    arch = loader.get_combined_arch()
    print("Arch center:", loader.get_arch_center())
    print("Bounding box:", arch.get_axis_aligned_bounding_box())

    o3d.visualization.draw_geometries(
        [arch],
        window_name="Reconstructed Dental Arch",
        width=900,
        height=700,
    )


if __name__ == "__main__":
    # Point this at a real case folder from Person 2's pushed output
    CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"
    visualize_arch(CASE_DIR)