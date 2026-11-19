from arch_loader import ArchLoader

CASE_DIR = r"C:\Users\utkar\Downloads\DENTAL\VITA-3D-Master-Shade-System-for-Dental-Simulators\processed_part1_upper\M7RTPNPC"
loader = ArchLoader(CASE_DIR)
loader.load_teeth()

for fdi, mesh in loader.tooth_meshes.items():
    print(f"Tooth {fdi}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")