"""
This script reads the original point cloud data from the ScanNet dataset and downsamples it using Open3D.
"""

from pathlib import Path
import open3d as o3d
from tqdm import tqdm

root_data_folder = [
    Path("/nvmestore/yli7/datasets/scannet/scans_test"),
    Path("/nvmestore/yli7/datasets/scannet/scans"),
]

for root_data_folder in root_data_folder:
    if not root_data_folder.exists():
        print(f"Data folder {root_data_folder} not found. Skipping.")
        continue
    for scene_folder in tqdm(root_data_folder.iterdir()):
        if scene_folder.is_dir():  # Check if it is a directory
            scene_name = scene_folder.name
            ply_file_path = scene_folder / f"{scene_name}_vh_clean.ply"

            if ply_file_path.exists():
                print(f"Downsampling {ply_file_path}")
                pcd = o3d.io.read_point_cloud(str(ply_file_path))

                # Print the number of vertices before downsampling
                num_vertices_before = len(pcd.points)
                voxel_size = 0.02
                downsampled_pcd = pcd.voxel_down_sample(voxel_size)

                # Print the number of vertices after downsampling
                num_vertices_after = len(downsampled_pcd.points)
                print(
                    f"Vertices before/after: {num_vertices_before} -> {num_vertices_after}"
                )

                output_ply_file = scene_folder / "points3d.ply"
                o3d.io.write_point_cloud(str(output_ply_file), downsampled_pcd)
            else:
                print(f"PLY file {ply_file_path} not found. Skipping this scene.")
