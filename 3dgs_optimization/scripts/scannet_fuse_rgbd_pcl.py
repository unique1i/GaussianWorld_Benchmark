"""
This script processes the rgbd images of a scannet scene to generate a point cloud using the depth images.
"""

import os
import zipfile
import numpy as np
from PIL import Image
import open3d as o3d
import argparse


def load_intrinsics(intrinsics_file):
    with open(intrinsics_file, "r") as f:
        lines = f.readlines()
        fx = float(lines[0].split()[0])
        fy = float(lines[1].split()[1])
        cx = float(lines[0].split()[2])
        cy = float(lines[1].split()[2])
    return fx, fy, cx, cy


def create_point_cloud(color_img, depth_img, fx, fy, cx, cy):
    height, width = depth_img.shape

    # Convert depth image to meters
    z = depth_img
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack the coordinates and filter out points with small depth values
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color_img.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]

    # Filter out points with depth < 0.01 meters
    valid_mask = points[:, 2] >= 0.01
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points, colors


def apply_pose(points, pose):
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply the pose transformation: cam2world transformation matrix is applied on the left
    transformed_points_homogeneous = pose @ points_homogeneous.T

    # Convert back to Cartesian coordinates
    transformed_points = transformed_points_homogeneous[:3, :].T

    return transformed_points


def process_images(scene_folder):
    print(f"Processing scene: {scene_folder}")
    color_zip_path = os.path.join(scene_folder, "color_interval.zip")
    depth_zip_path = os.path.join(scene_folder, "depth.zip")
    pose_file = os.path.join(scene_folder, "pose", "camera_poses.npy")
    intrinsics_file = os.path.join(scene_folder, "intrinsic", "intrinsic_depth.txt")

    # Check if all necessary files exist
    required_files = [color_zip_path, depth_zip_path, pose_file, intrinsics_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required file missing: {file_path}. Skipping this scene.")
            return

    # Load camera intrinsics for depth
    fx, fy, cx, cy = load_intrinsics(intrinsics_file)

    crop_edge = 10
    cx -= crop_edge
    cy -= crop_edge

    depth_scale = 1000.0  # check the dataset's depth scale

    camera_poses = np.load(pose_file)  # n*4*4 camera2world poses
    combined_points = []
    combined_colors = []

    with (
        zipfile.ZipFile(color_zip_path, "r") as color_zip,
        zipfile.ZipFile(depth_zip_path, "r") as depth_zip,
    ):
        color_files = sorted(color_zip.namelist())
        for color_file in color_files:
            frame_index = int(os.path.splitext(os.path.basename(color_file))[0])

            # Load color image
            with color_zip.open(color_file) as cf:
                color_img = Image.open(cf).convert("RGB")
                color_img = color_img.resize((640, 480))
                color_img = color_img.crop(
                    (crop_edge, crop_edge, 640 - crop_edge, 480 - crop_edge)
                )
                color_img = np.array(color_img)

            # Load depth image
            depth_file = f"{str(frame_index).zfill(5)}.png"
            with depth_zip.open(depth_file) as df:
                depth_img = Image.open(df)
                depth_img = (
                    np.array(depth_img).astype(np.float32) / depth_scale
                )  # Convert to meters
                depth_img = depth_img[
                    crop_edge : 480 - crop_edge, crop_edge : 640 - crop_edge
                ]

            # Create the point cloud manually, excluding points with depth < 0.01 meters
            points, colors = create_point_cloud(color_img, depth_img, fx, fy, cx, cy)

            # Get the camera pose
            pose = camera_poses[frame_index]

            # Apply the pose transformation manually
            transformed_points = apply_pose(points, pose)

            # Accumulate points and colors
            combined_points.append(transformed_points)
            combined_colors.append(colors)

    if combined_points and combined_colors:
        # Combine all points and colors
        combined_points = np.vstack(combined_points)
        combined_colors = np.vstack(combined_colors)

        # Create Open3D point cloud and save it
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)

        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        output_file = os.path.join(scene_folder, "points3d_depth.ply")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Saved point cloud to: {output_file}")
    else:
        print(f"No valid points found for scene: {scene_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process scene images to generate point clouds."
    )
    parser.add_argument(
        "--scene_folder", type=str, required=True, help="Path to the scene folder."
    )
    args = parser.parse_args()

    scene_folder = args.scene_folder
    if os.path.isdir(scene_folder):
        process_images(scene_folder)
    else:
        print(f"Scene folder does not exist: {scene_folder}")
