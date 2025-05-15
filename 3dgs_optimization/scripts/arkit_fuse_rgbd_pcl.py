import os
import sys
import open3d as o3d
import numpy as np

sys.path.append("examples")  # noqa
from tqdm import tqdm
from datasets.arkitscenes import ARKitScenesDataset


def fuse_rgbd_point_clouds(data_root: str, output_filename: str = "fused_pcl.ply"):
    """
    Fuses RGB-D frames from a single scene into a single point cloud.

    Args:
        data_root (str): Path to the scene directory containing 'transforms_train.json'.
        split (str): Dataset split to use ('train', 'test', or 'train+test').
        output_filename (str): Name of the output fused point cloud file.
    """
    dataset = ARKitScenesDataset(
        data_root=data_root,
        split="train",
    )

    fused_pcd = o3d.geometry.PointCloud()
    for idx in tqdm(range(len(dataset)), desc="Fusing Frames"):
        data = dataset[idx]
        image_tensor = data["image"]  # Shape: [3, H, W], in [0, 255]
        depth_tensor = data.get("depth", None)  # Shape: [H, W], in meters
        camtoworld = data["camtoworld"].numpy()  # Shape: [4, 4]
        K = data["K"].numpy()  # Shape: [3, 3]

        # o3d also assume 3 x H x W shape
        image_np = image_tensor.contiguous().numpy().astype(np.uint8)
        color_o3d = o3d.geometry.Image(image_np)

        # Convert depth tensor to numpy array and Open3D Image
        depth_np = depth_tensor.squeeze(0).numpy()  # Shape: [H, W]
        depth_o3d = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=1.0,  # depth is already in meters
            depth_trunc=15.0,  # adjust as needed
            convert_rgb_to_intensity=False,
        )

        # Extract intrinsic parameters
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        width, height = image_np.shape[1], image_np.shape[0]  # [width, height]

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform(camtoworld)
        fused_pcd += pcd

    # optional
    voxel_size = 0.02  # adjust as needed
    fused_pcd_down = fused_pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = fused_pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cleaned_pcd = fused_pcd_down.select_by_index(ind)

    # Save the fused point cloud
    output_path = os.path.join(data_root, output_filename)
    o3d.io.write_point_cloud(output_path, cleaned_pcd)
    print(f"Saved fused point cloud to: {output_path}")


if __name__ == "__main__":
    data_root_dir = "data/41048190"
    fuse_rgbd_point_clouds(
        data_root=data_root_dir,
        output_filename="fused_point_cloud.ply",
    )
