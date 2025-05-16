#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np

import os
from tqdm import tqdm

# from os import makedirs
# from gaussian_renderer import render
# import torchvision
# from utils.general_utils import safe_state
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
_have_open3d = True
try:
    pass
except:
    _have_open3d = False
import argparse
import json
import math
from PIL import Image
import cv2


try:
    from pytorch3d.renderer.mesh import rasterize_meshes
    from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
    from pytorch3d.renderer.mesh.shader import (
        SoftDepthShader,
    )  # Used for perspective correct Z
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        TexturesUV,
        MeshRasterizer,
        MeshRenderer,
        HardPhongShader,
        PointLights,
        RasterizationSettings,
        BlendParams,
    )
    from pytorch3d.renderer.mesh.shading import interpolate_face_attributes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PerspectiveCameras,
        SoftPhongShader,
    )

    _has_pytorch3d = True
except ImportError:
    _has_pytorch3d = False
    print("WARNING: PyTorch3D not found. Optimized rasterization is unavailable.")

import sys


try:
    import renderpy
except ImportError:
    print(
        "renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy"
    )
    sys.exit(1)


from matplotlib.colors import hsv_to_rgb


def generate_distinct_colors(n=100, seed=42):
    np.random.seed(seed)

    # Evenly space hues, randomize saturation and value a bit
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)  # shuffle to prevent similar colors being close in order
    saturations = np.random.uniform(0.6, 0.9, n)
    values = np.random.uniform(0.7, 0.95, n)

    hsv_colors = np.stack([hues, saturations, values], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors


# Example usage
SCANNET_100_COLORS = generate_distinct_colors(100)
HOLICITY_6_COLORS = generate_distinct_colors(6)


def project_points_to_2d(
    points_3d,
    labels,
    depth_img,
    world_to_camera,
    camera_matrix,
    dist_coeffs,
    image_size,
):
    """
    Project 3D points to 2D using OpenCV's projection function

    Args:
        points_3d: Nx3 numpy array of 3D points in world coordinates
        labels: Nx1 numpy array of semantic labels
        depth_img: HxW numpy array of depth values
        world_to_camera: 4x4 transformation matrix
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients (usually just zeros for modern cameras)
        image_size: (width, height) tuple

    Returns:
        points_2d: Mx2 array of 2D coordinates (only points in front of camera)
        visible_labels: Mx1 array of corresponding labels
        depth: Mx1 array of depth values
    """
    # Transform points to camera coordinates
    width, height = image_size

    points_cam = cv2.transform(
        points_3d.reshape(-1, 1, 3), world_to_camera[:3]
    ).reshape(-1, 3)

    # Filter points behind camera (z < 0)
    depth_values = points_cam[:, 2]
    in_front = depth_values > 0

    # Project to 2D using OpenCV's projectPoints
    points_2d, _ = cv2.projectPoints(
        points_cam,
        np.zeros(3),  # rvec (zero rotation)
        np.zeros(3),  # tvec (zero translation)
        camera_matrix,
        dist_coeffs,
    )
    points_2d = points_2d.reshape(-1, 2)

    # visibility_2d_buffer = np.zeros((image_size[1], image_size[0]), dtype=np.int32)
    seg_image = np.zeros((height, width), dtype=np.int32) - 1
    depth_2d_buffer = np.zeros((image_size[1], image_size[0]), dtype=np.float32) + 1e3
    for iter, index_2d in enumerate(points_2d):
        if in_front[iter]:
            x, y = int(index_2d[0]), int(index_2d[1])
            if 0 <= x < width and 0 <= y < height:
                depth_xy = depth_img[y, x]
                render_depth = depth_values[iter]
                if np.abs(depth_xy - render_depth) < 0.1:
                    seg_image[y, x] = labels[iter]

    return seg_image


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def get_argparse():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/srv/beegfs02/scratch/qimaqi_data/data/holicity_val_set_suite/",
        #'/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/holicity_mini_val_set_suite/',
        help="Path to the dataset root",
    )
    parser.add_argument(
        "--pred_label_root",
        type=str,
        default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/results/holicity/",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/3dgs-gradient-backprojection-benchmark/splits/holicity_mini_val.txt",
        help="Path to the split file",
    )
    parser.add_argument(
        "--preprocessed_data_root",
        type=str,
        default="/srv/beegfs02/scratch/qimaqi_data/data/holicity_val_set_preprocessed/",
        help="Path to the preprocessed data root",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_argparse()
    split_path = args.split_path
    val_split = np.loadtxt(split_path, dtype=str)
    # print("val_split", val_split)
    val_split = sorted(val_split)
    preprocessed_data_root = args.preprocessed_data_root

    val_split = ["iLPO_wRcFDcg4muQaPXfFg_HD"]
    for val_i in tqdm(val_split):
        val_data_path = os.path.join(args.dataset_root, "original_data", val_i)
        if "scannetpp" in args.dataset_root:
            selected_json = os.path.join(
                val_data_path, "dslr", "nerfstudio", "lang_feat_selected_imgs.json"
            )
        else:
            selected_json = os.path.join(val_data_path, "lang_feat_selected_imgs.json")

        points_3d_path = os.path.join(preprocessed_data_root, val_i, "coord.npy")
        gt_coord = np.load(points_3d_path)
        gt_seg_path = os.path.join(preprocessed_data_root, val_i, "segment.npy")
        gt_seg = np.load(gt_seg_path)
        print("gt_seg", gt_seg.min(), gt_seg.max(), np.unique(gt_seg))
        pred_seg_path = os.path.join(
            args.pred_label_root,
            val_i,
            f"holicity_backproject_lseg_{val_i}_semseg_pred.npy",
        )
        pred_seg = np.load(pred_seg_path)
        results_save_root = os.path.join(args.pred_label_root, val_i)

        with open(selected_json, "r") as f:
            contents = json.load(f)
        selected_frames = contents["frames"]
        selected_imgs_list = [frame_i["file_path"] for frame_i in selected_frames]

        # holicity
        org_height = contents["height"]
        org_width = contents["width"]
        focal_len_x = contents["fx"]
        focal_len_y = contents["fy"]
        cx = contents["cx"]
        cy = contents["cy"]
        fovx = focal2fov(focal_len_x, org_width)
        fovy = focal2fov(focal_len_y, org_height)
        width = org_width
        height = org_height

        FovY = fovy
        FovX = fovx
        frames = contents["frames"]

        # coord = coord - coord.min(axis=0)
        # verts = torch.tensor(coord, dtype=torch.float32).cuda()
        # print("x min", verts[:, 0].min(), "x max", verts[:, 0].max())
        # print("y min", verts[:, 1].min(), "y max", verts[:, 1].max())
        # print("z min", verts[:, 2].min(), "z max", verts[:, 2].max())

        # faces = torch.tensor(mesh.triangles, dtype=torch.int32).cuda()

        points_3d = np.array(gt_coord)
        labels = np.array(gt_seg)
        pred_seg = np.array(pred_seg)
        # sparse_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_sparse')
        # dense_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_dense')
        # os.makedirs(sparse_save_path, exist_ok=True)
        # os.makedirs(dense_save_path, exist_ok=True)
        semantic_save_path = os.path.join(results_save_root, "segmentation_2d")
        semantic_save_path_img = os.path.join(results_save_root, "segmentation_2d_img")
        pred_save_path = os.path.join(results_save_root, "openvocb_2d_pred")
        os.makedirs(semantic_save_path, exist_ok=True)
        os.makedirs(semantic_save_path_img, exist_ok=True)
        os.makedirs(pred_save_path, exist_ok=True)

        for idx, frame in enumerate(frames):

            cam_name = frame["file_path"].split("/")[-1]
            depth_img_path = os.path.join(
                val_data_path, "depth", cam_name.replace("imag.jpg", "dpth.npz")
            )
            depth_img = np.load(depth_img_path)["depth"] / 100.0
            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            world_to_camera = w2c
            resize_camera_matrix = np.array(
                [[focal_len_x, 0, cx], [0, focal_len_y, cy], [0, 0, 1]], dtype=float
            )
            dist_coeffs = np.zeros((4, 1), dtype=float)
            sparse_size = [width, height]

            seg_image_sparse = project_points_to_2d(
                points_3d,
                labels,
                depth_img,
                world_to_camera,
                resize_camera_matrix,
                dist_coeffs,
                sparse_size,
            )

            seg_image_save_path = os.path.join(
                semantic_save_path, cam_name.replace("imag.jpg", "seg.npy")
            )
            seg_image_save_path_img = os.path.join(
                semantic_save_path_img, cam_name.replace("imag.jpg", "seg.png")
            )
            pred_image_sparse = project_points_to_2d(
                points_3d,
                pred_seg,
                depth_img,
                world_to_camera,
                resize_camera_matrix,
                dist_coeffs,
                sparse_size,
            )
            pred_image_save_path = os.path.join(
                pred_save_path, cam_name.replace("imag.jpg", "seg.npy")
            )
            pred_image_save_path_img = os.path.join(
                pred_save_path, cam_name.replace("imag.jpg", "seg.png")
            )

            # save sparse segmentation
            np.save(seg_image_save_path, seg_image_sparse)
            # seg_img_vis = seg_image_sparse.copy()
            # seg_img = seg_img.astype(np.int32)
            seg_img_color_maped = seg_image_sparse.copy().astype(np.int32)
            seg_img_color_maped[seg_img_color_maped == -1] = 5
            # seg_img_color_maped = cv2.applyColorMap(seg_img_color_maped, cv2.COLORMAP_JET)
            seg_img_color_maped = HOLICITY_6_COLORS[seg_img_color_maped]
            seg_img_color_maped = (seg_img_color_maped * 255).astype(np.uint8)
            seg_img_color_maped = Image.fromarray(seg_img_color_maped)
            seg_img_color_maped.save(seg_image_save_path_img)
            print("finished saving segmentation image", seg_image_save_path_img)

            np.save(pred_image_save_path, pred_image_sparse)
            pred_img_vis = pred_image_sparse.copy()
            pred_img = pred_img_vis.astype(np.int32)
            pred_img_color_maped = pred_img.copy()
            pred_img_color_maped[pred_img_color_maped == -1] = 5
            # pred_img_color_maped = cv2.applyColorMap(pred_img_color_maped, cv2.COLORMAP_JET)
            pred_img_color_maped = HOLICITY_6_COLORS[pred_img_color_maped]
            pred_img_color_maped = (pred_img_color_maped * 255).astype(np.uint8)
            pred_img_color_maped = Image.fromarray(pred_img_color_maped)
            pred_img_color_maped.save(pred_image_save_path_img)
            print("finished saving segmentation image", pred_image_save_path_img)
