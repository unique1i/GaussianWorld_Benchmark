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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    depth: np.array  # not used
    sam_mask: np.array  # modify -----
    mask_feat: np.array  # modify -----
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if not os.path.exists(image_path):
            # modify -----
            base, ext = os.path.splitext(image_path)
            if ext.lower() == ".jpg":
                image_path = base + ".png"
            elif ext.lower() == ".png":
                image_path = base + ".jpg"
            if not os.path.exists(image_path):
                continue
            # modify ----

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # NOTE: load SAM mask and CLIP feat. [OpenGaussian]
        mask_seg_path = os.path.join(
            images_folder[:-6],
            "language_features/" + extr.name.split("/")[-1][:-4] + "_s.npy",
        )
        mask_feat_path = os.path.join(
            images_folder[:-6],
            "language_features/" + extr.name.split("/")[-1][:-4] + "_f.npy",
        )
        if os.path.exists(mask_seg_path):
            sam_mask = np.load(mask_seg_path)  # [level=4, H, W]
        else:
            sam_mask = None
        if mask_feat_path is not None and os.path.exists(mask_feat_path):
            mask_feat = np.load(mask_feat_path)  # [level=4, H, W]
        else:
            mask_feat = None
        # modify -----

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            cx=width / 2,
            cy=height / 2,
            image=image,
            depth=None,
            sam_mask=sam_mask,
            mask_feat=mask_feat,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    if {"red", "green", "blue"}.issubset(vertices.data.dtype.names):
        colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
    else:
        colors = np.random.rand(positions.shape[0], 3)
    if {"nx", "ny", "nz"}.issubset(vertices.data.dtype.names):
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    else:
        normals = np.random.rand(positions.shape[0], 3)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        # ----- modify -----
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"]
        # ----- modify -----

        # modify -----
        cx, cy = -1, -1
        if "cx" in contents.keys():
            cx = contents["cx"]
            cy = contents["cy"]
        elif "h" in contents.keys():
            cx = contents["w"] / 2
            cy = contents["h"] / 2
        # modify -----

        frames = contents["frames"]
        # for idx, frame in enumerate(frames):
        for idx, frame in tqdm(
            enumerate(frames), total=len(frames), desc="load images"
        ):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1  # TODO

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            if not os.path.exists(image_path):
                # modify -----
                base, ext = os.path.splitext(image_path)
                if ext.lower() == ".jpg":
                    image_path = base + ".png"
                elif ext.lower() == ".png":
                    image_path = base + ".jpg"
                if not os.path.exists(image_path):
                    continue
                # modify ----

            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            # NOTE: load SAM mask and CLIP feat. [OpenGaussian]
            mask_seg_path = os.path.join(
                path,
                "language_features/" + frame["file_path"].split("/")[-1] + "_s.npy",
            )
            mask_feat_path = os.path.join(
                path,
                "language_features/" + frame["file_path"].split("/")[-1] + "_f.npy",
            )
            if os.path.exists(mask_seg_path):
                sam_mask = np.load(mask_seg_path)  # [level=4, H, W]
            else:
                sam_mask = None
            if os.path.exists(mask_feat_path):
                mask_feat = np.load(mask_feat_path)  # [num_mask, dim=512]
            else:
                mask_feat = None
            # modify -----

            # ----- modify -----
            if "K" in frame.keys():
                cx = frame["K"][0][2]
                cy = frame["K"][1][2]
            if cx == -1:
                cx = image.size[0] / 2
                cy = image.size[1] / 2
            # ----- modify -----

            # ----- modify -----
            if fovx == None:
                if "K" in frame.keys():
                    focal_length = frame["K"][0][0]
                if "fl_x" in contents.keys():
                    focal_length = contents["fl_x"]
                if "fl_x" in frame.keys():
                    focal_length = frame["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx
                FovX = fovy
            # ----- modify -----

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    cx=cx,
                    cy=cy,
                    image=image,
                    depth=None,
                    sam_mask=sam_mask,
                    mask_feat=mask_feat,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    if os.path.exists(os.path.join(path, "transforms_test.json")):
        test_cam_infos = readCamerasFromTransforms(
            path, "transforms_test.json", white_background, extension
        )
    else:
        test_cam_infos = train_cam_infos

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms_scannetpp(
    path, transformsfile, white_background, extension=""
):
    cam_infos = []
    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"]
        cy = contents["cy"]
        if "crop_edge" in contents:
            cx -= contents["crop_edge"]
            cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "width" in contents and "height" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["width"], contents["height"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2

        fovx = focal2fov(focal_len_x, width)  #
        fovy = focal2fov(focal_len_y, height)

        FovY = fovy
        FovX = fovx

        frames = contents["frames"]
        # frames = frames[::skip]
        # raise ValueError("Frames: ", frames)
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            image_path = path  # os.path.join(path, 'color_interval')
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            applied_transform = np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=float,
            )
            c2w = np.dot(applied_transform, c2w)
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            w2c = np.linalg.inv(c2w)
            w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, "undistorted_images", cam_name)
            image_name = Path(cam_name).stem

            image = Image.open(image_path + extension)
            # NOTE: load SAM mask and CLIP feat. [OpenGaussian]

            if "resize" in contents:
                resize = contents["resize"]
            else:
                resize = [584, 876]

            if resize is not None:
                # resize
                # resize = [584, 876]
                resize_img = (
                    (resize[1], resize[0])
                    if resize[1] > resize[0]
                    else (resize[0], resize[1])
                )
                # image = image.resize(resize_img, Image.Resampling.LANCZOS)
                image = image.resize(resize_img, Image.LANCZOS)

            if "crop_edge" in contents:
                image = image.crop(
                    (
                        contents["crop_edge"],
                        contents["crop_edge"],
                        image.width - contents["crop_edge"],
                        image.height - contents["crop_edge"],
                    )
                )

            mask_seg_path = os.path.join(
                path.replace("original_data", "language_features_clip").replace(
                    "dslr", ""
                ),
                image_name + "_s.npy",
            )
            mask_feat_path = os.path.join(
                path.replace("original_data", "language_features_clip").replace(
                    "dslr", ""
                ),
                image_name + "_f.npy",
            )

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue

            print("debug FovX", FovX, "FovY", FovY)
            # os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_s.npy")
            # mask_feat_path = os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_f.npy")
            if os.path.exists(mask_seg_path):
                sam_mask = np.load(mask_seg_path)  # [level=4, H, W]
                print("sam_mask", sam_mask.shape)
            else:
                raise ValueError("No mask file found: ", mask_seg_path)
            if os.path.exists(mask_feat_path):
                mask_feat = np.load(mask_feat_path)  # [num_mask, dim=512]
                print("mask_feat", mask_feat.shape)
            else:
                raise ValueError("No mask file found: ", mask_feat_path)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    cx=cx,
                    cy=cy,
                    image=image,
                    depth=None,
                    sam_mask=sam_mask,
                    mask_feat=mask_feat,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readCamerasFromTransforms_scannet(
    path, transformsfile, white_background, extension="", feat_root=None
):
    cam_infos = []
    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"]
        cy = contents["cy"]
        if "crop_edge" in contents:
            cx -= contents["crop_edge"]
            cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        elif "width" in contents and "height" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["width"], contents["height"]
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2

        fovx = focal2fov(focal_len_x, width)  #
        fovy = focal2fov(focal_len_y, height)

        FovY = fovy
        FovX = fovx

        frames = contents["frames"]
        # frames = frames[::skip]
        # raise ValueError("Frames: ", frames)
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            image_path = path  # os.path.join(path, 'color_interval')
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem

            image = Image.open(image_path + extension)

            print("debug FovX", FovX, "FovY", FovY)
            # NOTE: load SAM mask and CLIP feat. [OpenGaussian]

            if "resize" in contents:
                resize = contents["resize"]
            else:
                resize = [height, width]

            if resize is not None:
                # resize
                # resize = [584, 876]
                resize_img = (
                    (resize[1], resize[0])
                    if resize[1] > resize[0]
                    else (resize[0], resize[1])
                )
                # image = image.resize(resize_img, Image.Resampling.LANCZOS)
                image = image.resize(resize_img, Image.LANCZOS)

            if "crop_edge" in contents:
                image = image.crop(
                    (
                        contents["crop_edge"],
                        contents["crop_edge"],
                        image.width - contents["crop_edge"],
                        image.height - contents["crop_edge"],
                    )
                )

            if feat_root is None:
                mask_seg_path = os.path.join(
                    path.replace("original_data", "language_features_clip"),
                    image_name + "_s.npy",
                )
                mask_feat_path = os.path.join(
                    path.replace("original_data", "language_features_clip"),
                    image_name + "_f.npy",
                )
            else:
                print("feat_root", feat_root)
                mask_seg_path = os.path.join(feat_root, image_name + "_s.npy")
                mask_feat_path = os.path.join(feat_root, image_name + "_f.npy")

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue

            try:
                if os.path.exists(mask_seg_path):
                    sam_mask = np.load(mask_seg_path)  # [level=4, H, W]
                    print("sam_mask", sam_mask.shape)

                if os.path.exists(mask_feat_path):
                    mask_feat = np.load(mask_feat_path)  # [num_mask, dim=512]
                    print("clip feat", mask_feat.shape)
            except:
                print("mask_seg_path loading wrong", mask_seg_path)
                print("mask_feat_path loading wrong", mask_feat_path)
                continue

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    cx=cx,
                    cy=cy,
                    image=image,
                    depth=None,
                    sam_mask=sam_mask,
                    mask_feat=mask_feat,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readScanNetInfo(path, white_background, eval, extension=".png", feat_root=None):
    print("Reading Training Transforms")
    basename = os.path.basename(path)
    lang_path = os.path.join(path, "rgb_feature_langseg")
    transform_path = os.path.join(path, "lang_feat_selected_imgs.json")

    all_cam_infos = readCamerasFromTransforms_scannet(
        path, transform_path, white_background, extension=extension, feat_root=feat_root
    )

    train_cam_infos = (
        all_cam_infos  # [c for idx, c in enumerate(all_cam_infos) if idx % 8 != 0]
    )
    test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readScanNetPPInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    transform_path = os.path.join(path, "nerfstudio", "lang_feat_selected_imgs.json")

    all_cam_infos = readCamerasFromTransforms_scannetpp(
        path, transform_path, white_background, extension=""
    )

    train_cam_infos = (
        all_cam_infos  # [c for idx, c in enumerate(all_cam_infos) if idx % 8 != 0]
    )
    test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms_matterport(
    path, transformsfile, white_background, extension=""
):
    cam_infos = []
    with open(transformsfile) as json_file:
        contents = json.load(json_file)

        width = contents["width"]
        height = contents["height"]
        frames = contents["frames"]
        # frames = frames[::skip]
        # raise ValueError("Frames: ", frames)
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            image_path = path  # os.path.join(path, 'color_interval')
            cam_name = frame["file_path"]
            fx = frame["fx"]
            fy = frame["fy"]
            cx = frame["cx"]
            cy = frame["cy"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem

            image = Image.open(image_path + extension)

            read_w, read_h = image.size
            assert (
                read_h == height and read_w == width
            ), f"image size {read_h}x{read_w} not match {height}x{width}"
            FovX = focal2fov(fx, width)  #
            FovY = focal2fov(fy, height)

            print("debug FovX", FovX, "FovY", FovY)
            # NOTE: load SAM mask and CLIP feat. [OpenGaussian]

            mask_seg_path = os.path.join(
                path.replace("original_data", "language_features_clip"),
                image_name + "_s.npy",
            )
            mask_feat_path = os.path.join(
                path.replace("original_data", "language_features_clip"),
                image_name + "_f.npy",
            )

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                continue

            # os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_s.npy")
            # mask_feat_path = os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_f.npy")
            if os.path.exists(mask_seg_path):
                sam_mask = np.load(mask_seg_path)  # [level=4, H, W]
                print("sam_mask", sam_mask.shape)
            else:
                raise ValueError("No mask file found: ", mask_seg_path)
            if os.path.exists(mask_feat_path):
                mask_feat = np.load(mask_feat_path)  # [num_mask, dim=512]
                print("mask_feat", mask_feat.shape)
            else:
                raise ValueError("No mask file found: ", mask_feat_path)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    cx=cx,
                    cy=cy,
                    image=image,
                    depth=None,
                    sam_mask=sam_mask,
                    mask_feat=mask_feat,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readMatterport3DInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    transform_path = os.path.join(path, "lang_feat_selected_imgs.json")

    all_cam_infos = readCamerasFromTransforms_matterport(
        path, transform_path, white_background, extension=""
    )

    train_cam_infos = (
        all_cam_infos  # [c for idx, c in enumerate(all_cam_infos) if idx % 8 != 0]
    )
    test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Scannet": readScanNetInfo,
    "ScanNetpp": readScanNetPPInfo,
    "Matterport3D": readMatterport3DInfo,
    "Holicity": readScanNetInfo,
}
