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
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

# from colorama import Fore, init, Style


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None


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
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
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
    try:
        colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

    # extract mesh information
    # mesh.compute_vertex_normals(normalized=True)
    # coord = np.array(mesh.vertices).astype(np.float32)
    # color = (np.array(mesh.vertex_colors) * 255).astype(np.uint8)
    # normal = np.array(mesh.vertex_normals).astype(np.float32)

    # return BasicPointCloud(points=positions, colors=colors, normals=normals)


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
    # reading_dir_F = "language_feature" if language_feature == None else language_feature
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        print("##### Colmap dataset evaluation mode ######")
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        print("#### Using all cameras for training ####")
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
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name_F = os.path.join(path, frame["file_path"] + "") # TODO: extension?

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # language_feature_path = os.path.join(path, cam_name_F)
            # language_feature_name = Path(cam_name_F).stem
            # language_feature = Image.open(language_feature_path) # TODO: data read

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
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
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

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


def readCamerasFromTransforms_matterport(
    path,
    transformsfile,
    depths_folder,
    white_background,
    is_test,
    extension=".png",
    resize=None,
    language_features_name=None,
):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        # sort frames by frame["file_path"]
        frames = sorted(frames, key=lambda x: x["file_path"])
        width = contents["width"]
        height = contents["height"]

        for idx, frame in enumerate(frames):
            image_path = (
                path.replace("nerfstudio", "undistorted_images")
                if "nerfstudio" in path
                else path
            )  # not used for feature extraction
            cam_name = frame["file_path"]

            fx = frame["fx"]
            fy = frame["fy"]
            cx = frame["cx"]
            cy = frame["cy"]
            focal_len_x = fx
            focal_len_y = fy

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, f"{cam_name + extension}")
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)

            if resize is not None:
                # resize = [584, 876]
                resize_img = (
                    (resize[1], resize[0])
                    if resize[1] > resize[0]
                    else (resize[0], resize[1])
                )
                # print("resize_img: ", resize_img)
                # print("image.size: ", image.size)
                # resize we need to also adjust the fovx
                # image = image.resize(resize_img, Image.Resampling.LANCZOS)
                image = image.resize(resize_img, Image.LANCZOS)
                resize_ratio_x = resize[1] / width
                resize_ratio_y = resize[0] / height
                assert (
                    resize_ratio_x == resize_ratio_y
                ), "resize ratio x and y should be the same"
                fx_resize = fx * resize_ratio_x
                fy_resize = fy * resize_ratio_y
                cx_resize = cx * resize_ratio_x
                cy_resize = cy * resize_ratio_y
                FovX = focal2fov(fx_resize, resize[1])
                FovY = focal2fov(fy_resize, resize[0])

            if language_features_name is not None:
                mask_seg_path = os.path.join(
                    path, language_features_name, image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, language_features_name, image_name + "_f.npy"
                )
            else:
                mask_seg_path = os.path.join(
                    path, "language_features_dim3", image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, "language_features_dim3", image_name + "_f.npy"
                )

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    image=image,
                    cx=cx,
                    cy=cy,
                    fx=focal_len_x,
                    fy=focal_len_y,
                )
            )

    return cam_infos


def readCamerasFromTransforms_opencv(
    path,
    transformsfile,
    depths_folder,
    white_background,
    is_test,
    extension=".png",
    resize=None,
    language_features_name=None,
):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"]
        cy = contents["cy"]
        # if "crop_edge" in contents:
        #     cx -= contents["crop_edge"]
        #     cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        if "width" in contents and "height" in contents:
            # nerfstudio case, fx, fy, cx, cy in nerfstudio json are already for image size 640x480
            width, height = contents["width"], contents["height"]

        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
        #     if "crop_edge" in contents:
        #         width -= 2*contents["crop_edge"]
        #         height -= 2*contents["crop_edge"]
        # else:
        #     # if not specify, we assume the weight and height are twice the cx and cy
        # width, height = cx * 2, cy * 2
        fovx = focal2fov(focal_len_x, width)
        fovy = focal2fov(focal_len_y, height)

        FovY = fovy
        FovX = fovx
        frames = contents["frames"]
        # sort frames by frame["file_path"]
        frames = sorted(frames, key=lambda x: x["file_path"])
        # take frames by some interval
        # frames = frames[::2]
        feat_find_counts = 0
        for idx, frame in enumerate(frames):
            image_path = (
                path.replace("nerfstudio", "undistorted_images")
                if "nerfstudio" in path
                else path
            )  # not used for feature extraction
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, f"{cam_name + extension}")
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if "resize" in contents:
                resize = contents["resize"]
            else:
                resize = [height, width]

            if resize is not None:
                # resize = [584, 876]
                resize_img = (
                    (resize[1], resize[0])
                    if resize[1] > resize[0]
                    else (resize[0], resize[1])
                )
                image = image.resize(resize_img, Image.LANCZOS)

            # if "crop_edge" in contents:
            #     image = image.crop(
            #         (
            #             contents["crop_edge"],
            #             contents["crop_edge"],
            #             image.width - contents["crop_edge"],
            #             image.height - contents["crop_edge"],
            #         )
            #     )

            # mask_seg_path = os.path.join(path.replace('original_data', 'language_features_clip'),image_name + "_s.npy")
            # mask_feat_path = os.path.join(path.replace('original_data', 'language_features_clip'),image_name + "_f.npy")
            if language_features_name is not None:
                mask_seg_path = os.path.join(
                    path, language_features_name, image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, language_features_name, image_name + "_f.npy"
                )
            else:
                mask_seg_path = os.path.join(
                    path, "language_features_dim3", image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, "language_features_dim3", image_name + "_f.npy"
                )

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue
            else:
                feat_find_counts += 1

                # resize_ratio_x = resize[1] / width
                # resize_ratio_y = resize[0] / height
                # assert resize_ratio_x == resize_ratio_y, "resize ratio x and y should be the same"
                # fx_resize = focal_len_x * resize_ratio_x
                # fy_resize = focal_len_y * resize_ratio_y
                # cx_resize = cx * resize_ratio_x
                # cy_resize = cy * resize_ratio_y
                # fovx = focal2fov(fx_resize, resize[1])
                # fovy = focal2fov(fy_resize, resize[0])

            # print("width, height: ", width, height)
            # print("focal_len_x, focal_len_y: ", focal_len_x, focal_len_y)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    image=image,
                    cx=cx,
                    cy=cy,
                    fx=focal_len_x,
                    fy=focal_len_y,
                )
            )
    if feat_find_counts == 0:
        raise ValueError("mask_seg_path not exists")

    return cam_infos


def readCamerasFromTransforms_scannet(
    path,
    transformsfile,
    depths_folder,
    white_background,
    is_test,
    extension=".png",
    resize=None,
    language_features_name=None,
):
    cam_infos = []

    example_mask_seg = None
    example_mask_seg_path = os.path.join(path, language_features_name, "00000_s.npy")
    example_mask = np.load(example_mask_seg_path)
    feat_H, feat_W = example_mask.shape[1], example_mask.shape[2]
    do_crop = True
    print("feat_H, feat_W: ", feat_H, feat_W)
    if feat_H == 456 and feat_W == 624:
        do_crop = True
    elif feat_H == 478 and feat_W == 640:
        do_crop = False

    with open(transformsfile) as json_file:

        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"]
        cy = contents["cy"]

        if do_crop:
            if "crop_edge" in contents:
                cx -= contents["crop_edge"]
                cy -= contents["crop_edge"]
            if "w" in contents and "h" in contents:
                # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
                width, height = contents["w"], contents["h"]
            if "width" in contents and "height" in contents:
                # nerfstudio case, fx, fy, cx, cy in nerfstudio json are already for image size 640x480
                width, height = contents["width"], contents["height"]

            elif "resize" in contents:
                # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
                width, height = contents["resize"]
                if "crop_edge" in contents:
                    width -= 2 * contents["crop_edge"]
                    height -= 2 * contents["crop_edge"]
            else:
                # if not specify, we assume the weight and height are twice the cx and cy
                width, height = cx * 2, cy * 2
        else:
            width = 640
            height = 478
            focal_len_y = focal_len_y * height / 480
        fovx = focal2fov(focal_len_x, width)
        fovy = focal2fov(focal_len_y, height)

        FovY = fovy
        FovX = fovx
        frames = contents["frames"]
        # sort frames by frame["file_path"]
        frames = sorted(frames, key=lambda x: x["file_path"])
        # take frames by some interval
        # frames = frames[::2]
        feat_find_counts = 0
        for idx, frame in enumerate(frames):
            image_path = (
                path.replace("nerfstudio", "undistorted_images")
                if "nerfstudio" in path
                else path
            )  # not used for feature extraction
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, f"{cam_name + extension}")
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if do_crop:
                if "resize" in contents:
                    resize = contents["resize"]
                else:
                    resize = [height, width]
                # resize = [478, 640]
                if resize is not None:
                    # resize = [584, 876]
                    resize_img = (
                        (resize[1], resize[0])
                        if resize[1] > resize[0]
                        else (resize[0], resize[1])
                    )
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
            else:
                resize = [478, 640]
                resize_img = (
                    (resize[1], resize[0])
                    if resize[1] > resize[0]
                    else (resize[0], resize[1])
                )
                image = image.resize(resize_img, Image.LANCZOS)

            # mask_seg_path = os.path.join(path.replace('original_data', 'language_features_clip'),image_name + "_s.npy")
            # mask_feat_path = os.path.join(path.replace('original_data', 'language_features_clip'),image_name + "_f.npy")
            if language_features_name is not None:
                mask_seg_path = os.path.join(
                    path, language_features_name, image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, language_features_name, image_name + "_f.npy"
                )
            else:
                mask_seg_path = os.path.join(
                    path, "language_features_dim3", image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, "language_features_dim3", image_name + "_f.npy"
                )

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue
            else:
                try:
                    mask_seg = np.load(mask_seg_path)
                    mask_feat = np.load(mask_feat_path)
                except:
                    print("mask_seg_path loading problem", mask_seg_path)
                    print("mask_feat_path loading problem", mask_feat_path)
                    continue
                feat_find_counts += 1

                # resize_ratio_x = resize[1] / width
                # resize_ratio_y = resize[0] / height
                # assert resize_ratio_x == resize_ratio_y, "resize ratio x and y should be the same"
                # fx_resize = focal_len_x * resize_ratio_x
                # fy_resize = focal_len_y * resize_ratio_y
                # cx_resize = cx * resize_ratio_x
                # cy_resize = cy * resize_ratio_y
                # fovx = focal2fov(fx_resize, resize[1])
                # fovy = focal2fov(fy_resize, resize[0])

            print("final image shape", image.size)
            print("final width, final height: ", width, height)
            # print("focal_len_x, focal_len_y: ", focal_len_x, focal_len_y)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    image=image,
                    cx=cx,
                    cy=cy,
                    fx=focal_len_x,
                    fy=focal_len_y,
                )
            )
    if feat_find_counts == 0:
        raise ValueError("mask_seg_path not exists")

    return cam_infos


def readCamerasFromTransforms_opencv(
    path,
    transformsfile,
    depths_folder,
    white_background,
    is_test,
    extension=".png",
    resize=None,
    language_features_name=None,
):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"] if "fl_x" in contents else contents["fx"]
        focal_len_y = contents["fl_y"] if "fl_y" in contents else contents["fy"]

        cx = contents["cx"]
        cy = contents["cy"]
        # if "crop_edge" in contents:
        #     cx -= contents["crop_edge"]
        #     cy -= contents["crop_edge"]
        if "w" in contents and "h" in contents:
            # scannetpp case, fx, fy, cx, cy in scannetpp json are for 1752*1168, not our target size
            width, height = contents["w"], contents["h"]
        if "width" in contents and "height" in contents:
            # nerfstudio case, fx, fy, cx, cy in nerfstudio json are already for image size 640x480
            width, height = contents["width"], contents["height"]

        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
        #     if "crop_edge" in contents:
        #         width -= 2*contents["crop_edge"]
        #         height -= 2*contents["crop_edge"]
        # else:
        #     # if not specify, we assume the weight and height are twice the cx and cy
        # width, height = cx * 2, cy * 2
        fovx = focal2fov(focal_len_x, width)
        fovy = focal2fov(focal_len_y, height)

        FovY = fovy
        FovX = fovx
        frames = contents["frames"]
        # sort frames by frame["file_path"]
        frames = sorted(frames, key=lambda x: x["file_path"])
        # take frames by some interval
        # frames = frames[::2]
        feat_find_counts = 0
        for idx, frame in enumerate(frames):
            image_path = (
                path.replace("nerfstudio", "undistorted_images")
                if "nerfstudio" in path
                else path
            )  # not used for feature extraction
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, f"{cam_name + extension}")
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            if "resize" in contents:
                resize = contents["resize"]
            else:
                resize = [height, width]

            if resize is not None:
                # resize = [584, 876]
                resize_img = (
                    (resize[1], resize[0])
                    if resize[1] > resize[0]
                    else (resize[0], resize[1])
                )
                image = image.resize(resize_img, Image.LANCZOS)

            # if "crop_edge" in contents:
            #     image = image.crop(
            #         (
            #             contents["crop_edge"],
            #             contents["crop_edge"],
            #             image.width - contents["crop_edge"],
            #             image.height - contents["crop_edge"],
            #         )
            #     )

            if language_features_name is not None:
                mask_seg_path = os.path.join(
                    path, language_features_name, image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, language_features_name, image_name + "_f.npy"
                )
            else:
                mask_seg_path = os.path.join(
                    path, "language_features_dim3", image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, "language_features_dim3", image_name + "_f.npy"
                )

            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue
            else:
                feat_find_counts += 1

                # resize_ratio_x = resize[1] / width
                # resize_ratio_y = resize[0] / height
                # assert resize_ratio_x == resize_ratio_y, "resize ratio x and y should be the same"
                # fx_resize = focal_len_x * resize_ratio_x
                # fy_resize = focal_len_y * resize_ratio_y
                # cx_resize = cx * resize_ratio_x
                # cy_resize = cy * resize_ratio_y
                # fovx = focal2fov(fx_resize, resize[1])
                # fovy = focal2fov(fy_resize, resize[0])

            # print("width, height: ", width, height)
            # print("focal_len_x, focal_len_y: ", focal_len_x, focal_len_y)

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    image=image,
                    cx=cx,
                    cy=cy,
                    fx=focal_len_x,
                    fy=focal_len_y,
                )
            )
    if feat_find_counts == 0:
        raise ValueError("mask_seg_path not exists")

    return cam_infos


def readCamerasFromTransforms_nerfstudio(
    path,
    transformsfile,
    depths_folder,
    white_background,
    is_test,
    extension=".png",
    language_features_name=None,
):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"]
        focal_len_y = contents["fl_y"]
        cx = contents["cx"] * 2
        cy = contents["cy"] * 2
        fovx = focal2fov(focal_len_x, cx)
        fovy = focal2fov(focal_len_y, cy)

        FovY = fovy
        FovX = fovx
        frames = contents["frames"]
        mask_seg_path_find_count = 0
        # raise ValueError("Frames: ", frames)
        for idx, frame in enumerate(frames):
            image_path = os.path.join(
                path, "dslr", "undistorted_images"
            )  # path.replace("nerfstudio", "undistorted_images")
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
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem

            image = Image.open(image_path)
            # resize
            resize = [584, 876]
            resize_img = (
                (resize[1], resize[0])
                if resize[1] > resize[0]
                else (resize[0], resize[1])
            )

            if language_features_name is not None:
                mask_seg_path = os.path.join(
                    path, "dslr", language_features_name, image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, "dslr", language_features_name, image_name + "_f.npy"
                )
            else:
                mask_seg_path = os.path.join(
                    path, "dslr", "language_features_dim3", image_name + "_s.npy"
                )
                mask_feat_path = os.path.join(
                    path, "dslr", "language_features_dim3", image_name + "_f.npy"
                )
            if not os.path.exists(mask_seg_path) or not os.path.exists(mask_feat_path):
                print("mask_seg_path not exists", mask_seg_path)
                print("mask_feat_path not exists", mask_feat_path)
                continue
            else:
                mask_seg_path_find_count += 1

                # raise ValueError("mask_seg_path not exists", mask_seg_path)

            image = image.resize(resize_img, Image.LANCZOS)

            depth_path = (
                os.path.join(depths_folder, f"{image_name}.png")
                if depths_folder != ""
                else ""
            )
            # , depth_path=depth_path
            # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
            #                 image_path=image_path, image_name=image_name,
            #                 width=cx, height=cy, depth_path=depth_path, depth_params=None, is_test=is_test))
            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=cx,
                    height=cy,
                )
            )
    if mask_seg_path_find_count == 0:
        raise ValueError("mask_seg_path not exists")
    return cam_infos


def readScanNetppInfo(
    path,
    white_background,
    depths,
    eval,
    transformsfile,
    llff_hold=8,
    extension=".JPG",
    language_features_name=None,
):

    depths_folder = ""
    print("Reading Training Transforms")

    # raise ValueError("language_features_name: ", language_features_name)
    all_cam_infos = readCamerasFromTransforms_nerfstudio(
        path,
        transformsfile,
        depths_folder,
        white_background,
        False,
        extension,
        language_features_name=language_features_name,
    )

    train_cam_infos = all_cam_infos
    test_cam_infos = []

    # if not eval:
    #     train_cam_infos = all_cam_infos
    #     test_cam_infos = []
    # else:
    #     train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]

    # train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
    # test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]

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
        #    is_nerf_synthetic=False
    )
    return scene_info


def readHolicityInfo(
    path,
    white_background,
    depths,
    eval,
    transformsfile,
    llff_hold=8,
    extension=".jpg",
    language_features_name=None,
):
    depths_folder = ""
    print("Reading Training Transforms")
    all_cam_infos = readCamerasFromTransforms_opencv(
        path,
        transformsfile,
        depths_folder,
        white_background,
        False,
        extension="",
        language_features_name=language_features_name,
    )

    train_cam_infos = all_cam_infos
    test_cam_infos = []

    # if eval == False:
    #     train_cam_infos = all_cam_infos
    #     test_cam_infos = []
    # else:
    #     train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]
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
        #    is_nerf_synthetic=False
    )
    return scene_info


def readScanNetInfo(
    path,
    white_background,
    depths,
    eval,
    transformsfile,
    llff_hold=8,
    extension=".jpg",
    language_features_name=None,
):
    depths_folder = ""
    print("Reading Training Transforms")
    all_cam_infos = readCamerasFromTransforms_scannet(
        path,
        transformsfile,
        depths_folder,
        white_background,
        False,
        extension,
        language_features_name=language_features_name,
    )
    # if eval == False:
    #     train_cam_infos = all_cam_infos
    #     test_cam_infos = []
    # else:
    #     train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]
    train_cam_infos = all_cam_infos
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
        #    is_nerf_synthetic=False
    )
    return scene_info


def readMatterportInfo(
    path,
    white_background,
    depths,
    eval,
    transformsfile,
    llff_hold=8,
    extension=".jpg",
    language_features_name=None,
):
    depths_folder = ""
    print("Reading Training Transforms")
    all_cam_infos = readCamerasFromTransforms_matterport(
        path,
        transformsfile,
        depths_folder,
        white_background,
        False,
        extension="",
        language_features_name=language_features_name,
    )

    train_cam_infos = all_cam_infos
    test_cam_infos = all_cam_infos

    # train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
    # test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]
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
    # "DL3DV": readDL3DVSyntheticInfo,
    # "SCANNETPP": readSCANNETPPSyntheticInfo
    "ScanNetpp": readScanNetppInfo,
    "ScanNet": readScanNetInfo,
    "Matterport3D": readMatterportInfo,
    "Holicity": readHolicityInfo,
}
