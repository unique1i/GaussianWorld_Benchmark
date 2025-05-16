import os
import time
from typing import Literal
import torch
from gsplat import rasterization

# import pycolmap_scene_manager as pycolmap
import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # To avoid conflict with cv2
from tqdm import tqdm
from lseg import LSegNet
import json

from utils import (
    load_ply,
    get_viewmat_from_colmap_image,
    prune_by_gradients_json,
    test_proper_pruning_json,
)


def create_feature_field_lseg(
    splats, batch_size=1, use_cpu=False, inverse_extrinsics=True, resize=[240, 320]
):
    device = "cpu" if use_cpu else "cuda"

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(
        torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device)
    )
    net.eval()
    net.to(device)

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colors.to(device)
    colors_0.to(device)

    # colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    # K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    gaussian_features = torch.zeros(colors.shape[0], 512, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    t1 = time.time()

    colors_feats = torch.zeros(
        colors.shape[0], 512, device=colors.device, requires_grad=True
    )
    colors_feats_0 = torch.zeros(
        colors.shape[0], 3, device=colors.device, requires_grad=True
    )
    transformsfile = splats["transform_json_file"]

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
        elif "resize" in contents:
            # scannet case, fx, fy, cx, cy in scannet json are already for image size 640x480
            width, height = contents["resize"]
            if "crop_edge" in contents:
                width -= 2 * contents["crop_edge"]
                height -= 2 * contents["crop_edge"]
        else:
            # if not specify, we assume the weight and height are twice the cx and cy
            width, height = cx * 2, cy * 2
        frames = contents["frames"]
        for idx, frame in tqdm(
            enumerate(frames), desc="Feature backprojection (frames)", total=len(frames)
        ):
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
            if (
                inverse_extrinsics
            ):  # some dataset save world-to-camera, some camera-to-world, careful!
                w2c = np.linalg.inv(c2w)
            else:
                w2c = c2w

            w2c[1:3] *= -1
            R = w2c[
                :3, :3
            ]  # np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            # image_path = os.path.join(image_path, f'{cam_name + extension}')
            # image_name = Path(cam_name).stem
            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)
            resize_ratio = resize[1] / 1752
            fx_resize = focal_len_x * resize_ratio

            fy_resize = focal_len_y * resize_ratio
            cx_resize = cx * resize_ratio
            cy_resize = cy * resize_ratio

            K = torch.tensor(
                [
                    [fx_resize, 0, cx_resize],
                    [0, fy_resize, cy_resize],
                    [0, 0, 1],
                ]
            ).float()

            # images = sorted(colmap_project.images.values(), key=lambda x: x.name)
            # batch_size = math.ceil(len(images) / batch_count) if batch_count > 0 else 1

            # for batch_start in tqdm(
            #     range(0, len(images), batch_size),
            #     desc="Feature backprojection (batches)",
            # ):
            #     batch = images[batch_start : batch_start + batch_size]
            #     for image in batch:
            #         viewmat = get_viewmat_from_colmap_image(image)
            width = int(K[0, 2] * 2)
            height = int(K[1, 2] * 2)

            with torch.no_grad():
                output, _, meta = rasterization(
                    means,
                    quats,
                    scales,
                    opacities,
                    colors_all,
                    viewmat[None],
                    K[None],
                    width=width,
                    height=height,
                    sh_degree=3,
                )

                output = torch.nn.functional.interpolate(
                    output.permute(0, 3, 1, 2).to(device),
                    size=(480, 480),
                    mode="bilinear",
                )
                output.to(device)
                feats = net.forward(output)
                feats = torch.nn.functional.normalize(feats, dim=1)
                feats = torch.nn.functional.interpolate(
                    feats, size=(height, width), mode="bilinear"
                )[0]
                feats = feats.permute(1, 2, 0)

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )

            target = (output_for_grad[0].to(device) * feats).sum()
            target.to(device)
            target.backward()
            colors_feats_copy = colors_feats.grad.clone()
            colors_feats.grad.zero_()

            output_for_grad, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_feats_0,
                viewmat[None],
                K[None],
                width=width,
                height=height,
            )

            target_0 = (output_for_grad[0]).sum()
            target_0.to(device)
            target_0.backward()

            gaussian_features += colors_feats_copy
            gaussian_denoms += colors_feats_0.grad[:, 0]
            colors_feats_0.grad.zero_()

            # Clean up unused variables and free GPU memory
            del (
                viewmat,
                meta,
                _,
                output,
                feats,
                output_for_grad,
                colors_feats_copy,
                target,
                target_0,
            )
            torch.cuda.empty_cache()
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    gaussian_features[torch.isnan(gaussian_features)] = 0
    t2 = time.time()
    print("Time taken for feature backprojection", t2 - t1)
    return gaussian_features


def create_feature_field_dino(splats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = (
        torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitl14_reg")
        .to(device)
        .eval()
    )

    dinov2_vits14_reg = feature_extractor

    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors_all = torch.cat([colors_dc, colors_rest], dim=1)

    colors = colors_dc[:, 0, :]  # * 0
    colors_0 = colors_dc[:, 0, :] * 0
    colmap_project = splats["colmap_project"]

    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    colors.requires_grad = True
    colors_0.requires_grad = True

    DIM = 1024

    gaussian_features = torch.zeros(colors.shape[0], DIM, device=colors.device)
    gaussian_denoms = torch.ones(colors.shape[0], device=colors.device) * 1e-12

    t1 = time.time()

    colors_feats = torch.zeros(colors.shape[0], 1024, device=colors.device)
    colors_feats.requires_grad = True
    colors_feats_0 = torch.zeros(colors.shape[0], 3, device=colors.device)
    colors_feats_0.requires_grad = True

    print("Distilling features...")
    for image in tqdm(sorted(colmap_project.images.values(), key=lambda x: x.name)):
        image_name = image.name  # .split(".")[0] + ".jpg"

        viewmat = get_viewmat_from_colmap_image(image)

        width = int(K[0, 2] * 2)
        height = int(K[1, 2] * 2)
        with torch.no_grad():
            output, _, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors_all,
                viewmat[None],
                K[None],
                width=width,
                height=height,
                sh_degree=3,
            )

            output = torch.nn.functional.interpolate(
                output.permute(0, 3, 1, 2).cuda(),
                size=(224 * 4, 224 * 4),
                mode="bilinear",
                align_corners=False,
            )
            feats = dinov2_vits14_reg.forward_features(output)["x_norm_patchtokens"]
            feats = feats[0].reshape((16 * 4, 16 * 4, DIM))
            feats = torch.nn.functional.interpolate(
                feats.unsqueeze(0).permute(0, 3, 1, 2),
                size=(height, width),
                mode="nearest",
            )[0]
            feats = feats.permute(1, 2, 0)

        output_for_grad, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors_feats,
            viewmat[None],
            K[None],
            width=width,
            height=height,
        )

        target = (output_for_grad[0] * feats).mean()

        target.backward()

        colors_feats_copy = colors_feats.grad.clone()

        colors_feats.grad.zero_()

        output_for_grad, _, meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors_feats_0,
            viewmat[None],
            K[None],
            width=width,
            height=height,
        )

        target_0 = (output_for_grad[0]).mean()

        target_0.backward()

        gaussian_features += colors_feats_copy  # / (colors_feats_0.grad[:,0:1]+1e-12)
        gaussian_denoms += colors_feats_0.grad[:, 0]
        colors_feats_0.grad.zero_()
    print(gaussian_features.shape, gaussian_denoms.shape)
    gaussian_features = gaussian_features / gaussian_denoms[..., None]
    gaussian_features = gaussian_features / gaussian_features.norm(dim=-1, keepdim=True)
    # Replace nan values with 0
    print("NaN features", torch.isnan(gaussian_features).sum())
    gaussian_features[torch.isnan(gaussian_features)] = 0
    t2 = time.time()
    print("Time taken for feature distillation", t2 - t1)
    return gaussian_features


def main(
    data_root_path: str = "/scannetpp_mini_val_set_suite/original_data",  # subset
    ply_root_path: str = "/scannetpp_mini_val_set_suite/mcmc_3dgs/",  # checkpoint path, can generate from original 3DGS repo
    results_root_dir: str = "./results/scannetpp/",  # output path
    rasterizer: Literal["inria", "gsplat"] = "gsplat",
    feature_field_batch_count: int = 1,  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False,  # Run feature field on CPU
    feature: Literal["lseg", "dino"] = "lseg",  # Feature field type
    scene_i: str = "09c1414f1b",  # Scene index for processing
    rescale: int = 0,  # Rescale factor for images
):
    print("Processing scene:", scene_i)
    result_i_dir = os.path.join(results_root_dir, scene_i)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(result_i_dir, exist_ok=True)
    ply_path_i = os.path.join(ply_root_path, scene_i, "ckpts", "point_cloud_30000.ply")
    data_dir = os.path.join(data_root_path, scene_i)
    splats = load_ply(ply_path_i, data_dir, rasterizer=rasterizer, dataset="scannetpp")
    splats_optimized = prune_by_gradients_json(splats)

    test_proper_pruning_json(splats, splats_optimized)

    splats = splats_optimized
    if feature == "lseg":
        if rescale == 0:
            features_save_dir = os.path.join(result_i_dir, "features_lseg_584_876.pt")
            if not os.path.exists(features_save_dir):
                features = create_feature_field_lseg(
                    splats,
                    feature_field_batch_count,
                    run_feature_field_on_cpu,
                    resize=[584, 876],
                )
                torch.save(features, features_save_dir)
            else:
                # features = torch.load(features_save_dir)
                print("Feature file already exists, skipping feature extraction.")
            xyz_save_dir = os.path.join(result_i_dir, "xyz_lseg_584_876.npy")

            if not os.path.exists(xyz_save_dir):
                xyz = splats["means"].cpu().numpy()
                xyz = xyz.reshape(-1, 3)
                np.save(xyz_save_dir, xyz)
            else:
                xyz = np.load(xyz_save_dir)
                print("XYZ file already exists, skipping XYZ extraction.")

        if rescale == 1:
            features_save_dir = os.path.join(result_i_dir, "features_lseg_480_640.pt")
            if not os.path.exists(features_save_dir):
                features = create_feature_field_lseg(
                    splats,
                    feature_field_batch_count,
                    run_feature_field_on_cpu,
                    resize=[480, 640],
                )
                torch.save(features, features_save_dir)
            xyz_save_dir = os.path.join(result_i_dir, "xyz_lseg_480_640.npy")
            if not os.path.exists(xyz_save_dir):
                xyz = splats["means"].cpu().numpy()
                xyz = xyz.reshape(-1, 3)
                np.save(xyz_save_dir, xyz)

        if rescale == 2:
            features = create_feature_field_lseg(
                splats,
                feature_field_batch_count,
                run_feature_field_on_cpu,
                resize=[240, 320],
            )
            torch.save(features, f"{result_i_dir}/features_lseg_240_320.pt")

        try:
            del splats
            del features
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error in LSeg feature extraction: {e}")


import argparse


def get_arguments():
    argparser = argparse.ArgumentParser(description="Feature Field Extraction")
    argparser.add_argument(
        "--data_root_path",
        type=str,
        default="/scannetpp_mini_val_set_suite/original_data",
        help="Path to the dataset",
    )
    argparser.add_argument(
        "--ply_root_path",
        type=str,
        default="/scannetpp_mini_val_set_suite/mcmc_3dgs",
        help="Path to the ply files",
    )
    argparser.add_argument(
        "--results_root_dir",
        type=str,
        default="./results/scannetpp/",
        help="Path to the results directory",
    )
    argparser.add_argument(
        "--scene_name",
        type=str,
        default="09c1414f1b",
        help="Path to the validation split file",
    )
    argparser.add_argument(
        "--rescale",
        type=int,
        default=0,
        help="Rescale factor for images",
    )

    return argparser.parse_args()


if __name__ == "__main__":
    # tyro.cli(main)
    args = get_arguments()
    main(
        data_root_path=args.data_root_path,
        ply_root_path=args.ply_root_path,
        results_root_dir=args.results_root_dir,
        scene_i=args.scene_name,
        rescale=args.rescale,
    )
