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
    prune_by_gradients_matterport,
    test_proper_pruning_matterport,
)


def create_feature_field_lseg(
    splats, batch_size=1, use_cpu=False, inverse_extrinsics=True, resize=[512, 640]
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

        frames = contents["frames"]
        width = contents["width"]
        height = contents["height"]

        for idx, frame in tqdm(
            enumerate(frames), desc="Feature backprojection (frames)", total=len(frames)
        ):
            fx = frame["fx"]
            fy = frame["fy"]
            cx = frame["cx"]
            cy = frame["cy"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            # some dataset save world-to-camera, some camera-to-world, careful!
            w2c = np.linalg.inv(c2w)
            # w2c[1:3] *= -1
            R = w2c[:3, :3]  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            viewmat = torch.eye(4).float()  # .to(device)
            viewmat[:3, :3] = torch.tensor(R).float()  # .to(device)
            viewmat[:3, 3] = torch.tensor(T).float()  # .to(device)

            resize_ratio = resize[1] / 640
            fx_resize = fx * resize_ratio

            fy_resize = fy * resize_ratio
            cx_resize = cx * resize_ratio
            cy_resize = cy * resize_ratio

            K = torch.tensor(
                [
                    [fx_resize, 0, cx_resize],
                    [0, fy_resize, cy_resize],
                    [0, 0, 1],
                ]
            ).float()

            # width = int(K[0, 2] * 2)
            # height = int(K[1, 2] * 2)
            width = resize[1]
            height = resize[0]

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
    data_root_path: str = "/matterport3d_region_mini_test_set_suite/original_data/",
    ply_root_path: str = "/matterport3d_region_mini_test_set_suite/mcmc_3dgs/",  # checkpoint path, can generate from original 3DGS repo
    results_root_dir: str = "./results/matterport/",  # output path
    rasterizer: Literal["inria", "gsplat"] = "gsplat",
    feature_field_batch_count: int = 1,  # Number of batches to process for feature field
    run_feature_field_on_cpu: bool = False,  # Run feature field on CPU
    feature: Literal["lseg", "dino"] = "lseg",  # Feature field type
    scene_i: str = "2t7WUuJeko7_02",  # Scene name
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
    splats = load_ply(ply_path_i, data_dir, rasterizer=rasterizer, dataset="matterport")
    splats_optimized = prune_by_gradients_matterport(splats)

    test_proper_pruning_matterport(splats, splats_optimized)

    splats = splats_optimized
    if feature == "lseg":
        if rescale == 0:
            feat_save_path = f"{result_i_dir}/features_lseg_512_640.pt"
            if not os.path.exists(feat_save_path):
                features = create_feature_field_lseg(
                    splats,
                    feature_field_batch_count,
                    run_feature_field_on_cpu,
                    resize=[512, 640],
                )
                torch.save(features, feat_save_path)
            else:
                print(f"Feature file already exists: {feat_save_path}")
            xyz_save_path = f"{result_i_dir}/xyz_lseg_512_640.npy"
            if not os.path.exists(xyz_save_path):
                xyz = splats["means"].cpu().numpy()
                np.save(xyz_save_path, xyz)
            else:
                print(f"XYZ file already exists: {xyz_save_path}")

        if rescale == 1:
            feat_save_path = f"{result_i_dir}/features_lseg_480_640.pt"
            if not os.path.exists(feat_save_path):
                features = create_feature_field_lseg(
                    splats,
                    feature_field_batch_count,
                    run_feature_field_on_cpu,
                    resize=[480, 640],
                )
                torch.save(features, feat_save_path)
            else:
                print(f"Feature file already exists: {feat_save_path}")
            xyz_save_path = f"{result_i_dir}/xyz_lseg_480_640.npy"
            if not os.path.exists(xyz_save_path):
                xyz = splats["means"].cpu().numpy()
                np.save(xyz_save_path, xyz)
            else:
                print(f"XYZ file already exists: {xyz_save_path}")

        try:
            del splats
            del features
            torch.cuda.empty_cache()
        except:
            print("Error in LSeg feature extraction")


import argparse


def get_arguments():
    argparser = argparse.ArgumentParser(description="Feature Field Extraction")
    argparser.add_argument(
        "--data_root_path",
        type=str,
        default="/matterport3d_region_mini_test_set_suite/original_data",
        help="Path to the dataset",
    )
    argparser.add_argument(
        "--ply_root_path",
        type=str,
        default="/matterport3d_region_mini_test_set_suite/mcmc_3dgs",
        help="Path to the ply files",
    )
    argparser.add_argument(
        "--results_root_dir",
        type=str,
        default="./results/matterport/",
        help="Path to the results directory",
    )
    argparser.add_argument(
        "--scene_name",
        type=str,
        default="2t7WUuJeko7_02",
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
