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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.opengs_utlis import get_SAM_mask_and_feat, load_code_book

# Randomly initialize 300 colors for visualizing the SAM mask. [OpenGaussian]
np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0])  # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path1 = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_ins_feat1"
    )
    render_ins_feat_path2 = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_ins_feat2"
    )
    gt_sam_mask_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_sam_mask"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path1, exist_ok=True)
    makedirs(render_ins_feat_path2, exist_ok=True)
    makedirs(gt_sam_mask_path, exist_ok=True)

    # load codebook
    root_code_book_path = os.path.join(
        model_path, "point_cloud", f"iteration_{iteration}", "root_code_book"
    )
    leaf_code_book_path = os.path.join(
        model_path, "point_cloud", f"iteration_{iteration}", "leaf_code_book"
    )
    if os.path.exists(os.path.join(root_code_book_path, "kmeans_inds.bin")):
        root_code_book, root_cluster_indices = load_code_book(root_code_book_path)
        root_cluster_indices = torch.from_numpy(root_cluster_indices).cuda()
    if os.path.exists(os.path.join(leaf_code_book_path, "kmeans_inds.bin")):
        leaf_code_book, leaf_cluster_indices = load_code_book(leaf_code_book_path)
        leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()
    else:
        leaf_cluster_indices = None

    # render
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(
            view, gaussians, pipeline, background, iteration, rescale=False
        )

        # RGB
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]

        # ins_feat
        rendered_ins_feat = render_pkg["ins_feat"]
        gt_sam_mask = view.original_sam_mask.cuda()  # [4, H, W]

        # Rendered RGB
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, view.image_name + ".png")
        )
        # GT RGB
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, view.image_name + ".png")
        )

        # ins_feat
        torchvision.utils.save_image(
            rendered_ins_feat[:3, :, :],
            os.path.join(render_ins_feat_path1, view.image_name + "_1.png"),
        )
        torchvision.utils.save_image(
            rendered_ins_feat[3:6, :, :],
            os.path.join(render_ins_feat_path2, view.image_name + "_2.png"),
        )

        # NOTE get SAM id, mask bool, mask_feat, invalid pix
        mask_id, _, _, _ = get_SAM_mask_and_feat(
            gt_sam_mask, level=0, original_mask_feat=view.original_mask_feat
        )
        # mask visualization
        mask_color_rand = colors_defined[mask_id.detach().cpu().type(torch.int64)].type(
            torch.float64
        )
        mask_color_rand = mask_color_rand.permute(2, 0, 1)
        torchvision.utils.save_image(
            mask_color_rand / 255.0,
            os.path.join(gt_sam_mask_path, view.image_name + ".png"),
        )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )
