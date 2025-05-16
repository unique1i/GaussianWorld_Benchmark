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
import torch.nn.functional as F
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
import json
from utils.opengs_utlis import get_SAM_mask_and_feat, load_code_book

np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0])  # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)


def render_set(
    model_path, name, iteration, views, gaussians, pipeline, background, scene_name
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "renders_ins_feat"
    )
    gt_sam_mask_path = os.path.join(
        model_path, name, "ours_{}".format(iteration), "gt_sam_mask"
    )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path, exist_ok=True)
    makedirs(gt_sam_mask_path, exist_ok=True)

    # load codebook
    root_code_book, root_cluster_indices = load_code_book(
        os.path.join(
            model_path, "point_cloud", f"iteration_{iteration}", "root_code_book"
        )
    )
    leaf_code_book, leaf_cluster_indices = load_code_book(
        os.path.join(
            model_path, "point_cloud", f"iteration_{iteration}", "leaf_code_book"
        )
    )
    root_cluster_indices = torch.from_numpy(root_cluster_indices).cuda()
    leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()
    # counts = torch.bincount(torch.from_numpy(cluster_indices), minlength=64)

    # load the saved codebook(leaf id) and instance-level language feature
    # 'leaf_feat', 'leaf_acore', 'occu_count', 'leaf_ind'
    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)
    leaf_lang_feat = torch.from_numpy(
        saved_data["leaf_feat.npy"]
    ).cuda()  # [num_leaf=k1*k2, 512] cluster lang feat
    leaf_score = torch.from_numpy(
        saved_data["leaf_score.npy"]
    ).cuda()  # [num_leaf=k1*k2] cluster score
    leaf_occu_count = torch.from_numpy(
        saved_data["occu_count.npy"]
    ).cuda()  # [num_leaf=k1*k2]
    leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()  # [num_pts] fine id
    leaf_lang_feat[
        leaf_occu_count < 5
    ] *= 0.0  # Filter out clusters that occur too infrequently.
    leaf_cluster_indices = leaf_ind

    root_num = root_cluster_indices.max() + 1
    leaf_num = leaf_lang_feat.shape[0] / root_num

    # text feature
    with open("assets/text_features.json", "r") as f:
        data_loaded = json.load(f)
    all_texts = list(data_loaded.keys())
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(
        torch.float32
    )  # [num_text, 512]

    scene_texts = {
        "waldo_kitchen": [
            "Stainless steel pots",
            "dark cup",
            "refrigerator",
            "frog cup",
            "pot",
            "spatula",
            "plate",
            "spoon",
            "toaster",
            "ottolenghi",
            "plastic ladle",
            "sink",
            "ketchup",
            "cabinet",
            "red cup",
            "pour-over vessel",
            "knife",
            "yellow desk",
        ],
        "ramen": [
            "nori",
            "sake cup",
            "kamaboko",
            "corn",
            "spoon",
            "egg",
            "onion segments",
            "plate",
            "napkin",
            "bowl",
            "glass of water",
            "hand",
            "chopsticks",
            "wavy noodles",
        ],
        "figurines": [
            "jake",
            "pirate hat",
            "pikachu",
            "rubber duck with hat",
            "porcelain hand",
            "red apple",
            "tesla door handle",
            "waldo",
            "bag",
            "toy cat statue",
            "miffy",
            "green apple",
            "pumpkin",
            "rubics cube",
            "old camera",
            "rubber duck with buoy",
            "red toy chair",
            "pink ice cream",
            "spatula",
            "green toy chair",
            "toy elephant",
        ],
        "teatime": [
            "sheep",
            "yellow pouf",
            "stuffed bear",
            "coffee mug",
            "tea in a glass",
            "apple",
            "coffee",
            "hooves",
            "bear nose",
            "dall-e brand",
            "plate",
            "paper napkin",
            "three cookies",
            "bag of cookies",
        ],
    }
    # note: query text
    target_text = scene_texts[scene_name]

    query_text_feats = torch.zeros(len(target_text), 512).cuda()
    for i, text in enumerate(target_text):
        feat = text_features[all_texts.index(text)].unsqueeze(0)
        query_text_feats[i] = feat

    for t_i, text_feat in enumerate(query_text_feats):
        # if target_text[t_i] != "old camera":
        #     continue

        print(
            f"rendering the {t_i + 1}-th query of {len(target_text)} texts: {target_text[t_i]}"
        )
        # compute cosine similarity
        text_feat = F.normalize(text_feat.unsqueeze(0), dim=1, p=2)
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
        cosine_similarity = torch.matmul(text_feat, leaf_lang_feat.transpose(0, 1))
        max_id = torch.argmax(cosine_similarity, dim=-1)  # [cluster_num]
        text_leaf_indices = max_id

        top_values, top_indices = torch.topk(cosine_similarity, 10)
        for candidate_id in top_indices[0][1:]:
            if candidate_id - max_id < leaf_num:  # TODO !!!!!!
                max_feat = leaf_code_book["ins_feat"][max_id]
                candi_feat = leaf_code_book["ins_feat"][candidate_id]
                distances = torch.norm(max_feat - candi_feat, dim=1)
                if distances < 0.9:
                    text_leaf_indices = torch.cat(
                        [text_leaf_indices, candidate_id.unsqueeze(0)]
                    )

        # render
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            # note: evaluation frame
            scene_gt_frames = {
                "waldo_kitchen": [
                    "frame_00053",
                    "frame_00066",
                    "frame_00089",
                    "frame_00140",
                    "frame_00154",
                ],
                "ramen": [
                    "frame_00006",
                    "frame_00024",
                    "frame_00060",
                    "frame_00065",
                    "frame_00081",
                    "frame_00119",
                    "frame_00128",
                ],
                "figurines": [
                    "frame_00041",
                    "frame_00105",
                    "frame_00152",
                    "frame_00195",
                ],
                "teatime": [
                    "frame_00002",
                    "frame_00025",
                    "frame_00043",
                    "frame_00107",
                    "frame_00129",
                    "frame_00140",
                ],
            }
            candidate_frames = scene_gt_frames[scene_name]

            if view.image_name not in candidate_frames:
                continue

            render_pkg = render(
                view, gaussians, pipeline, background, iteration, rescale=False
            )
            # RGB
            rendering = render_pkg["render"]
            gt = view.original_image[0:3, :, :]

            # ins_feat
            rendered_ins_feat = render_pkg["ins_feat"]
            gt_sam_mask = view.original_sam_mask.cuda()  # [4, H, W]

            # RGB
            torchvision.utils.save_image(
                rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            )
            torchvision.utils.save_image(
                gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
            )

            # ins_feat
            torchvision.utils.save_image(
                rendered_ins_feat[:3, :, :],
                os.path.join(render_ins_feat_path, "{0:05d}".format(idx) + "_1.png"),
            )
            torchvision.utils.save_image(
                rendered_ins_feat[3:6, :, :],
                os.path.join(render_ins_feat_path, "{0:05d}".format(idx) + "_2.png"),
            )

            # NOTE get SAM id, mask bool, mask_feat, invalid pix
            mask_id, mask_bool, mask_feat, invalid_pix = get_SAM_mask_and_feat(
                gt_sam_mask, level=3, original_mask_feat=view.original_mask_feat
            )

            # sam mask
            mask_color_rand = colors_defined[
                mask_id.detach().cpu().type(torch.int64)
            ].type(torch.float64)
            mask_color_rand = mask_color_rand.permute(2, 0, 1)
            torchvision.utils.save_image(
                mask_color_rand / 255.0,
                os.path.join(gt_sam_mask_path, "{0:05d}".format(idx) + ".png"),
            )

            # render target object
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                iteration,
                rescale=False,  # )  # wherther to re-scale the gaussian scale
                # cluster_idx=leaf_cluster_indices,     # root id
                leaf_cluster_idx=leaf_cluster_indices,  # leaf id
                selected_leaf_id=text_leaf_indices.cuda(),  # text query 所选择的 leaf id
                render_feat_map=False,
                render_cluster=False,
                better_vis=True,
                seg_rgb=True,
                post_process=True,
                root_num=root_num,
                leaf_num=leaf_num,
            )
            rendered_cluster_imgs = render_pkg["leaf_clusters_imgs"]
            occured_leaf_id = render_pkg["occured_leaf_id"]
            rendered_leaf_cluster_silhouettes = render_pkg["leaf_cluster_silhouettes"]

            render_cluster_path = os.path.join(
                model_path, name, "ours_{}".format(iteration), "renders_cluster"
            )
            render_cluster_silhouette_path = os.path.join(
                model_path,
                name,
                "ours_{}".format(iteration),
                "renders_cluster_silhouette",
            )
            makedirs(render_cluster_path, exist_ok=True)
            makedirs(render_cluster_silhouette_path, exist_ok=True)
            for i, img in enumerate(rendered_cluster_imgs):
                # save object RGB
                torchvision.utils.save_image(
                    img[:3, :, :],
                    os.path.join(
                        render_cluster_path,
                        view.image_name + f"_{target_text[t_i]}.png",
                    ),
                )
                # save object mask
                cluster_silhouette = rendered_leaf_cluster_silhouettes[i] > 0.7
                torchvision.utils.save_image(
                    cluster_silhouette.to(torch.float32),
                    os.path.join(
                        render_cluster_silhouette_path,
                        view.image_name + f"_{target_text[t_i]}.png",
                    ),
                )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    scene_name: str,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "text2obj",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                scene_name,
            )
        if not skip_test:
            render_set(
                dataset.model_path,
                "text2obj",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                scene_name,
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
    parser.add_argument(
        "--scene_name",
        type=str,
        choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
        help="Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen",
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not args.scene_name:
        parser.error(
            "The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime"
        )

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.scene_name,
    )
