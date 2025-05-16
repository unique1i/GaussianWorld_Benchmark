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
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from os import makedirs
import torchvision
import numpy as np
from utils.sh_utils import RGB2SH

# import faiss
from scene.kmeans_quantize import Quantize_kMeans
from bitarray import bitarray
from utils.system_utils import mkdir_p
from utils.opengs_utlis import (
    mask_feature_mean,
    pair_mask_feature_mean,
    get_SAM_mask_and_feat,
    load_code_book,
    calculate_iou,
    calculate_distances,
    calculate_pairwise_distances,
)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# Randomly initialize 300 colors for visualizing the SAM mask. [OpenGaussian]
np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0])  # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)


def dec2binary(x, n_bits=None):
    """Convert decimal integer x to binary.

    Code from: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    if n_bits is None:
        n_bits = torch.ceil(torch.log2(x)).type(torch.int64)
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)


def save_kmeans(kmeans_list, quantized_params, out_dir, mode="root"):
    """Save the codebook and indices of KMeans."""
    # Convert to bitarray object to save compressed version
    # saving as npy or pth will use 8bits per digit (or boolean) for the indices
    # Convert to binary, concat the indices for all params and save.
    if mode == "root":
        out_dir = os.path.join(out_dir, "root_code_book")
    elif mode == "leaf":
        out_dir = os.path.join(out_dir, "leaf_code_book")

    mkdir_p(out_dir)
    bitarray_all = bitarray([])
    for kmeans in kmeans_list:
        if mode == "root":
            cls_ids = kmeans.cls_ids
        elif mode == "leaf":
            cls_ids = kmeans.leaf_cls_ids
        n_bits = int(np.ceil(np.log2(len(cls_ids))))
        assignments = dec2binary(cls_ids, n_bits)
        bitarr = bitarray(list(assignments.cpu().numpy().flatten()))
        bitarray_all.extend(bitarr)
    with open(os.path.join(out_dir, "kmeans_inds.bin"), "wb") as file:  # cls_ids
        bitarray_all.tofile(file)

    # Save details needed for loading
    args_dict = {}
    args_dict["params"] = quantized_params
    args_dict["n_bits"] = n_bits
    args_dict["total_len"] = len(bitarray_all)
    np.save(os.path.join(out_dir, "kmeans_args.npy"), args_dict)
    if mode == "root":
        centers_dict = {
            param: kmeans.centers
            for (kmeans, param) in zip(kmeans_list, quantized_params)
        }
    elif mode == "leaf":
        centers_dict = {
            param: kmeans.leaf_centers
            for (kmeans, param) in zip(kmeans_list, quantized_params)
        }

    # Save codebook
    torch.save(centers_dict, os.path.join(out_dir, "kmeans_centers.pth"))


def cohesion_loss(feat_map, gt_mask, feat_mean_stack):
    """intra-mask smoothing loss. Eq.(1) in the paper
    Constrain the feature of each pixel within the mask to be close to the mean feature of that mask.
    """
    N, H, W = gt_mask.shape
    C = feat_map.shape[0]
    # expand feat_map [6, H, W] to [N, 6, H, W]
    feat_map_expanded = feat_map.unsqueeze(0).expand(N, C, H, W)
    # expand mean feat [N, 6] to [N, 6, H, W]
    feat_mean_stack_expanded = (
        feat_mean_stack.unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)
    )

    # fature distance
    masked_feat = feat_map_expanded * gt_mask.unsqueeze(1)  # [N, 6, H, W]
    dist = (masked_feat - feat_mean_stack_expanded).norm(p=2, dim=1)  # [N, H, W]

    # per mask feature distance (loss)
    masked_dist = dist * gt_mask  # [N, H, W]
    loss_per_mask = masked_dist.sum(dim=[1, 2]) / gt_mask.sum(dim=[1, 2]).clamp(min=1)

    return loss_per_mask.mean()


def separation_loss(feat_mean_stack, iteration):
    """inter-mask contrastive loss Eq.(2) in the paper
    Constrain the instance features within different masks to be as far apart as possible.
    """
    N, _ = feat_mean_stack.shape

    # expand feat_mean_stack[N, 6] to [N, N, C]
    feat_expanded = feat_mean_stack.unsqueeze(1).expand(-1, N, -1)
    feat_transposed = feat_mean_stack.unsqueeze(0).expand(N, -1, -1)

    # distance
    diff_squared = (feat_expanded - feat_transposed).pow(2).sum(2)

    # Calculate the inverse of the distance to enhance discrimination
    epsilon = 1  # 1e-6
    inverse_distance = 1.0 / (diff_squared + epsilon)
    # Exclude diagonal elements (distance from itself) and calculate the mean inverse distance
    mask = torch.eye(N, device=feat_mean_stack.device).bool()
    inverse_distance.masked_fill_(mask, 0)

    # note: weight
    # sorted by distance
    sorted_indices = inverse_distance.argsort().argsort()
    loss_weight = (sorted_indices.float() / (N - 1)) * (
        1.0 - 0.1
    ) + 0.1  # scale to 0.1 - 1.0, [N, N]
    # small weight
    if iteration > 35_000:
        loss_weight[loss_weight < 0.9] = 0.1
    inverse_distance *= loss_weight  # [N, N]

    # final loss
    loss = inverse_distance.sum() / (N * (N - 1))

    return loss


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    ply_path,
    debug_from,
):
    iterations = [
        opt.start_ins_feat_iter,
        opt.start_leaf_cb_iter,
        opt.start_root_cb_iter,
    ]
    saving_iterations.extend(iterations)
    checkpoint_iterations.extend(iterations)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # ply_path

    if ply_path:
        gaussians.load_ply_gs_only(ply_path)
        first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        # NOTE: Load the original 3DGS pre-trained checkpoint and add the ins_feat attribute. [OpenGaussian]
        if len(model_params) == 12:
            # initialize instance color.
            ins_feat = torch.rand(
                (model_params[8].shape[0], opt.ins_feat_dim),
                dtype=torch.float,
                device="cuda",
            )
            ins_feat = torch.nn.Parameter(ins_feat.requires_grad_(True))
            to_list = list(model_params)
            # (1) replace optimizer
            to_list[10] = gaussians.optimizer.state_dict()
            # (2) add ins_feat
            to_list.insert(7, ins_feat)
            # (3) add ins_feat_q (quantized ins_feat)
            ins_feat_q = torch.empty(0)
            to_list.insert(8, ins_feat_q)
            model_params = tuple(to_list)
        gaussians.restore(model_params, opt)
        ins_feat_continue = gaussians._ins_feat.clone().detach()  # not used
    else:
        ins_feat_continue = None  # not used

    # initialize the codebook
    ins_feat_codebook = Quantize_kMeans(
        num_clusters=opt.root_node_num,  # k1
        num_leaf_clusters=opt.leaf_node_num,  # k2
        num_iters=5,
        dim=9,
    )

    # note: load the saved codebook
    leaf_cluster_indices = None
    if checkpoint:
        base_dir = os.path.dirname(checkpoint)
        load_iter = checkpoint.split("/")[-1].split(".")[0][6:]
        root_code_book_path = os.path.join(
            base_dir, "point_cloud", f"iteration_{load_iter}", "root_code_book"
        )
        leaf_code_book_path = os.path.join(
            base_dir, "point_cloud", f"iteration_{load_iter}", "leaf_code_book"
        )
        if os.path.exists(os.path.join(root_code_book_path, "kmeans_inds.bin")):
            root_center, root_indices = load_code_book(root_code_book_path)
            root_center_saved = root_center["ins_feat"]
            cluster_indices = torch.from_numpy(root_indices).cuda()
            ins_feat_codebook.centers = root_center_saved
            ins_feat_codebook.cls_ids = cluster_indices
        else:
            cluster_indices = None
        if os.path.exists(os.path.join(leaf_code_book_path, "kmeans_inds.bin")):
            leaf_center, leaf_indices = load_code_book(leaf_code_book_path)
            leaf_center_saved = leaf_center["ins_feat"]
            leaf_cluster_indices = torch.from_numpy(leaf_indices).cuda()
            ins_feat_codebook.leaf_centers = leaf_center_saved
            ins_feat_codebook.leaf_cls_ids = leaf_cluster_indices
        else:
            leaf_cluster_indices = None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    root_id = 0  # for stage 2.2
    loss = torch.tensor(0.0)
    Ll1 = torch.tensor(0.0)
    for iteration in range(first_iter, opt.iterations + 1):
        no_need_bk = False

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        iteration,
                        scaling_modifer,
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(
            iteration, opt.start_root_cb_iter, opt.start_leaf_cb_iter
        )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if not viewpoint_cam.data_on_gpu:
            viewpoint_cam.to_gpu()

        cb_mode = None  # Current status: No launch codebook discretization
        if iteration == 1:
            print("[Stage 0] Start 3dgs pre-train ...")
            sys.stdout.flush()
        if iteration == opt.start_ins_feat_iter + 1:
            print("[Stage 1] Start continuous instance feature learning ...")
            sys.stdout.flush()
        # Stage 2.1: Coarse-level codebook
        if iteration > opt.start_root_cb_iter and iteration <= opt.start_leaf_cb_iter:
            cb_mode = "root"
            if iteration == opt.start_root_cb_iter + 1:
                print("[Stage 2.1] Start coarse-level codebook discretization ...")
                sys.stdout.flush()
        elif iteration > opt.start_leaf_cb_iter:
            cb_mode = "leaf"
            # Stage 2.2: Fine-level codebook
            if iteration == opt.start_leaf_cb_iter + 1:
                print("[Stage 2.2] Start fine-level codebook discretization ...")
                sys.stdout.flush()
            # note Update a coarse cluster every leaf_update_fr(default 300) steps.
            if (iteration - opt.start_leaf_cb_iter) % opt.leaf_update_fr == 0:
                root_id += 1  # 0 ~ k1-1
                if root_id > (opt.root_node_num - 1):
                    root_id = 0

        # ###########################################################################
        # [Stage 2]: Two-Level Codebook for Discretization                          #
        #   - Preprocessing: construct pseudo labels (instance features of stage 1) #
        #     Will execute twice, before coarse-level and fine-level clustering     #
        # ###########################################################################
        if (cb_mode is not None and viewpoint_cam.pesudo_ins_feat is None) or (
            (iteration == opt.start_root_cb_iter + 1)
            or (iteration == opt.start_leaf_cb_iter + 1)
        ):
            with torch.no_grad():
                if cb_mode == "leaf" and cluster_indices is None:
                    cluster_indices = (
                        ins_feat_codebook.cls_ids
                    )  # [num_pts], Coarse-level ID of each point (0 ~ k1-1)
                construct_pseudo_ins_feat(
                    scene,
                    render,
                    (pipe, background, iteration),
                    cluster_indices=cluster_indices,
                    mode=cb_mode,
                    root_num=opt.root_node_num,
                    leaf_num=opt.leaf_node_num,
                    sam_level=opt.sam_level,
                    save_memory=opt.save_memory,
                )
                if not viewpoint_cam.data_on_gpu:
                    viewpoint_cam.to_gpu()
                if cb_mode == "leaf":
                    # Number of leaves per root
                    ins_feat_codebook.iLeafSubNum = gaussians.iClusterSubNum

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ####################################################
        # [Stage 2]: Two-Level Codebook for Discretization   #
        #   - Update codebook                                #
        # ####################################################
        freq_k_means = 200  # coarse-level codebook update frequency
        if cb_mode == "leaf":
            freq_k_means = 50  # todo fine-level codebook update frequency
        if cb_mode is not None:
            if (
                iteration % freq_k_means == 1
            ) or iteration == opt.start_root_cb_iter + 1:
                assign = True  # Reassign cluster centers
            else:
                assign = False  #  update cluster centers
            ins_feat_codebook.forward(
                gaussians,
                iteration,
                assign=assign,
                mode=cb_mode,
                selected_leaf=root_id,
                pos_weight=opt.pos_weight,
            )  # note: position weight

        # render function
        if iteration <= opt.start_ins_feat_iter:  # stage 0
            render_feat = False
            render_cluster = False
            cluster_indices = None
        elif iteration > opt.start_leaf_cb_iter:  # stage 2.2 (fine-level)
            render_feat = False
            render_cluster = True
        else:  # stage 1, stage 2.1(coarse-level)
            render_feat = True
            render_cluster = False
            cluster_indices = None
        # rescale
        if iteration > opt.start_root_cb_iter:  # stage 2, rescale
            rescale = True
        else:
            rescale = False

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            iteration,
            rescale=rescale,  # wherther to re-scale the gaussian scale
            cluster_idx=cluster_indices,  # coarse-level cluster id
            leaf_cluster_idx=ins_feat_codebook.leaf_cls_ids,  # fine-level cluster id
            render_feat_map=render_feat,
            render_cluster=render_cluster,
            selected_root_id=root_id,
        )  # coarse id (stage 2.2)
        # rendered results
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        alpha = render_pkg["alpha"]
        rendered_silhouette = (
            render_pkg["silhouette"] if render_pkg["silhouette"] is not None else alpha
        )
        rendered_silhouette = (rendered_silhouette > 0.7) * 1.0  # mask after re-scale
        rendered_ins_feat = render_pkg["ins_feat"]
        rendered_cluster_imgs = render_pkg["cluster_imgs"]  # [num_cl, 6, H, W]
        rendered_leaf_cluster_imgs = render_pkg["leaf_clusters_imgs"]
        rendered_cluster_silhouettes = render_pkg["cluster_silhouettes"]
        if render_cluster:
            if (
                rendered_cluster_silhouettes is not None
                and len(rendered_cluster_silhouettes) > 0
            ):
                rendered_cluster_silhouettes = rendered_cluster_silhouettes > 0.7
            else:
                # root_id-th coarse cluster not visible in current view
                no_need_bk = True

        # gt supervision: rgb image & SAM mask
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.original_sam_mask is not None:
            gt_sam_mask = viewpoint_cam.original_sam_mask.cuda()  # [4, H, W]

        # if iteration % 5000 == 0:
        #     source_name = os.path.basename(dataset.source_path)
        #     # print("gt_image", gt_image.shape, gt_image.min(), gt_image.max())
        #     # print("image", image.shape, image.min(), image.max())
        #     # print("Ll1", Ll1.item())
        #     ### save gt image for debug
        #     gt_image_save = torch.clamp(gt_image, min=0.0, max=1.0)
        #     gt_image_save = (gt_image_save * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        #     gt_image_save = gt_image_save[:, :, :3]
        #     gt_image_save = Image.fromarray(gt_image_save)
        #     gt_image_save.save("./{}_gt_image_{}.png".format(source_name, iteration))
        #     render_img_save = torch.clamp(image, min=0.0, max=1.0)
        #     render_img_save = (render_img_save * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        #     render_img_save = render_img_save[:, :, :3]
        #     render_img_save = Image.fromarray(render_img_save)
        #     render_img_save.save("./{}_render_image_{}.png".format(source_name,iteration))

        # ##################################################
        # [Stage 0]: 0 to 3w steps, Standard 3DGS RGB loss #
        # ##################################################
        if iteration <= opt.start_ins_feat_iter:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(image, gt_image)
            )

        # Start learning instance features after 3W steps.
        if iteration > opt.start_ins_feat_iter:
            # NOTE: Freeze the pre-trained Gaussian parameters and only train the instance features.
            scene.gaussians._xyz = scene.gaussians._xyz.detach()
            scene.gaussians._features_dc = scene.gaussians._features_dc.detach()
            scene.gaussians._features_rest = scene.gaussians._features_rest.detach()
            scene.gaussians._opacity = scene.gaussians._opacity.detach()
            scene.gaussians._scaling = scene.gaussians._scaling.detach()
            scene.gaussians._rotation = scene.gaussians._rotation.detach()

            # construct boolean masks [num_mask, H, W]
            # sam_level, leaf:3, scannet:0
            sam_level = opt.sam_level
            mask_id, mask_bool, invalid_pix = get_SAM_mask_and_feat(
                gt_sam_mask, level=sam_level, filter_th=50
            )

            # #################################################
            # [Stage 1]: Continuous instance feature learning #
            #           LERF 3W-4W steps; ScanNet 3w-5w steps #
            #           see Sec.3.1 in the paper              #
            # #################################################
            if cb_mode is None:
                # (0) compute the average instance features within each mask. [num_mask, 6]
                feat_mean_stack = mask_feature_mean(
                    rendered_ins_feat, mask_bool, image_mask=rendered_silhouette
                )
                # (1) intra-mask smoothing loss. Eq.(1) in the paper
                loss_cohesion = cohesion_loss(
                    rendered_ins_feat, mask_bool, feat_mean_stack
                )
                # (2) inter-mask contrastive loss Eq.(2) in the paper
                loss_separation = separation_loss(feat_mean_stack, iteration)
                # total loss, opt.loss_weight: 0.1
                loss = loss_separation + opt.loss_weight * loss_cohesion

        # ####################################################
        # [Stage 2]: Two-Level Codebook for Discretization
        #   - coarse-level(root) loss computation
        #   - fine-level(leaf) loss computation
        # ####################################################
        # 2.1 coarse-level
        if cb_mode == "root":
            # Only consider valid pixels
            keeped_pix = (
                viewpoint_cam.pesudo_ins_feat.sum(dim=(0)) > 0
            )  # Invalid pixels of pseudo-labels
            keeped_pix = (
                keeped_pix.bool() & rendered_silhouette.bool()
            )  # Empty regions after rescaling
            keeped_pix = keeped_pix & (
                ~invalid_pix.unsqueeze(0)
            )  # Invalid area of the original mask
            keeped_pix = rendered_silhouette.bool()
            # loss  Eq.(4) in the paper.
            feat_loss = l1_loss(
                rendered_ins_feat, viewpoint_cam.pesudo_ins_feat, keeped_pix
            )
            # feat_loss = l2_loss(rendered_ins_feat, viewpoint_cam.pesudo_ins_feat, keeped_pix)
            loss = feat_loss
        # 2.2 fine-level
        if cb_mode == "leaf" and no_need_bk == False:
            total_pix = gt_image.shape[1] * gt_image.shape[2]
            for i in range(len(rendered_cluster_imgs)):
                cluster_pred = rendered_cluster_imgs[i]
                cluster_silhouette = rendered_cluster_silhouettes[i]  # [H, W] bool
                rendered_ins_feat = cluster_pred  #
                # cluster_mask = viewpoint_cam.cluster_masks[i]     # [H, W] bool
                # cluster_silhouette = cluster_silhouette & cluster_mask
                feat_loss = l2_loss(
                    cluster_pred, viewpoint_cam.pesudo_ins_feat, cluster_silhouette
                )
                if i == 0:
                    # loss = feat_loss * (cluster_silhouette.sum() / total_pix)
                    loss = feat_loss
                else:
                    # loss += (feat_loss * (cluster_silhouette.sum() / total_pix))
                    loss += feat_loss

        # mask loss. modify -----
        if viewpoint_cam.original_mask is not None:
            gt_mask = viewpoint_cam.original_mask.cuda()
            mask_loss = F.mse_loss(alpha, gt_mask)
            loss = loss + mask_loss

        if no_need_bk == False:
            loss.backward()

        iter_end.record()

        # Save the intermediate training results. [OpenGaussian]
        save_intermediate = True
        save_fre = 1000
        if iteration > opt.start_leaf_cb_iter:
            save_fre = 100
        if (iteration % save_fre == 0) and save_intermediate:
            gts_path = os.path.join(scene.model_path, "train_process", "gt")
            makedirs(gts_path, exist_ok=True)
            torchvision.utils.save_image(
                gt_image.detach().cpu(),
                os.path.join(gts_path, "{0:05d}".format(iteration) + ".png"),
            )

            render_path = os.path.join(scene.model_path, "train_process", "renders")
            makedirs(render_path, exist_ok=True)
            torchvision.utils.save_image(
                image.detach().cpu(),
                os.path.join(render_path, "{0:05d}".format(iteration) + ".png"),
            )

            # alpha_path = os.path.join(scene.model_path, "train_process", "alpha")
            # makedirs(alpha_path, exist_ok=True)
            # torchvision.utils.save_image(alpha.detach().cpu(), os.path.join(alpha_path, '{0:05d}'.format(iteration) + ".png"))

            if iteration > opt.start_ins_feat_iter:
                if cb_mode is None:
                    sub_floader = "stage1"
                elif cb_mode == "root":
                    sub_floader = "stage2_1"
                elif cb_mode == "leaf":
                    sub_floader = "stage2_2"
                # Visualize the SAM mask. [OpenGaussian]
                if gt_sam_mask is not None and iteration > opt.start_ins_feat_iter:
                    # read predefined mask color
                    mask_color_rand = colors_defined[mask_id.detach().cpu()].type(
                        torch.float64
                    )
                    mask_color_rand = mask_color_rand.permute(2, 0, 1)
                    gt_sam_path = os.path.join(
                        scene.model_path,
                        "train_process",
                        sub_floader,
                        "gt_sam_mask_" + str(opt.sam_level),
                    )
                    makedirs(gt_sam_path, exist_ok=True)
                    torchvision.utils.save_image(
                        mask_color_rand / 255.0,
                        os.path.join(gt_sam_path, "{0:05d}".format(iteration) + ".png"),
                    )

                # TODO
                if viewpoint_cam.pesudo_ins_feat is not None:
                    feat = viewpoint_cam.pesudo_ins_feat
                    pseudo_ins_feat_path = os.path.join(
                        scene.model_path,
                        "train_process",
                        sub_floader,
                        "pseudo_ins_feat",
                    )
                    makedirs(pseudo_ins_feat_path, exist_ok=True)
                    torchvision.utils.save_image(
                        feat.detach().cpu()[:3, :, :],
                        os.path.join(
                            pseudo_ins_feat_path, "{0:05d}".format(iteration) + "_1.png"
                        ),
                    )
                    torchvision.utils.save_image(
                        feat.detach().cpu()[3:6, :, :],
                        os.path.join(
                            pseudo_ins_feat_path, "{0:05d}".format(iteration) + "_2.png"
                        ),
                    )

                if cb_mode is not None:
                    # silhouette (alpha to mask) [OpenGaussian] stage 2
                    silhouette_path = os.path.join(
                        scene.model_path, "train_process", sub_floader, "silhouette"
                    )
                    makedirs(silhouette_path, exist_ok=True)
                    torchvision.utils.save_image(
                        rendered_silhouette.detach().cpu(),
                        os.path.join(
                            silhouette_path, "{0:05d}".format(iteration) + ".png"
                        ),
                    )

                # Visualize the 6-dimensional instance feature. [OpenGuassian]
                if rendered_ins_feat is not None:
                    # dim 0:3
                    ins_feat_path = os.path.join(
                        scene.model_path, "train_process", sub_floader, "ins_feat"
                    )
                    makedirs(ins_feat_path, exist_ok=True)
                    torchvision.utils.save_image(
                        rendered_ins_feat.detach().cpu()[:3, :, :],
                        os.path.join(
                            ins_feat_path, "{0:05d}".format(iteration) + ".png"
                        ),
                    )
                    # dim 3:6
                    ins_feat_path2 = os.path.join(
                        scene.model_path, "train_process", sub_floader, "ins_feat2"
                    )
                    makedirs(ins_feat_path2, exist_ok=True)
                    torchvision.utils.save_image(
                        rendered_ins_feat.detach().cpu()[3:6, :, :],
                        os.path.join(
                            ins_feat_path2, "{0:05d}".format(iteration) + ".png"
                        ),
                    )

                # # fine-level cluster
                # if rendered_leaf_cluster_imgs is not None:
                #     leaf_cluster_path = os.path.join(scene.model_path, "train_process", sub_floader, "cluster_leaf")
                #     makedirs(leaf_cluster_path, exist_ok=True)
                #     for i, leaf_img in enumerate(rendered_leaf_cluster_imgs):
                #         torchvision.utils.save_image(leaf_img.detach().cpu()[:3, :, :], os.path.join(leaf_cluster_path, '{0:05d}'.format(iteration) + "leaf_{}.png".format(i)))

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save .ply
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
            #     testing_iterations, opt.start_root_cb_iter, scene, render, (pipe, background, iteration))
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                sys.stdout.flush()
                if iteration > opt.start_root_cb_iter:
                    # note: save codebook [OpenGaussian]
                    out_dir = os.path.join(
                        scene.model_path, "point_cloud/iteration_%d" % iteration
                    )
                    save_kmeans([ins_feat_codebook], ["ins_feat"], out_dir, mode="root")
                    if cb_mode == "leaf":
                        save_kmeans(
                            [ins_feat_codebook], ["ins_feat"], out_dir, mode="leaf"
                        )
                    scene.save(iteration, ["ins_feat"])
                else:
                    scene.save(iteration)

            # Densification
            if (
                iteration < opt.densify_until_iter and not opt.frozen_init_pts
            ):  # note: ScanNet dataset is not densified [OpenGaussian]
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                sys.stdout.flush()
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # ###########################################################
            # Stage 3. associate language feature (training-free stage) #
            #   - Performed after training.                             #
            # ###########################################################
            if iteration == opt.iterations and iteration > opt.start_leaf_cb_iter:
                print(
                    "[Stage 3] Start 2D language feature - 3D cluster association ..."
                )
                sys.stdout.flush()
                if leaf_cluster_indices is None:
                    leaf_cluster_indices = (
                        ins_feat_codebook.leaf_cls_ids
                    )  # fine-level cluster id
                construct_pseudo_ins_feat(
                    scene,
                    render,
                    (pipe, background, first_iter),
                    cluster_indices=leaf_cluster_indices,
                    mode="lang",
                    root_num=opt.root_node_num,
                    leaf_num=opt.leaf_node_num,
                    sam_level=opt.sam_level,
                    save_memory=opt.save_memory,
                )

        # note: save memory (only stage 2, 3)
        if viewpoint_cam.data_on_gpu and opt.save_memory and cb_mode is not None:
            viewpoint_cam.to_cpu()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def construct_pseudo_ins_feat(
    scene: Scene,
    renderFunc,
    renderArgs,
    filter=True,  # filter pseudo features
    cluster_indices=None,  # coarse-level ID of each point (0 ~ k1-1)
    mode="root",  # root, leaf, lang
    root_num=64,
    leaf_num=10,  # k1, k2
    sam_level=3,
    save_memory=False,
):
    torch.cuda.empty_cache()
    # ##############################################################################################
    # [Stage 2.1, 2.2] Render all training views once to construct pseudo-instance feature labels. #
    #   - view.pesudo_ins_feat  [C=6, H, W]                                                        #
    #   - view.pesudo_mask_bool [num_mask, H, W]                                                   #
    # ##############################################################################################
    sorted_train_cameras = sorted(
        scene.getTrainCameras(), key=lambda Camera: Camera.image_name
    )
    for idx, view in enumerate(tqdm(sorted_train_cameras, desc="construt pseudo feat")):
        if not view.data_on_gpu:
            view.to_gpu()

        # render
        render_pkg = renderFunc(
            view, scene.gaussians, *renderArgs, rescale=False, origin_feat=True
        )
        rendered_ins_feat = render_pkg["ins_feat"]

        # get gt sam mask
        mask_id, mask_bool, invalid_pix = get_SAM_mask_and_feat(
            view.original_sam_mask.cuda(), level=sam_level
        )

        # construt pseudo ins_feat, mask levle
        pseudo_mask_ins_feat_, mask_var, pix_count = mask_feature_mean(
            rendered_ins_feat, mask_bool, return_var=True
        )  # [num_mask, 6]
        pseudo_mask_ins_feat = torch.cat(
            (torch.zeros((1, 6)).cuda(), pseudo_mask_ins_feat_), dim=0
        )  # [num_mask+1, 6]
        # Filter out masks with high variance. Potentially incorrect segmentation.
        filter_mask = mask_var > 0.006  # True->del
        filter_mask = torch.cat(
            (torch.tensor([False]).cuda(), filter_mask), dim=0
        )  # [num_mask+1]
        # Masks with large pixel ratio may be background points, inevitably leading to a large variance， Keep them.
        ignored_mask_ind = torch.nonzero(pix_count > pix_count.max() * 0.8).squeeze()
        filter_mask[ignored_mask_ind + 1] = False
        filtered_mask_pseudo_ins_feat = pseudo_mask_ins_feat.clone()
        filtered_mask_pseudo_ins_feat[filter_mask] *= 0

        # pseudo ins_feat, image level
        pseudo_ins_feat = pseudo_mask_ins_feat[
            mask_id
        ]  # Retrieve corresponding ins_feat by mask ID
        pseudo_ins_feat = pseudo_ins_feat.permute(2, 0, 1)  # [H, W, 6]->[6, H, W]

        # filterd pseudo ins_feat, image level
        filter_pseudo_ins_feat = filtered_mask_pseudo_ins_feat[mask_id]
        filter_pseudo_ins_feat = filter_pseudo_ins_feat.permute(2, 0, 1)

        # filtered mask [1+num_mask, H, W]
        mask_bool_filtered = torch.cat(
            (torch.zeros_like(mask_bool[0].unsqueeze(0)), mask_bool), dim=0
        )
        mask_bool_filtered[filter_mask] *= 0

        # NOTE: save the construct pesudo_ins_feat
        # total_feat.append(pseudo_mask_ins_feat[1:,:])
        # if view.pesudo_ins_feat is None:
        view.pesudo_ins_feat = filter_pseudo_ins_feat if filter else pseudo_ins_feat
        # view.pesudo_ins_feat = rendered_ins_feat
        view.pesudo_mask_bool = mask_bool_filtered.to(torch.bool)

        # Save some results for visualization.
        pseudo_debug = True
        if idx % 20 == 0 and pseudo_debug:
            pseudo_ins_feat_path = os.path.join(
                scene.model_path,
                "train_process",
                "debug_pseudo_label",
                "all_pseudo_ins_feat",
            )
            filter_pseudo_ins_feat_path = os.path.join(
                scene.model_path,
                "train_process",
                "debug_pseudo_label",
                "all_filter_pseudo_ins_feat",
            )
            rendered_ins_feat_path = os.path.join(
                scene.model_path,
                "train_process",
                "debug_pseudo_label",
                "all_render_ins_feat",
            )
            sam_mask_path = os.path.join(
                scene.model_path, "train_process", "debug_pseudo_label", "all_sam_mask"
            )
            makedirs(pseudo_ins_feat_path, exist_ok=True)
            makedirs(filter_pseudo_ins_feat_path, exist_ok=True)
            makedirs(rendered_ins_feat_path, exist_ok=True)
            makedirs(sam_mask_path, exist_ok=True)

            # pseudo ins_feat
            torchvision.utils.save_image(
                pseudo_ins_feat[:3, :, :],
                os.path.join(pseudo_ins_feat_path, "{0:05d}".format(idx) + "_1.png"),
            )
            # torchvision.utils.save_image(pseudo_ins_feat[3:6,:,:], os.path.join(pseudo_ins_feat_path, '{0:05d}'.format(idx) + "_2.png"))
            # filtered pseudo ins_feat
            torchvision.utils.save_image(
                filter_pseudo_ins_feat[:3, :, :],
                os.path.join(
                    filter_pseudo_ins_feat_path, "{0:05d}".format(idx) + "_1.png"
                ),
            )
            # torchvision.utils.save_image(filter_pseudo_ins_feat[3:6,:,:], os.path.join(filter_pseudo_ins_feat_path, '{0:05d}'.format(idx) + "_2.png"))
            # rendered ins_feat
            torchvision.utils.save_image(
                rendered_ins_feat[:3, :, :],
                os.path.join(rendered_ins_feat_path, "{0:05d}".format(idx) + "_1.png"),
            )
            # torchvision.utils.save_image(rendered_ins_feat[3:6,:,:], os.path.join(rendered_ins_feat_path, '{0:05d}'.format(idx) + "_2.png"))
            # gt SAM mask, read predefined mask color
            mask_color_rand = colors_defined[mask_id.detach().cpu()].type(torch.float64)
            mask_color_rand = mask_color_rand.permute(2, 0, 1)
            torchvision.utils.save_image(
                mask_color_rand / 255.0,
                os.path.join(sam_mask_path, "{0:05d}".format(idx) + ".png"),
            )
        # to cpu
        if view.data_on_gpu and save_memory:
            view.to_cpu()

    # ##################################################################################################
    # Preprocessing for Stage 2.2
    # determine how many objects are in each coarse cluster, not just setting a fixed k2 value.
    # ##################################################################################################
    torch.cuda.empty_cache()
    if mode == "leaf":
        iClusterSubNum = torch.ones(cluster_indices.max() + 1).to(torch.int32)
        for idx, view in enumerate(
            tqdm(sorted_train_cameras, desc="render coarse-level cluster")
        ):
            if not view.data_on_gpu:
                view.to_gpu()
            render_pkg = renderFunc(
                view,
                scene.gaussians,
                *renderArgs,
                cluster_idx=cluster_indices,
                rescale=False,
                render_feat_map=False,
                render_cluster=True,
                origin_feat=True,
                better_vis=True,
                root_num=root_num,
                leaf_num=leaf_num,
            )
            rendered_cluster_imgs = render_pkg[
                "cluster_imgs"
            ]  # coarse cluster feature map
            rendered_cluster_silhouettes = render_pkg[
                "cluster_silhouettes"
            ]  # coarse cluster mask
            cluster_occur = render_pkg[
                "cluster_occur"
            ]  # bool [k1] Whether coarse clusters visible in the current view

            pser_cluster_pesudo_mask = []
            i = -1
            for cluster_idx in range(cluster_indices.max() + 1):
                if not cluster_occur[
                    cluster_idx
                ]:  # Process only coarse clusters visible in the current view
                    continue

                i += 1
                rendered_ins_feat = rendered_cluster_imgs[i]  # cluster feat map
                rendered_silhouette = (rendered_cluster_silhouettes[i] > 0.9).unsqueeze(
                    0
                )  # cluster mask

                # (1) compute the IoU of this cluster with pseudo masks.
                ious = calculate_iou(
                    view.pesudo_mask_bool, rendered_silhouette, base="former"
                )
                # pseudo masks with IoU above threshold
                inters_mask = view.pesudo_mask_bool[ious[0] > 0.2]  # [num_mask, H, W]
                inters_mask_ = inters_mask.sum(0).to(torch.bool)  # [H, W] bool
                # pseudo mask features, noly for visalization [6, H, W]
                inters_pesudo_ins_feat = view.pesudo_ins_feat * inters_mask_.unsqueeze(
                    0
                )

                # (2) compute the distance between coarse cluster features and pseudo features
                # mean feature of the pesudo mask, [num_mask, 6]
                inters_mask_feat_mean = mask_feature_mean(
                    view.pesudo_ins_feat, inters_mask
                )
                # mean feature of the cluster, [num_mask, 6]
                cluster_mask_feat_mean = mask_feature_mean(
                    rendered_ins_feat, inters_mask, image_mask=rendered_silhouette
                )
                # distance
                l1_dis, l2_dis = calculate_distances(
                    inters_mask_feat_mean, cluster_mask_feat_mean
                )  # metric="l1"

                # (3) filter out some pseudo masks
                inters_mask_filter = inters_mask[
                    (l1_dis < 0.9) & (l2_dis < 0.5)
                ]  # l2_disk < 0.8
                if inters_mask_filter.shape[0] > 10:  # TODO 10? --> leaf_num
                    smallest_10 = torch.topk(l1_dis, 10, largest=False)[1]
                    inters_mask_filter = inters_mask[smallest_10]
                inters_mask_filter_ = inters_mask_filter.sum(0).to(torch.bool)
                inters_pesudo_ins_feat2 = (
                    view.pesudo_ins_feat * inters_mask_filter_.unsqueeze(0)
                )  # noly for visalization
                if (
                    inters_mask_filter_.any() == False
                ):  # Skip if the cluster doesn’t intersect with any pseudo masks.
                    cluster_occur[cluster_idx] = False
                    continue

                pser_cluster_pesudo_mask.append(inters_mask_filter_)  # valid mask
                # NOTE: (4) Determine the number of masks (i.e., objects) in each coarse cluster.
                iClusterSubNum[cluster_idx] = max(
                    iClusterSubNum[cluster_idx], inters_mask_filter.shape[0]
                )

                # (5) save some intermediate results for debugging
                coarse_debug = False
                if coarse_debug:
                    cluster_path = os.path.join(
                        scene.model_path,
                        "train_process",
                        "debug_coarse_cluster",
                        "cluster",
                    )
                    cluster_silhouette_path = os.path.join(
                        scene.model_path,
                        "train_process",
                        "debug_coarse_cluster",
                        "cluster_silhouette",
                    )
                    cluster_inters_pesudo_path = os.path.join(
                        scene.model_path,
                        "train_process",
                        "debug_coarse_cluster",
                        "cluster_inters_pesudo",
                    )
                    makedirs(cluster_path, exist_ok=True)
                    makedirs(cluster_silhouette_path, exist_ok=True)
                    makedirs(cluster_inters_pesudo_path, exist_ok=True)

                    # coarse-level cluster feature map
                    torchvision.utils.save_image(
                        rendered_ins_feat[:3, :, :].cpu(),
                        os.path.join(
                            cluster_path,
                            "{0:05d}".format(idx) + f"_c_{cluster_idx}" + "_1.png",
                        ),
                    )
                    # torchvision.utils.save_image(rendered_ins_feat[3:,:,:].cpu(), os.path.join(cluster_path, '{0:05d}'.format(idx) + f"_c_{cluster_idx}" + "_2.png"))
                    torchvision.utils.save_image(
                        rendered_silhouette.to(torch.float32).cpu(),
                        os.path.join(
                            cluster_silhouette_path,
                            "{0:05d}".format(idx) + f"_c_{cluster_idx}" + "_1.png",
                        ),
                    )

                    # pseudo masks of coarse cluster (_f represents the filtered.)
                    torchvision.utils.save_image(
                        inters_pesudo_ins_feat[:3, :, :].cpu(),
                        os.path.join(
                            cluster_inters_pesudo_path,
                            "{0:05d}".format(idx) + f"_c_{cluster_idx}" + "_1.png",
                        ),
                    )
                    # torchvision.utils.save_image(inters_pesudo_ins_feat[3:,:,:].cpu(), os.path.join(cluster_inters_pesudo_path, '{0:05d}'.format(idx) + f"_c_{cluster_idx}" + "_2.png"))
                    torchvision.utils.save_image(
                        inters_pesudo_ins_feat2[:3, :, :].cpu(),
                        os.path.join(
                            cluster_inters_pesudo_path,
                            "{0:05d}".format(idx) + f"_c_{cluster_idx}" + "_1_f.png",
                        ),
                    )
                    # torchvision.utils.save_image(inters_pesudo_ins_feat2[3:,:,:].cpu(), os.path.join(cluster_inters_pesudo_path, '{0:05d}'.format(idx) + f"_c_{cluster_idx}" + "_2_f.png"))

            if view.cluster_masks is None:
                view.cluster_masks = (
                    pser_cluster_pesudo_mask  # pseudo masks of coarse cluster
                )
                view.bClusterOccur = (
                    cluster_occur  # whether visible in the current view
                )

            if view.data_on_gpu and save_memory:
                view.to_cpu()

        # update
        scene.gaussians.iClusterSubNum = (iClusterSubNum + 1).clamp(max=leaf_num)
        torch.cuda.empty_cache()

    # ###########################################################################
    # [Stage 3] 2D mask(and language feat) - 3D fine level cluster association  #
    #   - Sec. 3.3 in the paper                                                 #
    # ###########################################################################
    if mode == "lang":
        # [leaf_num, view_num, (matched_mask_id, matched_score, b_matched)]
        match_info = torch.zeros(
            root_num * leaf_num, len(sorted_train_cameras), 3
        ).cuda()  # [k1*k2, num_imgs, 3]
        # iterate over the coarse-level clusters
        for root_id, _ in enumerate(tqdm(range(root_num), desc="mapping")):
            # iterate over all training views
            for v_id, view in enumerate(sorted_train_cameras):
                if not view.data_on_gpu:
                    view.to_gpu()

                # (0) render
                render_pkg = renderFunc(
                    view,
                    scene.gaussians,
                    *renderArgs,
                    leaf_cluster_idx=cluster_indices,
                    rescale=False,
                    render_feat_map=False,
                    render_cluster=True,
                    origin_feat=True,
                    better_vis=False,
                    selected_root_id=root_id,
                    root_num=root_num,
                    leaf_num=leaf_num,
                )
                rendered_leaf_cluster_imgs = render_pkg[
                    "leaf_clusters_imgs"
                ]  # all fine-level clusters of the root_id-th coarse-level.
                rendered_leaf_cluster_silhouettes = render_pkg[
                    "leaf_cluster_silhouettes"
                ]
                occured_leaf_id = render_pkg["occured_leaf_id"]
                if len(occured_leaf_id) > 0:
                    occured_leaf_id = torch.tensor(occured_leaf_id).cuda()
                    rendered_leaf_cluster_imgs = torch.stack(
                        rendered_leaf_cluster_imgs, dim=0
                    )  # [N, C, H, W]
                    rendered_leaf_cluster_silhouettes = (
                        rendered_leaf_cluster_silhouettes > 0.8
                    )  # [N, H, W]
                else:
                    if view.data_on_gpu and save_memory:
                        view.to_cpu()
                    continue  # root_id not visible in current view

                # (1) iou  [num_rendered_leaf, num_mask]
                ious = calculate_iou(
                    view.pesudo_mask_bool, rendered_leaf_cluster_silhouettes
                )

                # (2) feature distance
                # cluster mean feat, [num_leaf, dim]
                pred_mask_feat_mean = pair_mask_feature_mean(
                    rendered_leaf_cluster_imgs, rendered_leaf_cluster_silhouettes
                )
                # pesudo mean feat, [num_pesudo_mask, dim]
                pesudo_mask_feat_mean = mask_feature_mean(
                    view.pesudo_ins_feat, view.pesudo_mask_bool
                )
                # only for visualization, [num_pesudo_mask, dim， H, W]
                pesudo_mask_feat = (
                    view.pesudo_ins_feat * view.pesudo_mask_bool.unsqueeze(1)
                )
                # distance
                l1_dis, _ = calculate_pairwise_distances(
                    pred_mask_feat_mean, pesudo_mask_feat_mean, metric="l1"
                )  # method="l1"

                # (3) iou-feature distance joint score
                scores = ious * (1 - l1_dis)  # Eq.(5) in the paper

                # (4) save the association result
                max_score, max_ind = torch.max(scores, dim=-1)  # [num_leaf]
                b_matched = max_score > 0.2  # todo
                max_score[~b_matched] *= 0
                max_ind[~b_matched] *= 0
                match_info[occured_leaf_id, v_id] = torch.stack(
                    (max_ind, max_score, b_matched), dim=1
                )

                # (5) save matching results for visualization. (only save the paired mask)
                association_debug = True
                if association_debug:
                    leaf_cluster_path = os.path.join(
                        scene.model_path, "train_process", "stage3", "leaf_cluster"
                    )
                    leaf_cluster_silhouette_path = os.path.join(
                        scene.model_path,
                        "train_process",
                        "stage3",
                        "leaf_cluster_silhouettes",
                    )
                    leaf_pesudo_mask_path = os.path.join(
                        scene.model_path, "train_process", "stage3", "leaf_pesudo_mask"
                    )
                    makedirs(leaf_cluster_path, exist_ok=True)
                    makedirs(leaf_cluster_silhouette_path, exist_ok=True)
                    makedirs(leaf_pesudo_mask_path, exist_ok=True)
                    if b_matched.sum() > 0:
                        for i, img in enumerate(rendered_leaf_cluster_imgs):
                            if not b_matched[i]:
                                continue
                            if (
                                max_score[i] < 0.8
                            ):  # note: 0.8 is just for visualization
                                continue
                            torchvision.utils.save_image(
                                img[:3, :, :],
                                os.path.join(
                                    leaf_cluster_path, f"r{root_id}_l{i}_v{v_id}.png"
                                ),
                            )
                            torchvision.utils.save_image(
                                rendered_leaf_cluster_silhouettes[i].to(torch.float32),
                                os.path.join(
                                    leaf_cluster_silhouette_path,
                                    f"r{root_id}_l{i}_v{v_id}.png",
                                ),
                            )
                            torchvision.utils.save_image(
                                pesudo_mask_feat[max_ind[i]][:3, :, :],
                                os.path.join(
                                    leaf_pesudo_mask_path,
                                    f"r{root_id}_l{i}_v{v_id}.png",
                                ),
                            )
                    # print("end one root cluster of one view")
                if view.data_on_gpu and save_memory:
                    view.to_cpu()
        # print("end matching")
        torch.cuda.empty_cache()

        # count the matches of each leaf (fine-level cluster) across all viewpoints.
        leaf_per_view_matched_mask = match_info[:, :, 0].to(
            torch.int64
        )  # [k1*k2, num_cam] matched mask id
        match_info_sum = match_info.sum(
            dim=1
        )  # [k1*k2, (matched_mask_id, matched_score, b_matched)]
        leaf_ave_score = match_info_sum[:, 1] / (
            match_info_sum[:, 2] + 1e-6
        )  # [k1*k2] ave score
        leaf_occu_count = match_info_sum[
            :, 2
        ]  # [k1*k2] number of matches for each leaf

        # accumulated 2D features of each leaf
        per_leaf_feat_sum = torch.zeros(root_num * leaf_num, 512).cuda()  # [k1*k2]
        for v_id, view in enumerate(sorted_train_cameras):  # all camera views
            if not view.data_on_gpu:
                view.to_gpu()
            if sam_level == 0:
                strat_id = 0
                end_id = view.original_sam_mask[sam_level].max().to(torch.int64) + 1
            else:
                strat_id = (
                    view.original_sam_mask[sam_level - 1].max().to(torch.int64) + 1
                )
                end_id = view.original_sam_mask[sam_level].max().to(torch.int64) + 1
            curr_view_lang_feat = view.original_mask_feat[
                strat_id:end_id, :
            ]  # [num_mask, 512]
            curr_view_lang_feat = torch.cat(
                (
                    torch.zeros_like(curr_view_lang_feat[0]).unsqueeze(0),
                    curr_view_lang_feat,
                )
            )  # note: [num_mask+1, 512] add a feature with all 0s, i.e., the feature with id=0.
            # current feat [k1*k2, 512]
            single_view_leaf_feat = curr_view_lang_feat[
                leaf_per_view_matched_mask[:, v_id]
            ]
            # accumulate
            per_leaf_feat_sum += single_view_leaf_feat

            if view.data_on_gpu and save_memory:
                view.to_cpu()

        # average language features [k1*k2, 512]
        per_leaf_feat = per_leaf_feat_sum / (leaf_occu_count + 1e-4).unsqueeze(1)

        # save per_leaf_feat[k1*k2, 512], leaf_ave_score[k1*k2], leaf_occu_count[k1*k2], cluster_indices[num_pts]
        np.savez(
            f"{scene.model_path}/cluster_lang.npz",
            leaf_feat=per_leaf_feat.cpu().numpy(),
            leaf_score=leaf_ave_score.cpu().numpy(),
            occu_count=leaf_occu_count.cpu().numpy(),
            leaf_ind=cluster_indices.cpu().numpy(),
        )


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    start_root_cb_iter,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                sys.stdout.flush()
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


# initialize new gaussian parameters. modify -----
def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3]  # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))  # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    logit_ins_feat = torch.zeros((num_pts, 3), dtype=torch.float, device="cuda")
    # color [N, 3, 16]
    max_sh_degree = 3
    fused_color = RGB2SH(new_pt_cld[:, 3:6])
    features = (
        torch.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2)).float().cuda()
    )  # [N, 3, 16]
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0
    params = {
        "new_xyz": means3D,
        "new_features_dc": features[:, :, 0:1].transpose(1, 2).contiguous(),
        "new_features_rest": features[:, :, 1:].transpose(1, 2).contiguous(),
        "new_opacities": logit_opacities,
        # 'new_scaling': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
        "new_scaling": torch.tile(
            torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3)
        ),
        "new_rotation": unnorm_rots,
        "new_ins_feat": logit_ins_feat,
    }

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(
                torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
            )
        else:
            params[k] = torch.nn.Parameter(
                v.cuda().float().contiguous().requires_grad_(True)
            )

    return params


# modify -----

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--start_ply", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.start_ply,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
