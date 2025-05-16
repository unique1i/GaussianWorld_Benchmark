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
import math

# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ashawkey_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.opengs_utlis import *

# from sklearn.neighbors import NearestNeighbors
import pytorch3d.ops


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    iteration,
    scaling_modifier=1.0,
    override_color=None,
    visible_mask=None,
    mask_num=0,
    cluster_idx=None,  # per-point cluster id (coarse-level)
    leaf_cluster_idx=None,  # per-point cluster id (fine-level)
    rescale=True,  # re-scale (for enhance ins_feat)
    origin_feat=False,  # origin ins_feat (not quantized)
    render_feat_map=True,  # render image-level feat map
    render_color=True,  # render rgb image
    render_cluster=False,  # render cluster, stage 2.2
    better_vis=False,  # filter some points
    selected_root_id=None,  # coarse-level cluster id
    selected_leaf_id=None,  # fine-level cluster id (possibly more than one)
    pre_mask=None,
    seg_rgb=False,  # render cluster rgb, not feat
    post_process=False,  # post
    root_num=64,
    leaf_num=10,
):  # k1, k2
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if render_color:
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = None, None, None, None

    # ################################################################
    # [Stage 1, Stage 2.1] Render image-level instance feature map   #
    #   - rendered_ins_feat: image-level feat map                    #
    # ################################################################
    # probabilistically rescale
    prob = torch.rand(1)
    rescale_factor = torch.tensor(1.0, dtype=torch.float32).cuda()
    if prob > 0.5 and rescale:
        rescale_factor = torch.rand(1).cuda()
    if render_feat_map:
        # get feature
        ins_feat = (
            pc.get_ins_feat(origin=origin_feat) + 1
        ) / 2  # pseudo -> norm, else -> raw
        # first three channels
        rendered_ins_feat, _, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=ins_feat[:, :3],  # render features as pre-computed colors
            opacities=opacity,
            scales=scales * rescale_factor,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        # last three channels
        if ins_feat.shape[-1] > 3:
            rendered_ins_feat2, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=ins_feat[
                    :, 3:6
                ],  # render features as pre-computed colors
                opacities=opacity,
                scales=scales * rescale_factor,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            rendered_ins_feat = torch.cat(
                (rendered_ins_feat, rendered_ins_feat2), dim=0
            )
        # mask
        _, _, _, silhouette = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales * rescale_factor,
            # opacities = opacity*0+1.0,    #
            # scales = scales*0+0.001,   # *0.1
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_ins_feat, silhouette = None, None

    # ########################################################################
    # [Preprocessing for Stage 2.2]: render (coarse) cluster-level feat map  #
    #   - rendered_clusters: feat map of the coarse clusters                 #
    #   - rendered_cluster_silhouettes: cluster mask                         #
    # ########################################################################
    viewed_pts = radii > 0  # ignore the invisible points
    if cluster_idx is not None:
        num_cluster = cluster_idx.max() + 1
        cluster_occur = torch.zeros(num_cluster).to(torch.bool)  # [num_cluster], bool
    else:
        cluster_occur = None
    if render_cluster and cluster_idx is not None and viewed_pts.sum() != 0:
        ins_feat = (
            pc.get_ins_feat(origin=origin_feat) + 1
        ) / 2  # pseudo -> norm, else -> raw
        rendered_clusters = []
        rendered_cluster_silhouettes = []
        scale_filter = (scales < 0.5).all(dim=1)  #   filter
        for idx in range(num_cluster):
            if not better_vis and idx != selected_root_id:
                continue

            # ignore the invisible coarse-level cluster
            if (
                viewpoint_camera.bClusterOccur is not None
                and viewpoint_camera.bClusterOccur[idx] == False
            ):
                continue

            # NOTE: Render only the idx-th coarse cluster
            filter_idx = cluster_idx == idx

            filter_idx = filter_idx & viewed_pts
            # todo: filter
            if better_vis:
                filter_idx = filter_idx & scale_filter
                if filter_idx.sum() < 100:
                    continue

            # render cluster-level feat map
            rendered_cluster, _, _, cluster_silhouette = rasterizer(
                means3D=means3D[filter_idx],
                means2D=means2D[filter_idx],
                shs=None,  # feat
                colors_precomp=ins_feat[:, :3][filter_idx],  # feat
                # shs = shs[filter_idx],  # rgb
                # colors_precomp = None,  # rgb
                opacities=opacity[filter_idx],
                scales=scales[filter_idx] * rescale_factor,
                rotations=rotations[filter_idx],
                cov3D_precomp=cov3D_precomp,
            )
            if ins_feat.shape[-1] > 3:
                rendered_cluster2, _, _, cluster_silhouette = rasterizer(
                    means3D=means3D[filter_idx],
                    means2D=means2D[filter_idx],
                    shs=None,  # feat
                    colors_precomp=ins_feat[:, 3:][filter_idx],  # feat
                    # shs = shs[filter_idx],  # rgb
                    # colors_precomp = None,  # rgb
                    opacities=opacity[filter_idx],
                    scales=scales[filter_idx] * rescale_factor,
                    rotations=rotations[filter_idx],
                    cov3D_precomp=cov3D_precomp,
                )
                rendered_cluster = torch.cat(
                    (rendered_cluster, rendered_cluster2), dim=0
                )

            # alpha --> mask
            if cluster_silhouette.max() > 0.8:
                cluster_occur[idx] = True
                rendered_clusters.append(rendered_cluster)
                rendered_cluster_silhouettes.append(cluster_silhouette)
        if len(rendered_cluster_silhouettes) != 0:
            rendered_cluster_silhouettes = torch.vstack(rendered_cluster_silhouettes)
    else:
        rendered_clusters, rendered_cluster_silhouettes = None, None

    # ###############################################################
    # [Stage 2.2 & Stage 3] render (fine) cluster-level feat map    #
    #   - rendered_leaf_clusters: feat map of the fine clusters     #
    #   - rendered_leaf_cluster_silhouettes: fine cluster mask      #
    #   - occured_leaf_id: visible fine cluster id                  #
    # ###############################################################
    if leaf_cluster_idx is not None and leaf_cluster_idx.numel() > 0:
        ins_feat = (
            pc.get_ins_feat(origin=origin_feat) + 1
        ) / 2  # pseudo -> norm, else -> raw
        # todo: rescale
        scale_filter = (scales < 0.1).all(dim=1)
        # scale_filter = (scales < 0.1).all(dim=1) & (opacity > 0.1).squeeze(-1)
        re_scale_factor = torch.ones_like(opacity)  # not used

        # determine the fine cluster ID range (lerf_range) based on the coarse cluster ID (selected_leaf_id).
        # root_num = 64   # todo modify
        # leaf_num = 5    # todo modify
        rendered_leaf_clusters = []
        rendered_leaf_cluster_silhouettes = []
        occured_leaf_id = []
        if selected_leaf_id is None:
            if selected_root_id is not None:
                start_leaf = selected_root_id * leaf_num  # todo 10
                end_leaf = start_leaf + leaf_num  # todo 10
            else:
                start_leaf = 0
                end_leaf = root_num * leaf_num  # todo 64 * 10
            lerf_range = range(start_leaf, end_leaf)
        else:
            lerf_range = selected_leaf_id.tolist()
        for _, leaf_idx in enumerate(lerf_range):  # render each fine cluster
            # ignore the invisible clusters
            if (
                viewpoint_camera.bClusterOccur is not None
                and viewpoint_camera.bClusterOccur[selected_root_id] == False
            ):
                continue

            if selected_leaf_id is None:
                filter_idx = (
                    leaf_cluster_idx == leaf_idx
                )  # Render only the idx-th fine cluster
                # filter_idx = labels != value      # remove the idx-th fine cluster
            else:
                filter_idx = (leaf_cluster_idx.unsqueeze(1) == selected_leaf_id).any(
                    dim=1
                )

            # pre-mask
            if pre_mask is not None:
                filter_idx = filter_idx & pre_mask

            filter_idx = filter_idx & viewed_pts
            # filter
            if better_vis:
                filter_idx = filter_idx & scale_filter
                if filter_idx.sum() < 100:
                    continue

            # TODO post process (for 3D object selection)
            # pre_count = filter_idx.sum()
            max_time = 5
            if post_process and max_time > 0:
                nearest_k_distance = pytorch3d.ops.knn_points(
                    means3D[filter_idx].unsqueeze(0),
                    means3D[filter_idx].unsqueeze(0),
                    # K=int(filter_idx.sum()**0.5),
                    K=int(filter_idx.sum() ** 0.5),
                ).dists
                mean_nearest_k_distance, std_nearest_k_distance = (
                    nearest_k_distance.mean(),
                    nearest_k_distance.std(),
                )
                # print(std_nearest_k_distance, "std_nearest_k_distance")

                mask = (
                    nearest_k_distance.mean(dim=-1)
                    < mean_nearest_k_distance + std_nearest_k_distance
                )
                # mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + 0.1 * std_nearest_k_distance

                mask = mask.squeeze()
                if filter_idx is not None:
                    filter_idx[filter_idx != 0] = mask
                max_time -= 1

            if filter_idx.sum() < 10:
                continue

            # record the fine cluster id appears in the current view.
            occured_leaf_id.append(leaf_idx)

            # note: render cluster rgb or feat.
            if seg_rgb:
                _shs = shs[filter_idx]
                _colors_precomp1 = None
                _colors_precomp2 = None
            else:
                _shs = None
                _colors_precomp1 = ins_feat[:, :3][filter_idx]
                _colors_precomp2 = ins_feat[:, 3:][filter_idx]

            rendered_leaf_cluster, _, _, leaf_cluster_silhouette = rasterizer(
                means3D=means3D[filter_idx],
                means2D=means2D[filter_idx],
                shs=_shs,  # rgb or feat
                colors_precomp=_colors_precomp1,  # rgb or feat
                opacities=opacity[filter_idx],
                scales=(scales * re_scale_factor)[filter_idx],
                rotations=rotations[filter_idx],
                cov3D_precomp=cov3D_precomp,
            )
            if ins_feat.shape[-1] > 3:
                rendered_leaf_cluster2, _, _, _ = rasterizer(
                    means3D=means3D[filter_idx],
                    means2D=means2D[filter_idx],
                    shs=_shs,  # rgb or feat
                    colors_precomp=_colors_precomp2,  # rgb or feat
                    opacities=opacity[filter_idx],
                    scales=(scales * re_scale_factor)[filter_idx],
                    rotations=rotations[filter_idx],
                    cov3D_precomp=cov3D_precomp,
                )
                rendered_leaf_cluster = torch.cat(
                    (rendered_leaf_cluster, rendered_leaf_cluster2), dim=0
                )
            rendered_leaf_clusters.append(rendered_leaf_cluster)
            rendered_leaf_cluster_silhouettes.append(leaf_cluster_silhouette)
            if selected_leaf_id is not None and len(rendered_leaf_clusters) > 0:
                break
        if len(rendered_leaf_cluster_silhouettes) != 0:
            rendered_leaf_cluster_silhouettes = torch.vstack(
                rendered_leaf_cluster_silhouettes
            )
    else:
        rendered_leaf_clusters = None
        rendered_leaf_cluster_silhouettes = None
        occured_leaf_id = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "alpha": rendered_alpha,
        "depth": rendered_depth,  # not used
        "silhouette": silhouette,
        "ins_feat": rendered_ins_feat,  # image-level feat map
        "cluster_imgs": rendered_clusters,  # coarse cluster feat map/image
        "cluster_silhouettes": rendered_cluster_silhouettes,  # coarse cluster mask
        "leaf_clusters_imgs": rendered_leaf_clusters,  # fine cluster feat map/image
        "leaf_cluster_silhouettes": rendered_leaf_cluster_silhouettes,  # fine cluster mask
        "occured_leaf_id": occured_leaf_id,  # fine cluster
        "cluster_occur": cluster_occur,  # coarse cluster
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
