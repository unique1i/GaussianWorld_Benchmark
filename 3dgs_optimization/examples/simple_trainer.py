import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import pprint
import wandb
from datasets.colmap import Parser
from datasets.dataset_factory import initialize_datasets
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)
from pathlib import Path

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from plyfile import PlyData, PlyElement


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    use_wandb: bool = False
    dataset_name: str = "blender"
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8097

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the modelƒ
    eval_steps: List[int] = field(default_factory=lambda: [30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.5
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e2

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = True

    # Use random background for training to discourage transparency
    random_bkgd: bool = False
    white_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1.0

    # ignore this pixel border when calculating the image loss
    ignore_pixel_border: int = 0

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False
    # Save val images
    val_save_image: bool = False
    # use init_point_num to cap the max number of GSs when MCMC strategy is used
    cap_max_by_init_point_num: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"
    render_traj: bool = False
    prune_by_bbox: bool = False
    disable_growing: bool = False  # disable the growing of GSs in default startegy
    disable_pruning: bool = False  # disable the pruning of GSs in default startegy
    # disable the optimization of xyz in default strategy
    disable_xyz_opti: bool = False
    reset_every: int = 3000

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(
                strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(
                strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            print("MCMC strategy in use.")
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Optional[Parser] = None,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.5,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    points3d_data: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if parser is not None:
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * \
                (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError(
                "Please specify a correct init_type: sfm or random")
        print(f"Init_type {init_type} used.")
    elif points3d_data is not None:
        # Initialize from points3d.ply data
        points = torch.from_numpy(points3d_data["points"]).float()
        rgbs = torch.from_numpy(points3d_data["colors"]).float()
        print("Init_type from points3d_data used.")
    else:
        print("No parser or points3d_data provided, use random 3DGS initialization.")
        points = init_extent * scene_scale * \
            (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(
        dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(
            colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.init_point_num = None
        self.max_sh_degree = cfg.sh_degree

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Dataset loading
        dataset_handle = initialize_datasets(cfg)
        self.parser = dataset_handle.parser
        self.trainset = dataset_handle.trainset
        self.valset = dataset_handle.valset
        self.init_point_num = dataset_handle.init_point_num
        self.init_bbox_min = dataset_handle.init_bbox_min
        self.init_bbox_max = dataset_handle.init_bbox_max
        self.points3d_data = dataset_handle.points3d_data
        self.scene_scale = dataset_handle.scene_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            points3d_data=self.points3d_data  # Pass the points3d data
        )
        if self.cfg.disable_xyz_opti:
            self.splats["means"].requires_grad_(False)
            for param_group in self.optimizers["means"].param_groups:
                param_group["lr"] = 0.0  # Set learning rate to zero
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
            self.cfg.strategy.disable_growing = self.cfg.disable_growing
            self.cfg.strategy.disable_pruning = self.cfg.disable_pruning
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)
        if self.init_point_num is not None and self.cfg.cap_max_by_init_point_num:
            # when init_point_num is the from point clouds voxel downsampling, it reflects the scene size
            cap_max = min(round(1.5 * self.init_point_num / 1_000)
                          * 1_000, self.cfg.strategy.cap_max)
            self.cfg.strategy.cap_max = int(cap_max)
            print(f"MCMC cap_max of GSs set to {cap_max}.")
        if self.cfg.prune_by_bbox:
            self.cfg.strategy.init_bbox_min = self.init_bbox_min
            self.cfg.strategy.init_bbox_max = self.init_bbox_max
            self.cfg.strategy.prune_by_bbox = True
        self.cfg.strategy.reset_every = self.cfg.reset_every
        pprint.pp(self.cfg)
        if self.cfg.use_wandb:
            import uuid
            wandb.init(
                project="GaussianWorld",
                config=self.cfg,
                group=self.cfg.dataset_name,
                name=f"{dataset_handle.scene_name}_{uuid.uuid4().hex[0:5]}",
            )

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(
                    f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(
                len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(
                len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat(
                [self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            **kwargs,
        )
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps+1))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)
                if data["image"] is None or (cfg.depth_loss and data["depth"] is None):
                    sample_info = f"image_id: {data['image_id'].item()}, name: {data['image_name']}"
                    print(
                        f"\033[93mWARNING [Step {step}]: Skipping sample with failed load ({sample_info})\033[0m")

                    if not cfg.disable_viewer:
                        self.viewer.lock.release()
                    continue
            except Exception as e:
                # catch any data loading error
                error_msg = f"[Step {step}] Skipping iteration due to data loading error"
                print(f"\033[93mWARNING: {error_msg}\033[0m")  # Yellow text
                # print(f"Error: {e}")

                if not cfg.disable_viewer:
                    self.viewer.lock.release()
                continue

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(
                device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                # points = data["points"].to(device)  # [1, M, 2]
                # depths_gt = data["depths"].to(device)  # [1, M]
                depth_gt = data["depth"].to(device)  # [1, H, W]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(
                step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+D" if cfg.depth_loss else "RGB",
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
            elif cfg.white_bkgd:
                colors = colors + 1.0 * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            if self.cfg.ignore_pixel_border > 0:
                ignore_pixel_border = self.cfg.ignore_pixel_border
                pixels = pixels[:, ignore_pixel_border:-ignore_pixel_border,
                                ignore_pixel_border:-ignore_pixel_border, :]
                colors = colors[:, ignore_pixel_border:-ignore_pixel_border,
                                ignore_pixel_border:-ignore_pixel_border, :]
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + \
                ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # if self.cfg.ignore_pixel_border > 0:
                #     depth_gt = depth_gt[:, ignore_pixel_border:-ignore_pixel_border,
                #                        ignore_pixel_border:-ignore_pixel_border]
                #     depths = depths[:, ignore_pixel_border:-ignore_pixel_border,
                #                     ignore_pixel_border:-ignore_pixel_border]

                nan_mask_1 = ~torch.isnan(depth_gt).squeeze()
                nan_mask_2 = ~torch.isnan(depths.squeeze())
                filter_mask = nan_mask_1 & nan_mask_2 & (
                    depth_gt.squeeze() > 0.01)

                if filter_mask.sum() > 0:
                    depth_err = torch.zeros_like(depth_gt.squeeze())
                    depth_err[filter_mask] = depth_gt.squeeze(
                    )[filter_mask] - depths.squeeze()[filter_mask]

                    valid_errors = depth_err[filter_mask]
                    if valid_errors.numel() > 0:
                        median_err = valid_errors.median()
                        if median_err > 0:
                            err_mask = depth_err.abs() < 10 * median_err
                        else:
                            err_mask = depth_err.abs() < 2  # fixed threshold to avoid too large error
                        loss_mask = filter_mask & err_mask

                        if loss_mask.sum() > 0:
                            depthloss = F.l1_loss(
                                depths.squeeze()[loss_mask], depth_gt.squeeze()[loss_mask])
                        else:
                            depthloss = torch.tensor(0, device=loss.device)
                    else:
                        depthloss = torch.tensor(0, device=loss.device)
                else:
                    depthloss = torch.tensor(0, device=loss.device)

                loss += depthloss * cfg.depth_lambda

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg *
                    torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"total loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| rgb loss={l1loss.item():.6f}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            if world_rank == 0 and step % 500 == 0:
                canvas = torch.cat(
                    [pixels, colors], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                if cfg.val_save_image:
                    imageio.imwrite(
                        f"{self.render_dir}/train_rank{self.world_rank}.png",
                        (canvas * 255).astype(np.uint8),
                    )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar(
                        "train/depthloss", depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat(
                        [pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                    disable_growing=self.cfg.disable_growing,
                    disable_pruning=self.cfg.disable_pruning,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i for i in cfg.eval_steps]:
                self.eval(step, eval_depth=cfg.depth_loss)
                self.save_ply(
                    path=f"{self.ckpt_dir}/point_cloud_{step}.ply", prune_by_bbox=True)
                if cfg.render_traj:
                    self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", eval_depth: bool = False):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # our custom collate function that handles error cases
        def collate_fn(batch):
            # Filter out any None values from failed loads
            valid_samples = [b for b in batch if b["image"] is not None and (
                not eval_depth or b["depth"] is not None)]
            if not valid_samples:
                # All samples in this batch failed to load
                return None
            # Use default collation for the valid samples
            return torch.utils.data.dataloader.default_collate(valid_samples)

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "depth_loss": []}

        for i, data in enumerate(valloader):
            try:
                if data is None:
                    print(
                        f"\033[93mWARNING [Eval {stage}, Batch {i}]: skipped due to failed loads\033[0m")
                    continue

                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                height, width = pixels.shape[1:3]
                if eval_depth:
                    depth_gt = data["depth"].to(device)  # [1, H, W]

                torch.cuda.synchronize()
                tic = time.time()
            except Exception as e:
                error_msg = f"[Eval {stage}, Sample {i}] Skipping sample due to error"
                print(f"\033[93mWARNING: {error_msg}\033[0m")
                skipped_samples += 1
                continue

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+D" if eval_depth else "RGB",
            )  # [1, H, W, 3]
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            if world_rank == 0:
                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(
                    0).cpu().numpy()
                if cfg.val_save_image:
                    if i < 50:
                        imageio.imwrite(
                            f"{self.render_dir}/{stage}_{i:04d}.png",
                            (canvas * 255.0).astype(np.uint8),
                        )
                pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                if self.cfg.ignore_pixel_border > 0:
                    ignore_pixel_border = self.cfg.ignore_pixel_border
                    pixels = pixels[:, :, ignore_pixel_border:-ignore_pixel_border,
                                    ignore_pixel_border:-ignore_pixel_border]
                    colors = colors[:, :, ignore_pixel_border:-ignore_pixel_border,
                                    ignore_pixel_border:-ignore_pixel_border]
                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))
                if eval_depth:
                    depth_mask = (depth_gt > 0.01).squeeze()
                    nan_mask_1 = ~torch.isnan(depth_gt).squeeze()
                    nan_mask_2 = ~torch.isnan(depths.squeeze())
                    depth_mask = depth_mask & nan_mask_1 & nan_mask_2
                    depthloss = F.l1_loss(
                        depths.squeeze()[depth_mask], depth_gt.squeeze()[depth_mask])
                    metrics["depth_loss"].append(depthloss)

        if world_rank == 0:
            ellipse_time /= len(valloader)

            psnr = torch.stack(metrics["psnr"]).mean()
            ssim = torch.stack(metrics["ssim"]).mean()
            lpips = torch.stack(metrics["lpips"]).mean()
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
            # save stats as json
            stats = {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
            if eval_depth:
                depth_loss = torch.stack(metrics["depth_loss"])
                valid_loss = depth_loss[~torch.isnan(depth_loss)]
                if len(valid_loss) > 0:
                    depth_loss = valid_loss.mean()
                    print(f"Depth loss: {depth_loss.item():.6f}")
                    stats["depth_loss"] = depth_loss.item()
                else:
                    print("All depth loss values are NaN")
                    stats["depth_loss"] = 0.
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
            if self.cfg.use_wandb:
                wandb.log(stats)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if self.parser is not None:
            # Original settings using self.parser
            camtoworlds = self.parser.camtoworlds[5:-5]
            camtoworlds = generate_interpolated_path(
                camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]),
                              len(camtoworlds), axis=0),
                ],
                axis=1,
            )  # [N, 4, 4]

            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = torch.from_numpy(list(self.parser.Ks_dict.values())[
                0]).float().to(device)
            width, height = list(self.parser.imsize_dict.values())[0]

        else:
            # using BlenderDatasetZipped
            camtoworlds = torch.stack([self.valset[i]['camtoworld']
                                       for i in range(len(self.valset))]).to(device)
            # Interpolating poses (if needed) or using as-is
            camtoworlds = generate_interpolated_path(
                camtoworlds.cpu(), 1)
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]),
                              len(camtoworlds), axis=0),
                ],
                axis=1,
            )  # [N, 4, 4]
            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)

            # assume same intrinsics for all frames
            K = self.valset[0]["K"].to(device)
            width, height = self.valset.resize[0] - \
                self.valset.crop_edge, self.valset.resize[1] - \
                self.valset.crop_edge

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i: i + 1],  # Use the current pose
                Ks=K[None],  # Add batch dimension to K
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+D",
            )  # [1, H, W, 4]

            # Process the rendered colors and depths
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() -
                                                # Normalize depth to [0, 1]
                                                depths.min())

            # Concatenate the RGB and depth images side by side
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            # Convert to uint8 for saving
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # Save the results as a video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            # skip GSs that have small image radius (in pixels)
            radius_clip=0.0,
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    # Experimental
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # Experimental
    @torch.no_grad()
    def save_ply(self, path, prune_by_bbox=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.splats["means"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.splats["sh0"].detach().transpose(
            1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(
            1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.splats["opacities"].detach(
        ).unsqueeze(-1).cpu().numpy()
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()

        list_of_attributes = self.construct_list_of_attributes()
        if self.max_sh_degree == 0:
            list_of_attributes = [
                attr for attr in list_of_attributes if 'f_rest' not in attr]
        dtype_full = [(attribute, 'f4') for attribute in list_of_attributes]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.max_sh_degree == 0:
            attributes = np.concatenate(
                (xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate(
                (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # Fill the elements with the attributes
        elements[:] = list(map(tuple, attributes))

        if prune_by_bbox and self.init_bbox_min is not None and self.init_bbox_max is not None:
            bbox_min = [i - 1 for i in self.init_bbox_min]
            bbox_max = [i + 1 for i in self.init_bbox_max]
            mask = (
                (xyz[:, 0] >= bbox_min[0]) & (xyz[:, 0] <= bbox_max[0]) &
                (xyz[:, 1] >= bbox_min[1]) & (xyz[:, 1] <= bbox_max[1]) &
                (xyz[:, 2] >= bbox_min[2]) & (xyz[:, 2] <= bbox_max[2])
            )
            elements = elements[mask]
            print(
                f"Pruned {len(mask) - len(elements)} gaussians by init bounding box.")
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Saving {len(elements)} gaussians to {path}")


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat(
                [ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        # runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=1.0,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    if cfg.max_steps not in cfg.save_steps:
        cfg.save_steps.append(cfg.max_steps)
    if cfg.max_steps not in cfg.eval_steps:
        cfg.eval_steps.append(cfg.max_steps)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
