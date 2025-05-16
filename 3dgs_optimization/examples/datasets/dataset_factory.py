import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

from utils import load_points3d
from dataclasses import dataclass
from datasets.colmap import Dataset, Parser

from datasets.blender import BlenderDatasetZipped
from datasets.scannetpp import ScannetppDataset
from datasets.arkitscenes import ARKitScenesDataset
from datasets.matterport3d_region import Matterport3DRegionDataset
from datasets.dl3dv import DL3DVDataset
from datasets.hypersim import HypersimDataset
from datasets.replica import ReplicaDataset
from datasets.objaverse import ObjaverseDataset
from datasets.holicity import HoliCityDataset
from datasets.aria_synthetic_env import AriaSyntheticEnvsDataset


@dataclass
class DatasetHandler:
    parser: Optional[Parser]
    trainset: any
    valset: any
    init_point_num: Optional[int]
    init_bbox_min: Optional[List[float]]
    init_bbox_max: Optional[List[float]]
    points3d_data: any
    scene_scale: float
    scene_name: str = "scene"


def initialize_datasets(cfg) -> Tuple[Optional[Parser], any, any]:
    """
    Initialize training and validation datasets based on the dataset name.

    Args:
        cfg: Configuration object containing dataset settings.

    Returns:
        A tuple containing the parser (if any), training dataset, and validation dataset.
    """
    supported_datasets = [
        "colmap",
        "blender",
        "scannet",
        "scannetpp",
        "scannetpp_v2",
        "arkitscenes",
        "matterport3d_region",
        "dl3dv",
        "replica",
        "hypersim",
        "objaverse",
        "holicity",
        "aria_synthetic_env",
    ]
    if not any(
        [dataset_name in cfg.dataset_name for dataset_name in supported_datasets]
    ):
        raise ValueError(f"Dataset {cfg.dataset_name} not supported.")

    parser = None
    json_path = None
    points3d_data = None
    init_point_num = None
    init_bbox_min = None
    init_bbox_max = None
    scene_scale = 1.0  # modify per dataset, if needed

    mesh_init = False
    if cfg.dataset_name == "colmap":
        # Colmap format
        parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=False,
            test_every=cfg.test_every,
        )
        trainset = Dataset(
            parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        valset = Dataset(parser, split="val")

    elif cfg.dataset_name == "blender" or "scannet_" in cfg.dataset_name:
        # Blender format
        trainset = BlenderDatasetZipped(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = BlenderDatasetZipped(
            data_root=cfg.data_dir,
            split="train",  # eval on all training views
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        if "scannet" in cfg.dataset_name:
            scene_name = Path(cfg.data_dir).name
            points3d_path = Path(cfg.data_dir) / f"{scene_name}_vh_clean_2.ply"
        points3d_data = load_points3d(points3d_path)

    elif "scannetpp" in cfg.dataset_name:
        # Scannetpp, zipped format
        trainset = ScannetppDataset(
            data_root=cfg.data_dir,
            split="train+test",
            zipped=False,  # change accordingly
        )
        valset = ScannetppDataset(data_root=cfg.data_dir, split="test", zipped=False)
        json_path = (
            Path(cfg.data_dir) / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        )
        points3d_path = Path(cfg.data_dir) / "scans" / "mesh_aligned_0.05.ply"
        points3d_data = load_points3d(points3d_path)

    elif "arkitscenes" in cfg.dataset_name:
        # ARKitScenes
        trainset = ARKitScenesDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = ARKitScenesDataset(
            data_root=cfg.data_dir,
            split="test",
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        scene_name = Path(cfg.data_dir).name
        points3d_path = Path(cfg.data_dir) / f"{scene_name}_3dod_mesh.ply"
        points3d_data = load_points3d(points3d_path)

    elif "matterport3d_region" in cfg.dataset_name:
        trainset = Matterport3DRegionDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = Matterport3DRegionDataset(
            data_root=cfg.data_dir,
            split="test",  # randomly sampled from the train set
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        scene_name = Path(cfg.data_dir).name
        points3d_path = Path(cfg.data_dir) / "point3d_fused.ply"
        points3d_data = load_points3d(points3d_path)

    elif "hypersim" in cfg.dataset_name:
        trainset = HypersimDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = HypersimDataset(
            data_root=cfg.data_dir,
            split="val",
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        scene_name = Path(cfg.data_dir).name
        points3d_path = Path(cfg.data_dir) / "fused_pcl.ply"
        points3d_data = load_points3d(points3d_path)

    elif "replica" in cfg.dataset_name:
        trainset = ReplicaDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = ReplicaDataset(
            data_root=cfg.data_dir,
            split="val",
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        scene_name = Path(cfg.data_dir).name
        parent_path = Path(cfg.data_dir).parent
        points3d_path = Path(parent_path) / f"{scene_name}_mesh.ply"
        points3d_data = load_points3d(points3d_path)

    elif "dl3dv" in cfg.dataset_name:
        trainset = DL3DVDataset(data_root=cfg.data_dir, split="train")
        valset = DL3DVDataset(data_root=cfg.data_dir, split="test")
        # trimesh
        json_path = Path(cfg.data_dir) / "transforms.json"
        points3d_path = Path(cfg.data_dir) / "fused.ply"
        points3d_data = load_points3d(points3d_path)

    elif "objaverse" in cfg.dataset_name:
        trainset = ObjaverseDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = ObjaverseDataset(
            data_root=cfg.data_dir,
            split="train",  # eval on all training views
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        points3d_path = Path(cfg.data_dir.replace("renders", "glbs") + ".glb")
        # surface sampling to get strategy_cap_max num of points
        # points3d_data = load_points3d(points3d_path, mesh_input=True, upper_num=cfg.strategy.cap_max, surface_sampling=True)
        points3d_data = load_points3d(
            points3d_path,
            mesh_input=True,
            surface_sampling=False,
            upper_num=50000,
            normalize_scale=True,
        )

    elif "holicity" in cfg.dataset_name:
        trainset = HoliCityDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        valset = HoliCityDataset(
            data_root=cfg.data_dir,
            split="train",
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        points3d_path = Path(cfg.data_dir) / "points3d.ply"
        points3d_data = load_points3d(points3d_path)

    elif "aria_synthetic_env" in cfg.dataset_name:
        trainset = AriaSyntheticEnvsDataset(
            data_root=cfg.data_dir,
            split="train",
            zipped=True,
        )
        valset = AriaSyntheticEnvsDataset(
            data_root=cfg.data_dir,
            split="train",
            zipped=True,
        )
        json_path = Path(cfg.data_dir) / "transforms_train.json"
        points3d_path = Path(cfg.data_dir) / "points3d_fused.ply"
        points3d_data = load_points3d(points3d_path)

    else:
        raise ValueError(f"Dataset {cfg.dataset_name} not supported.")
    assert json_path.exists(), f"Transforms file not found: {json_path}"
    with open(json_path, "r") as f:
        cfg_data = json.load(f)
        init_point_num = cfg_data.get("init_point_num", None)
        init_bbox_min = cfg_data.get("bbox_min", None)
        init_bbox_max = cfg_data.get("bbox_max", None)
        scene_scale = cfg_data.get("scene_scale", 1.0)

    # # temp, save the points3d_data
    # import open3d as o3d
    # if points3d_data is not None:
    #     points3d = o3d.geometry.PointCloud()
    #     points3d.points = o3d.utility.Vector3dVector(points3d_data["points"])
    #     points3d.colors = o3d.utility.Vector3dVector(points3d_data["colors"])
    #     o3d.io.write_point_cloud("init_points3d.ply", points3d)

    if cfg.cap_max_by_init_point_num and init_point_num:
        cap_max = min(round(1.5 * init_point_num / 1_000) * 1_000, cfg.strategy.cap_max)
        cfg.strategy.cap_max = int(cap_max)

    if (
        cfg.strategy.name == "mcmc"
        and points3d_data["points"].shape[0] > cfg.strategy.cap_max
    ):
        # randomly sample points3d_data
        idx = np.random.choice(
            len(points3d_data["points"]), cfg.strategy.cap_max, replace=False
        )
        points3d_data["points"] = points3d_data["points"][idx]
        points3d_data["colors"] = points3d_data["colors"][idx]
        print(
            f"Randomly sampled {cfg.strategy.cap_max}/{len(points3d_data['points'])} points from input points3d_data."
        )

    return DatasetHandler(
        parser=parser,
        trainset=trainset,
        valset=valset,
        init_point_num=init_point_num,
        init_bbox_min=init_bbox_min,
        init_bbox_max=init_bbox_max,
        points3d_data=points3d_data,
        scene_scale=scene_scale,
        scene_name=cfg.data_dir.split("/")[-1],
    )
