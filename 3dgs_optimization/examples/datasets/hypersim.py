import numpy as np
import torch
import json
import torch.utils.data
from pathlib import Path
from PIL import Image
import zipfile
from typing import Literal, Dict, Any
from utils import compute_intrinsics_matrix


class HypersimDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: Literal["train", "val", "test", "all"],
        load_depth: bool = True,
    ):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)
        self.load_depth = load_depth
        self.depth_scale = 1.0
        meta_path = self.data_root / f"transforms_{split}.json"

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"The specified split file does not exist: {self.data_root}"
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

        self.image_paths = []
        self.poses = []
        self.crop_edge = self.meta.get("crop_edge", 0)
        self.resize = [0, 0]
        # self.resize = tuple(self.meta.get("resize", [640, 480]))
        self.fx = self.meta["fl_x"]
        self.fy = self.meta["fl_y"]
        self.cx = self.meta["cx"]
        self.cy = self.meta["cy"]
        if self.crop_edge > 0:
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        if self.meta["zipped"]:
            self.image_zip_path = self.data_root / "images_zip.zip"
            self.depth_zip_path = self.data_root / "depth_zip.zip"
            # Initialize ZipFile references to None; they will be opened in each worker
            self.zip_ref = None
            self.depth_zip_ref = None
        else:
            self.image_zip_path = None
            self.depth_zip_path = None
            self.zip_ref = None
            self.depth_zip_ref = None

        for frame in self.meta["frames"]:
            self.image_paths.append(frame["file_path"])
            self.poses.append(np.array(frame["transform_matrix"]))

        self.poses = np.array(self.poses).astype(np.float32)
        self.poses = torch.from_numpy(self.poses)

    def __len__(self):
        return len(self.image_paths)

    def _init_zip_refs(self):
        if self.meta["zipped"]:
            if self.zip_ref is None and self.image_zip_path is not None:
                self.zip_ref = zipfile.ZipFile(self.image_zip_path, "r")
            if self.depth_zip_ref is None and self.depth_zip_path is not None:
                self.depth_zip_ref = zipfile.ZipFile(self.depth_zip_path, "r")

    def _load_image(self, image_name: str) -> Image.Image:
        if self.image_zip_path:
            self._init_zip_refs()
            image_name = image_name.split("/")[-1]
            with self.zip_ref.open(image_name) as file:
                img = Image.open(file)
                img.load()
        else:
            img_path = self.data_root / image_name
            img = Image.open(img_path)
        return img

    def _load_depth(self, image_name: str):
        if self.depth_zip_path:
            self._init_zip_refs()
            image_name = image_name.split("/")[-1]
            image_name = image_name.replace("png", "npy")
            # print("depth name", image_name)
            with self.depth_zip_ref.open(image_name) as file:
                img = np.load(file)

                # img = Image.open(file)
                # img.load()
        else:
            # Handle the case where depth images are not zipped
            img_path = self.data_root / image_name.replace("png", "npy")
            img = Image.open(img_path)
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor / self.depth_scale
        if self.crop_edge > 0:
            img_tensor = img_tensor[
                self.crop_edge : -self.crop_edge, self.crop_edge : -self.crop_edge
            ]

        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name = self.image_paths[idx]
        img = self._load_image(image_name)

        # Resize the image
        if self.resize[0] > 0 and self.resize[1] > 0:
            resize = (
                (self.resize[1], self.resize[0])
                if self.resize[1] > self.resize[0]
                else (self.resize[0], self.resize[1])
            )
            img = img.resize(resize, Image.ANTIALIAS)

        # Crop the image if crop_edge is specified
        if self.crop_edge > 0:
            img = img.crop(
                (
                    self.crop_edge,
                    self.crop_edge,
                    img.width - self.crop_edge,
                    img.height - self.crop_edge,
                )
            )

        img = np.array(img.convert("RGB"))  # Ensure RGB format

        # Update the intrinsic matrix
        intrinsics = compute_intrinsics_matrix(self.fx, self.fy, self.cx, self.cy)

        # Convert the image to tensor
        img_tensor = torch.from_numpy(img).float()

        data = {
            "K": intrinsics.clone(),
            "camtoworld": self.poses[idx],
            "image": img_tensor,
            "image_id": idx,
            "image_name": image_name,
        }
        if self.load_depth:
            data["depth"] = self._load_depth(image_name)

        return data

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not pickle ZipFile references
        state["zip_ref"] = None
        state["depth_zip_ref"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize ZipFile references to None
        self.zip_ref = None
        self.depth_zip_ref = None
