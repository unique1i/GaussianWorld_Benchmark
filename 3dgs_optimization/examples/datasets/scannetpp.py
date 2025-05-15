import numpy as np
import torch
import json
import torch.utils.data
from pathlib import Path
from PIL import Image
import zipfile
from typing import Literal, Dict, Any
from datasets.colmap_read_write import read_images_text
from utils import compute_intrinsics_matrix

class ScannetppDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: Literal["train", "test", "train+test"] = "train+test",
        load_depth: bool = True,
        exclude_blur_imgs: bool = True,
        zipped: bool = True, 
    ):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)
        self.load_depth = load_depth
        self.zipped = zipped
        self.depth_scale = 1000.0
        meta_path = self.data_root / "dslr/nerfstudio/transforms_undistorted.json"

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"The specified meta file does not exist: {self.data_root}"
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

        self.image_paths = []
        self.poses = []

        self.resize = tuple(self.meta.get("resize", [584, 876]))
        self.resize_ratio = self.resize[1] / self.meta.get("w", 1752)
        self.fx = self.meta["fl_x"] * self.resize_ratio
        self.fy = self.meta["fl_y"] * self.resize_ratio
        self.cx = self.meta["cx"] * self.resize_ratio
        self.cy = self.meta["cy"] * self.resize_ratio

        # Store paths to the zip files but do not open them yet
        self.image_zip_path = self.data_root / "dslr" / "undistorted_images.zip"
        self.depth_zip_path = self.data_root / "dslr" / "undistorted_depths.zip"
        self.zip_ref = None
        self.depth_zip_ref = None

        colmap_dir = self.data_root / "dslr" / "colmap"
        images_txt_path = colmap_dir / "images.txt"

        images = read_images_text(images_txt_path)
        images_name_2_id = {image.name: image.id for image in images.values()}

        if self.split == "train":
            frames_list = self.meta["frames"]
        elif self.split == "test":
            frames_list = self.meta["test_frames"]
        else:
            frames_list = self.meta["frames"] + self.meta["test_frames"]

        for frame in frames_list:
            if exclude_blur_imgs and frame.get("is_bad", False):
                continue
            # Find the image name in the COLMAP images
            frame_name = frame["file_path"]
            image_id = images_name_2_id[frame_name]
            pose = np.eye(4)
            rot = images[image_id].qvec2rotmat()
            pose[:3, :3] = rot
            pose[:3, 3] = images[image_id].tvec
            self.poses.append(np.linalg.inv(pose))
            self.image_paths.append(frame_name)

        self.poses = np.array(self.poses).astype(np.float32)
        self.poses = torch.from_numpy(self.poses)

    def __len__(self):
        return len(self.image_paths)

    def _init_zip_refs(self):
        if not self.zipped:
            return
        if self.zip_ref is None and self.image_zip_path.exists():
            self.zip_ref = zipfile.ZipFile(self.image_zip_path, "r")
        if self.depth_zip_ref is None and self.depth_zip_path.exists():
            self.depth_zip_ref = zipfile.ZipFile(self.depth_zip_path, "r")

    def _load_image(self, image_name: str) -> Image.Image:
        image_name = "undistorted_images/" + image_name
        if self.zipped:
            assert self.image_zip_path.exists(), f"Zip file not found: {self.image_zip_path}"
            self._init_zip_refs()
            with self.zip_ref.open(image_name) as file:
                img = Image.open(file)
                img.load()
        else:
            img_path = self.data_root / "dslr" / image_name
            img = Image.open(img_path)
        return img

    def _load_depth(self, image_name: str):
        image_name = "undistorted_depths/" + image_name
        if self.zipped:
            assert self.depth_zip_path.exists(), f"Zip file not found: {self.depth_zip_path}"
            self._init_zip_refs()
            image_name = image_name.replace("JPG", "png")
            image_name = image_name.replace("images", "depths")
            with self.depth_zip_ref.open(image_name) as file:
                img = Image.open(file)
                img.load()
        else:
            # Handle the case where depth images are not zipped
            image_name = image_name.replace("JPG", "png")
            image_name = image_name.replace("images", "depths")
            img_path = self.data_root / "dslr" / image_name
            img = Image.open(img_path)
        img_tensor = torch.from_numpy(np.array(img)).float()
        img_tensor = img_tensor / self.depth_scale
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0).unsqueeze(0), size=self.resize, mode="nearest"
        ).squeeze()

        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name = self.image_paths[idx]
        img = self._load_image(image_name)

        # Resize the image, note that PIL uses (width, height) while torch uses (height, width)
        resize = (
            (self.resize[1], self.resize[0])
            if self.resize[1] > self.resize[0]
            else (self.resize[0], self.resize[1])
        )
        img = img.resize(resize, Image.ANTIALIAS)
        img = np.array(img.convert("RGB"))  # Ensure RGB format

        # Update the intrinsic matrix
        intrinsics = compute_intrinsics_matrix(self.fx, self.fy, self.cx, self.cy)

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
        # Initialize ZipFile references as None
        self.zip_ref = None
        self.depth_zip_ref = None
