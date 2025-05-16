import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ARKitScenesDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        load_depth: bool = True,
        resize: Optional[List[int]] = None,
        crop_edge: int = 0,
        resize_mode: str = "bilinear",
        zipped: bool = False,
    ):
        """
        Args:
            scene_path (str): Path to the scene directory.
            split (str): One of ["train", "test", "train+test"].
            load_depth (bool): Whether to load depth maps.
            resize (List[int], optional): Desired image size [width, height]. Defaults to [640, 480].
            crop_edge (int): Number of pixels to crop from each edge after resizing.
            resize_mode (str): Resizing mode for depth maps. Options include "nearest", "bilinear", etc.
        """
        super().__init__()
        self.scene_path = Path(data_root)
        self.split = split
        self.load_depth = load_depth
        self.crop_edge = crop_edge
        self.resize_mode = resize_mode
        self.zipped = zipped
        self.resize = (
            resize if resize is not None else [960, 720]
        )  # original highres: 1920x1440
        self.frames = []  # List of dicts containing frame info
        self.wide_zip = None
        self.highres_depth_zip = None
        self.wide_intrinsics_zip = None

        # Verify that scene_path exists and is a directory
        if not self.scene_path.exists() or not self.scene_path.is_dir():
            raise FileNotFoundError(
                f"The specified scene path does not exist or is not a directory: {data_root}"
            )

        # Check for transforms_train.json
        transforms_path = self.scene_path / "transforms_train.json"
        if not transforms_path.exists():
            raise FileNotFoundError(
                f"transforms_train.json not found in {self.scene_path}"
            )

        # Read transforms_train.json
        with open(transforms_path, "r") as f:
            transforms = json.load(f)
        self.resize_ratio = self.resize[0] / transforms["width"]

        # Check for presence of zipped folders
        wide_zip_path = self.scene_path / "wide.zip"
        highres_depth_zip_path = self.scene_path / "highres_depth.zip"
        wide_intrinsics_zip_path = self.scene_path / "wide_intrinsics.zip"

        if self.zipped:
            is_zipped = (
                wide_zip_path.exists()
                and highres_depth_zip_path.exists()
                and wide_intrinsics_zip_path.exists()
            )
            assert (
                is_zipped
            ), f"Zipped files not found in the directory {self.scene_path}"
            self.wide_zip_path = wide_zip_path
            self.highres_depth_zip_path = highres_depth_zip_path
            self.wide_intrinsics_zip_path = wide_intrinsics_zip_path
        else:
            wide_dir = self.scene_path / "wide"
            highres_depth_dir = self.scene_path / "highres_depth"
            wide_intrinsics_dir = self.scene_path / "wide_intrinsics"

            if not (
                wide_dir.exists()
                and highres_depth_dir.exists()
                and wide_intrinsics_dir.exists()
            ):
                raise FileNotFoundError(
                    f"Required directories not found in {self.scene_path} for unzipped data."
                )

        # Iterate through frames based on split
        if self.split == "train":
            frames_list = transforms.get("frames", [])
        elif self.split == "test":
            frames_list = transforms.get("test_frames", [])
        elif self.split == "train+test":
            frames_list = transforms.get("frames", []) + transforms.get(
                "test_frames", []
            )
        else:
            raise ValueError(
                f"Invalid split: {self.split}. Must be one of ['train', 'test', 'train+test']."
            )

        # Collect frame information
        for frame in frames_list:
            # Handle intrinsics
            if not transforms.get("share_intrinsics", True):
                fx = frame.get("fx", None)
                fy = frame.get("fy", None)
                cx = frame.get("cx", None)
                cy = frame.get("cy", None)
                if fx is None or fy is None or cx is None or cy is None:
                    print(
                        f"Intrinsics missing for frame {frame.get('file_path')}. Skipping this frame."
                    )
                    continue
            else:
                # If intrinsics are shared, use top-level values
                fx = transforms.get("fx", 0.0)
                fy = transforms.get("fy", 0.0)
                cx = transforms.get("cx", 0.0)
                cy = transforms.get("cy", 0.0)

            # Construct relative image path
            image_rel_path = frame["file_path"]  # e.g., "wide/41048190_3419.682.png"

            # Construct relative depth path if needed
            depth_path = None
            if self.load_depth:
                # Assuming depth images are stored in "highres_depth" folder with same filename
                depth_rel_path = image_rel_path.replace("wide/", "highres_depth/")
                depth_path = depth_rel_path

            # Get transform matrix
            transform_matrix = frame.get("transform_matrix", None)
            if transform_matrix is None:
                continue

            # Append frame info to the list
            self.frames.append(
                {
                    "image_path": image_rel_path,
                    "depth_path": depth_path,
                    "transform_matrix": np.array(transform_matrix, dtype=np.float32),
                    "intrinsics": {
                        "fx": float(fx),
                        "fy": float(fy),
                        "cx": float(cx),
                        "cy": float(cy),
                    },
                }
            )

        print(f"{self.split} frames loaded: {len(self.frames)}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single data point.

        Returns:
            Dict[str, Any]: A dictionary containing the image, depth (if loaded), pose, intrinsics, and metadata.
        """
        frame = self.frames[idx]
        image_rel_path = frame["image_path"]
        depth_rel_path = frame["depth_path"]
        pose = frame["transform_matrix"]
        intrinsics = frame["intrinsics"]
        zipped = self.zipped
        if zipped and self.wide_zip is None:
            self._init_zip_refs()

        # Load image
        img = self._load_image(image_rel_path, zipped)
        img = img.resize((self.resize[0], self.resize[1]), Image.ANTIALIAS)
        if self.crop_edge > 0:
            img = img.crop(
                (
                    self.crop_edge,
                    self.crop_edge,
                    img.width - self.crop_edge,
                    img.height - self.crop_edge,
                )
            )
        img = np.array(img.convert("RGB"))
        img_tensor = torch.from_numpy(img).float()

        # Scale intrinsics according to resize ratio
        scaled_fx = intrinsics["fx"] * self.resize_ratio
        scaled_fy = intrinsics["fy"] * self.resize_ratio
        scaled_cx = intrinsics["cx"] * self.resize_ratio
        scaled_cy = intrinsics["cy"] * self.resize_ratio

        depth_tensor = None
        if self.load_depth and depth_rel_path is not None:
            depth_tensor = self._load_depth(depth_rel_path, zipped)
            if (
                depth_tensor.shape[0] != self.resize[1]
                or depth_tensor.shape[1] != self.resize[0]
            ):
                depth_tensor = torch.nn.functional.interpolate(
                    depth_tensor.reshape(1, 1, *depth_tensor.shape),
                    size=(self.resize[1], self.resize[0]),
                    mode="nearest",
                ).squeeze()
            if self.crop_edge > 0:
                depth_tensor = depth_tensor[
                    :,
                    self.crop_edge : -self.crop_edge,
                    self.crop_edge : -self.crop_edge,
                ]

        # Construct intrinsic matrix K
        K = torch.tensor(
            [[scaled_fx, 0.0, scaled_cx], [0.0, scaled_fy, scaled_cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # Prepare the data dictionary
        data = {
            "image": img_tensor,  # C x H x W, range [0, 255]
            "camtoworld": torch.from_numpy(pose),  # 4 x 4
            "K": K,  # 3 x 3 tensor
            "image_id": idx,
            "image_name": image_rel_path,
        }

        if self.load_depth:
            data["depth"] = depth_tensor  # H x W

        return data

    def _init_zip_refs(self):
        """
        Initialize zip file references for zipped scenes.
        This method should be called within each worker process.
        """
        if self.wide_zip is not None:
            return  # Already initialized

        try:
            self.wide_zip = zipfile.ZipFile(self.wide_zip_path, "r")
            self.highres_depth_zip = zipfile.ZipFile(self.highres_depth_zip_path, "r")
            self.wide_intrinsics_zip = zipfile.ZipFile(
                self.wide_intrinsics_zip_path, "r"
            )
        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(
                f"Failed to open zip files in {self.scene_path}: {e}"
            )

    def _load_image(self, image_rel_path: str, zipped: bool) -> Image.Image:
        """
        Load an image either from a zip file or the filesystem.

        Args:
            image_rel_path (str): Relative path to the image.
            zipped (bool): Whether the scene's data is zipped.

        Returns:
            Image.Image: Loaded PIL Image in RGB mode.
        """
        if zipped:
            # Load image from zip
            if self.wide_zip is None:
                self._init_zip_refs()
            try:
                with self.wide_zip.open(image_rel_path) as img_file:
                    img = Image.open(img_file).convert("RGB")
            except KeyError:
                raise FileNotFoundError(f"Image {image_rel_path} not found in zip.")
        else:
            # Load image from filesystem
            image_path = self.scene_path / image_rel_path
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            img = Image.open(image_path).convert("RGB")
        return img

    def _load_depth(self, depth_rel_path: str, zipped: bool) -> torch.Tensor:
        """
        Load a depth map either from a zip file or the filesystem.

        Args:
            depth_rel_path (str): Relative path to the depth image.
            zipped (bool): Whether the scene's data is zipped.

        Returns:
            torch.Tensor: Depth map tensor in meters, shape [1, H, W].
        """
        if zipped:
            # Load depth from zip
            if self.highres_depth_zip is None:
                self._init_zip_refs()
            try:
                with self.highres_depth_zip.open(depth_rel_path) as depth_file:
                    depth_img = Image.open(depth_file).convert(
                        "I"
                    )  # 32-bit signed integer pixels
            except KeyError:
                raise FileNotFoundError(
                    f"Depth image {depth_rel_path} not found in zip."
                )
        else:
            # Load depth from filesystem
            depth_path = self.scene_path / depth_rel_path
            if not depth_path.exists():
                raise FileNotFoundError(f"Depth file not found: {depth_path}")
            depth_img = Image.open(depth_path).convert("I")

        # Convert depth to tensor
        depth_np = np.array(depth_img).astype(
            np.float32
        )  # Assuming depth in millimeters
        depth_np = depth_np / 1000.0  # Convert to meters
        depth_tensor = torch.from_numpy(depth_np).squeeze()  # Shape: [1, H, W]

        return depth_tensor

    def __getstate__(self):
        """
        Customize pickling behavior to exclude zipfile references.
        """
        state = self.__dict__.copy()
        # Remove zipfile references to avoid pickling issues
        state["wide_zip"] = None
        state["highres_depth_zip"] = None
        state["wide_intrinsics_zip"] = None
        return state

    def __setstate__(self, state):
        """
        Restore state and ensure zipfile references are reopened if needed.
        """
        self.__dict__.update(state)
        # Zip files will be reopened lazily in each worker
        self.wide_zip = None
        self.highres_depth_zip = None
        self.wide_intrinsics_zip = None

    def close(self):
        """
        Close all open zip files.
        """
        if self.wide_zip is not None:
            self.wide_zip.close()
            self.wide_zip = None
        if self.highres_depth_zip is not None:
            self.highres_depth_zip.close()
            self.highres_depth_zip = None
        if self.wide_intrinsics_zip is not None:
            self.wide_intrinsics_zip.close()
            self.wide_intrinsics_zip = None

    def __del__(self):
        """
        Ensure all zip files are closed upon deletion.
        """
        self.close()
