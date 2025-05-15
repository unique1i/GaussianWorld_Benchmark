import numpy as np
import torch
import json
import torch.utils.data
from pathlib import Path
from PIL import Image
import zipfile
import cv2
from typing import Literal, Dict, Any
from datasets.colmap_read_write import read_images_text
from utils import compute_intrinsics_matrix

class ObjaverseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: Literal["train"] = "train",
        load_depth: bool = False,
        zipped: bool = True, 
    ):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)
        self.load_depth = load_depth
        self.zipped = zipped
        self.depth_scale = 1.0
        meta_path = self.data_root / "transforms_train.json"

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Meta file does not exist: {meta_path}"
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

        self.image_paths = []
        self.poses = []

        self.fx = self.meta["fl_x"] 
        self.fy = self.meta["fl_y"] 
        self.cx = self.meta["cx"] 
        self.cy = self.meta["cy"] 

        # Store paths to the zip files but do not open them yet
        self.image_zip_path = self.data_root / "image.zip"
        self.depth_zip_path = self.data_root / "depth.zip"
        self.zip_ref = None
        self.depth_zip_ref = None
        frames_list = self.meta["frames"]

        for frame in frames_list:
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            frame_name = frame["file_path"].split("/")[-1]
            pose = np.array(frame["transform_matrix"])
            pose[:3, 1:3] *= -1 

            self.poses.append(pose)
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

    def _load_image(self, image_name: str):
        """Load an image with error handling that returns None on failure"""
        try:
            image_name += ".png"
            if self.zipped:
                assert self.image_zip_path.exists(), f"Zip file not found: {self.image_zip_path}"
                self._init_zip_refs()
                with self.zip_ref.open(image_name) as file:
                    img = Image.open(file)
                    img.load()
                    img = img.convert("RGB")
            return np.array(img)
        except Exception as e:
            # Log the error and return None
            error_msg = f"Error loading image {image_name}: {str(e)}"
            print(f"\033[93mWARNING: {error_msg}\033[0m")  # Yellow warning text
            return None

    def _load_depth(self, image_name: str):
        """Load a depth map with error handling that returns None on failure"""
        try:
            image_name += ".png"
            if self.zipped:
                assert self.depth_zip_path.exists(), f"Zip file not found: {self.depth_zip_path}"
                self._init_zip_refs()
                with self.depth_zip_ref.open(image_name) as file:
                    data = file.read()
                    data_array = np.frombuffer(data, np.uint8)
                    img16 = cv2.imdecode(data_array, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    img = 8.0 * (65535.0 - img16) / 65535.0  # Depth range [0, 8]

            img_tensor = torch.from_numpy(np.array(img)).float()
            return img_tensor
        except Exception as e:
            # Log the error and return None
            error_msg = f"Error loading depth {image_name}: {str(e)}"
            print(f"\033[93mWARNING: {error_msg}\033[0m")  # Yellow warning text
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name = self.image_paths[idx] 
        img = self._load_image(image_name)
        if img is None:
            return {"image_id": idx, "image_name": image_name, "image": None}
        img_tensor = torch.from_numpy(img).float()
        data = {
            "K": compute_intrinsics_matrix(self.fx, self.fy, self.cx, self.cy).clone(),
            "camtoworld": self.poses[idx],
            "image": img_tensor,
            "image_id": idx,
            "image_name": image_name,
        }
        
        if self.load_depth:
            depth = self._load_depth(image_name)
            if depth is None and self.load_depth:
                data["depth"] = None
            else:
                data["depth"] = depth
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
