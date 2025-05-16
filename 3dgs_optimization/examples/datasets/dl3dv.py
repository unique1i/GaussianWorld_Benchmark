from typing import Any, Dict, Literal
import numpy as np
import torch
import json
import torch.utils.data
from pathlib import Path
from PIL import Image
from utils import compute_intrinsics_matrix


def gl2world_to_cv2world(gl2world):
    cv2gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cv2world = gl2world @ cv2gl

    return cv2world


class DL3DVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: Literal["train", "val", "test"],
    ):
        self.meta = {}
        self.split = split
        self.data_root = Path(data_root)
        meta_path = self.data_root / "transforms.json"

        if not self.data_root.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"The specified split file does not exist: {self.data_root}"
            )

        with open(meta_path, "r") as fp:
            self.meta = json.load(fp)

        self.image_paths = []
        self.poses = []

        self.fx = self.meta["fl_x"] / 4
        self.fy = self.meta["fl_y"] / 4
        self.cx = self.meta["cx"] / 4
        self.cy = self.meta["cy"] / 4

        # uniform sampling 10 frames for testing
        test_indices = np.arange(
            0, len(self.meta["frames"]), len(self.meta["frames"]) // 10
        )
        train_indices = set(np.arange(0, len(self.meta["frames"]))) - set(test_indices)

        if self.split == "test" or self.split == "val":
            selected_frames = [self.meta["frames"][i] for i in test_indices]
        else:
            train_indices = list(train_indices)
            selected_frames = [self.meta["frames"][i] for i in train_indices]

        for frame in selected_frames:
            self.image_paths.append(frame["file_path"].replace("images", "images_4"))
            pose_opengl = np.array(frame["transform_matrix"])
            pose_opencv = gl2world_to_cv2world(pose_opengl)
            self.poses.append(pose_opencv)

        self.poses = np.array(self.poses).astype(np.float32)
        self.poses = torch.from_numpy(self.poses)

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, image_name: str) -> Image.Image:
        img_path = self.data_root / image_name
        img = Image.open(img_path)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_name = self.image_paths[idx]
        img = self._load_image(image_name)

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
        return data
