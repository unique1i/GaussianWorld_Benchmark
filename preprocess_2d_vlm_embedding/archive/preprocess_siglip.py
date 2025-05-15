import os
import random
import argparse

import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
import cv2
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from IPython import embed
import json
import torchvision
from torch import nn

from transformers import SiglipModel, SiglipProcessor

CROP_SIZE = 512
@dataclass
class SigLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: SigLIPNetwork)
    # Use the SigLIP model ID from Hugging Face
    clip_model_pretrained: str = "google/siglip-base-patch16-512"  # google/siglip-so400m-patch14-384, google/siglip-base-patch16-224, google/siglip-base-patch16-384
    # SigLIP default embedding dimension for the vision encoder is 768.
    clip_n_dims: int = 768 # 1152
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class SigLIPNetwork(nn.Module):
    def __init__(self, config: SigLIPNetworkConfig):
        super().__init__()
        self.config = config
        # Instead of open_clip.create_model_and_transforms,
        # load the SigLIP model and its processor.
        self.model = SiglipModel.from_pretrained(
            self.config.clip_model_pretrained, torch_dtype=torch.float16
        ).to("cuda")
        self.model.eval()
        self.processor = SiglipProcessor.from_pretrained(
            self.config.clip_model_pretrained
        )
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((CROP_SIZE, CROP_SIZE)),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  # 0.5 used in SigLIP
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            # For text, follow the recommended prompt template.
            pos_texts = [f"This is a photo of {phrase}." for phrase in self.positives]
            pos_inputs = self.processor(
                text=pos_texts, padding="max_length", return_tensors="pt"
            )
            pos_inputs = {k: v.to("cuda") for k, v in pos_inputs.items()}
            self.pos_embeds = self.model.get_text_features(**pos_inputs)

            neg_texts = [f"This is a photo of {phrase}." for phrase in self.negatives]
            neg_inputs = self.processor(
                text=neg_texts, padding="max_length", return_tensors="pt"
            )
            neg_inputs = {k: v.to("cuda") for k, v in neg_inputs.items()}
            self.neg_embeds = self.model.get_text_features(**neg_inputs)

        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert self.pos_embeds.shape[1] == self.neg_embeds.shape[1], (
            "Positive and negative embeddings must have the same dimensionality"
        )
        assert self.pos_embeds.shape[1] == self.clip_n_dims, (
            "Embedding dimensionality must match the model dimensionality"
        )

    @property
    def name(self) -> str:
        return "siglip_{}".format(self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        pos_texts = [f"This is a photo of {phrase}." for phrase in self.positives]
        with torch.no_grad():
            pos_inputs = self.processor(
                text=pos_texts, padding="max_length", return_tensors="pt"
            )
            pos_inputs = {k: v.to("cuda") for k, v in pos_inputs.items()}
            self.pos_embeds = self.model.get_text_features(**pos_inputs)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x embedding_dim
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        # Note: SigLIP uses a pairwise sigmoid rather than softmax.
        # Here we still compute softmax on scaled logits for compatibility with downstream code.
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input):
        # Use the torchvision transforms (or optionally the processor) to preprocess.
        processed_input = self.process(input).half()
        # Use SigLIP’s image encoder: note that get_image_features expects the pixel_values input.
        return self.model.get_image_features(pixel_values=processed_input)


# --------------------------------------------------------------------
# All other functions remain unchanged.
# --------------------------------------------------------------------


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    # Update embed_size to match the SigLIP embedding dimension (768)
    embed_size = model.embedding_dim
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:]))
    mask_generator.predictor.model.to("cuda")

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)

        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat(
                [img_embeds, torch.zeros((len(image_list), pad, embed_size))], dim=1
            )

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed

        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j - 1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j] - 1}"
            v[v != -1] += lengths_cumsum[j - 1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

        save_path = os.path.join(save_folder, data_list[i].split(".")[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {"feature": img_embeds[i, : total_lengths[i]], "seg_maps": seg_maps[i]}
        sava_numpy(save_path, curr)
    mask_generator.predictor.model.to("cpu")


def sava_numpy(save_path, data):
    save_path_s = save_path + "_s.npy"
    save_path_f = save_path + "_f.npy"
    np.save(save_path_s, data["seg_maps"].numpy())
    np.save(save_path_f, data["feature"].numpy())
    print(f"Saved data to {save_path_s} and {save_path_f}")


def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in ["default"]:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()

    return clip_embeds, seg_map


def get_seg_img(mask, image):
    image = image.copy()
    image[mask["segmentation"] == 0] = np.array([0, 0, 0], dtype=np.uint8)
    x, y, w, h = np.int32(mask["bbox"])
    seg_img = image[y : y + h, x : x + w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w, h)
    pad = np.zeros((l, l, 3), dtype=np.uint8)
    if h > w:
        pad[:, (h - w) // 2 : (h - w) // 2 + w, :] = img
    else:
        pad[(w - h) // 2 : (w - h) // 2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros(
        (num_masks,) * 2, dtype=torch.float, device=masks.device
    )
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(
                torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float
            )
            union = torch.sum(
                torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float
            )
            iou = intersection / union
            iou_matrix[i, j] = iou
            if (
                intersection / masks_area[i] < 0.5
                and intersection / masks_area[j] >= 0.85
            ):
                inner_iou = 1 - (intersection / masks_area[j]) * (
                    intersection / masks_area[i]
                )
                inner_iou_matrix[i, j] = inner_iou
            if (
                intersection / masks_area[i] >= 0.85
                and intersection / masks_area[j] < 0.5
            ):
                inner_iou = 1 - (intersection / masks_area[j]) * (
                    intersection / masks_area[i]
                )
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    masks_new = ()
    for masks_lvl in args:
        seg_pred = torch.from_numpy(
            np.stack([m["segmentation"] for m in masks_lvl], axis=0)
        )
        iou_pred = torch.from_numpy(
            np.stack([m["predicted_iou"] for m in masks_lvl], axis=0)
        )
        stability = torch.from_numpy(
            np.stack([m["stability_score"] for m in masks_lvl], axis=0)
        )

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def sam_encoder(image):
    image = cv2.cvtColor(
        image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    masks_default = mask_generator.generate(image)
    masks_default = masks_update(
        masks_default, iou_thr=0.8, score_thr=0.7, inner_thr=0.5
    )[0]

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (CROP_SIZE, CROP_SIZE))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]["segmentation"]] = i
        seg_imgs = np.stack(seg_img_list, axis=0)
        seg_imgs = (
            torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
        ).to("cuda")

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images["default"], seg_maps["default"] = mask2segmap(masks_default, image)
    return seg_images, seg_maps


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def process_single_image(img, data_path, save_folder, i):
    """
    Process a single image and save the outputs (feature + seg_maps)
    to disk as .npy files.
    """
    # Update embed_size to match the SigLIP embedding dimension.
    embed_size = model.embedding_dim

    img_embed, seg_map = _embed_clip_sam_tiles(img, sam_encoder)

    lengths = [len(v) for k, v in img_embed.items()]
    total_length = sum(lengths)

    img_embed_concat = torch.cat([v for k, v in img_embed.items()], dim=0)
    assert img_embed_concat.shape[0] == total_length

    seg_map_tensor_list = []
    lengths_cumsum = lengths[:]
    for j in range(1, len(lengths)):
        lengths_cumsum[j] += lengths_cumsum[j - 1]

    for j, (k, v) in enumerate(seg_map.items()):
        mask_int = torch.from_numpy(v)
        if j != 0:
            assert mask_int.max() == lengths[j] - 1, (
                f"{j}, {mask_int.max()}, {lengths[j] - 1}"
            )
            mask_int[mask_int != -1] += lengths_cumsum[j - 1]
        seg_map_tensor_list.append(mask_int)

    seg_map_tensor = torch.stack(seg_map_tensor_list, dim=0)

    scannetpp_scene_name = data_path.split("/")[-4]
    image_name = data_path.split("/")[-1].split(".")[0]
    save_folder = os.path.join(save_folder, scannetpp_scene_name)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, image_name)
    curr = {"feature": img_embed_concat, "seg_maps": seg_map_tensor}
    sava_numpy(save_path, curr)


def get_selected_image_paths(scene_path):
    data_list = []
    json_path = os.path.join(scene_path, "nerfstudio", "lang_feat_selected_imgs.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        json_data = json.load(f)

    frames_list = json_data.get("frames_list", [])
    undistorted_images_path = os.path.join(scene_path, "undistorted_images")
    for img_name in frames_list:
        img_path = os.path.join(undistorted_images_path, img_name)
        if os.path.exists(img_path):
            data_list.append(img_path)
        else:
            print(f"Warning: Image {img_path} does not exist. Skipping this image.")

    data_list = sorted(data_list)
    return data_list


if __name__ == "__main__":
    seed_num = 709
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="sam2", help="sam2 or sam")
    parser.add_argument("--resolution", type=int, default=-1, help="target width size")
    parser.add_argument(
        "--sam_ckpt_path",
        type=str,
        default="segment_anything/ckpt/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--sam2_ckpt_path", type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    parser.add_argument("--save_folder", type=str, default=None)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    sam2_ckpt_path = args.sam2_ckpt_path
    img_folder = os.path.join(dataset_path, "undistorted_images")
    data_list = get_selected_image_paths(dataset_path)

    # Initialize SigLIP (replacing the OpenCLIP model)
    model = SigLIPNetwork(SigLIPNetworkConfig)

    sam_model_choice = args.model

    if sam_model_choice == "sam2":
        print(f"Using SAM2 model from {sam2_ckpt_path}")
        config_registry = {
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        sam_2_registry = sam2_ckpt_path.replace(".pt", "").split("/")[-1]
        sam2_config = config_registry[sam_2_registry]
        sam2 = build_sam2(sam2_config, sam2_ckpt_path)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
    elif sam_model_choice == "sam":
        print(f"Using SAM model from {sam_ckpt_path}")
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to("cuda")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )

    if not args.save_folder:
        save_folder = dataset_path.replace(
            "/ScanNetpp/data", "/ScanNetpp/language_features"
        ).replace("dslr/", "")
    else:
        save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    mask_generator.predictor.model.to("cuda")

    WARNED = False
    for i, data_path in tqdm(
        enumerate(data_list), desc="Processing images", total=len(data_list)
    ):
        image_path = data_path
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ WARNING ] Could not read {image_path}, skipping.")
            continue

        orig_h, orig_w = image.shape[0], image.shape[1]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print(
                        "[ INFO ] Encountered a large input image (>1080P), rescaling to 1080P. "
                        "If this is not desired, please specify `--resolution=1`."
                    )
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)[None, ...]
        try:
            process_single_image(image_tensor, data_path, save_folder, i)
        except Exception as e:
            if not os.path.exists("failed_images.txt"):
                with open("failed_images.txt", "w") as f:
                    f.write("Index,Data_Path,Error\n")
            with open("failed_images.txt", "a") as f:
                f.write(f"{i},{data_path},{str(e)}\n")
            continue

    mask_generator.predictor.model.to("cpu")
