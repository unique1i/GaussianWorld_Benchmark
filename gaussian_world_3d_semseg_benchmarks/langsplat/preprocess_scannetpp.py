import os
import random
import argparse

import numpy as np
import torch
from tqdm import trange
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type

import torchvision
from torch import nn
import json

try:
    import open_clip
except ImportError:
    assert (
        False
    ), "open_clip is not installed, install it with `pip install open-clip-torch`"


def load_scene_list(val_split_path):
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if line.strip()]
    return scene_ids


GLOBAL_IDX = "0001"
CLIP_TEXT_QUERY = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/embeddings/scanetpp100_text_embeddings_clip.pt"
CLIP_TEXT_QUERY_LOAD = torch.load(CLIP_TEXT_QUERY).cpu().float()  # N x 512
SCANNETPP_100_CLASS = load_scene_list(
    "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/metadata/scannetpp_semseg_top100.txt"
)
# np.loadtxt('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/language_feature_experiments/metadata/scannetpp_semseg_top100.txt', dtype=str)


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.negatives]
            ).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(
            self.config.clip_model_type, self.config.clip_model_pretrained
        )

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size = 512
    seg_maps = []
    total_lengths = []
    timer = 0

    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:]))
    mask_generator.predictor.model.to("cuda")

    for i, img in enumerate(
        image_list
    ):  # tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        print("scene", timer, "/", len(image_list))
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(f"Probelm in {timer}")

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
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j - 1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to("cpu")

    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split(".")[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {"feature": img_embeds[i, : total_lengths[i]], "seg_maps": seg_maps[i]}
        sava_numpy(save_path, curr)
        print("Save to: ", save_path)


def sava_numpy(save_path, data):
    save_path_s = save_path + "_s.npy"
    save_path_f = save_path + "_f.npy"
    np.save(save_path_s, data["seg_maps"].numpy())
    np.save(save_path_f, data["feature"].numpy())


def _embed_clip_sam_tiles(image, sam_encoder, save=True):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in ["default", "s", "m", "l"]:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
            if save:
                tiles_np = tiles.cpu().numpy()
                tiles_np = np.clip(tiles_np * 255, 0, 255).astype(np.uint8)
                print("tiles_np", tiles_np.shape, tiles_np.min(), tiles_np.max())
                for tile_i in range(tiles_np.shape[0]):
                    tile = tiles_np[tile_i]
                    tile = np.transpose(tile, (1, 2, 0))
                    clip_embed_i = clip_embed[tile_i].cpu().float()
                    clip_embed_i = clip_embed_i.unsqueeze(0)
                    dot_lang_text = torch.matmul(
                        clip_embed_i, CLIP_TEXT_QUERY_LOAD.t()
                    )  # (N, C)
                    pred_label = torch.argmax(dot_lang_text, dim=1)
                    print(
                        "pred_label",
                        pred_label,
                        "class ",
                        SCANNETPP_100_CLASS[pred_label],
                    )
                    class_i = SCANNETPP_100_CLASS[pred_label]
                    tile_path = os.path.join(
                        f"./tilt_{GLOBAL_IDX}/", f"{mode}_{tile_i}_{class_i}.png"
                    )
                    cv2.imwrite(tile_path, tile)

        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    raise ValueError("Debug")

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
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

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
            # select mask pairs that may have a severe internal relationship
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

    # If there are no masks with scores above threshold, the top 3 masks are selected
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
    # remove redundant masks based on the scores and overlap rate between masks
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
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = masks_update(
        masks_default,
        masks_s,
        masks_m,
        masks_l,
        iou_thr=0.8,
        score_thr=0.7,
        inner_thr=0.5,
    )

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224, 224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]["segmentation"]] = i
        seg_imgs = np.stack(seg_img_list, axis=0)  # b,H,W,3
        seg_imgs = (
            torch.from_numpy(seg_imgs.astype("float32")).permute(0, 3, 1, 2) / 255.0
        ).to("cuda")

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images["default"], seg_maps["default"] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images["s"], seg_maps["s"] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images["m"], seg_maps["m"] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images["l"], seg_maps["l"] = mask2segmap(masks_l, image)

    # 0:default 1:s 2:m 3:l
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


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/",
    )
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument(
        "--sam_ckpt_path", type=str, default="ckpts/sam_vit_h_4b8939.pth"
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    # get validation folder
    validation_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/splits/nvs_sem_val.txt"
    validation_set = np.loadtxt(validation_path, dtype=str)
    validation_set = sorted(validation_set)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path

    scannetpp_folder = os.listdir(dataset_path)
    # check if all is folder\
    # scannetpp_folder = [f for f in scannetpp_folder if os.path.isdir(os.path.join(dataset_path, f))]
    # scannetpp_folder = sorted(scannetpp_folder)
    dataset_path = "/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/"

    print("Find {} scenes in the dataset folder".format(len(scannetpp_folder)))

    start_idx = args.start_idx
    end_idx = args.end_idx
    if end_idx == -1:
        end_idx = len(validation_set)
    if end_idx > len(validation_set):
        end_idx = len(validation_set)

    validation_set = validation_set[start_idx:end_idx]

    # data_list = os.listdir(img_folder)
    # data_list.sort()

    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
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

    validation_set = ["0d2ee665be"]
    for scene_i in validation_set:
        # if scene_i not in validation_set:
        #     print("Skip training scene: ", scene_i)
        #     continue

        print("Processing validation scene: ", scene_i)
        data_list = os.listdir(
            os.path.join(dataset_path, scene_i, "dslr", "undistorted_images")
        )
        data_list = sorted(data_list)

        img_list = []
        WARNED = False
        scene_root_path = os.path.join(dataset_path, scene_i, "dslr")
        # consider subset of datalists in
        selected_json = os.path.join(
            dataset_path, scene_i, "dslr", "nerfstudio", "lang_feat_selected_imgs.json"
        )
        with open(selected_json, "r") as f:
            selected_data_list = json.load(f)
        selected_frames = selected_data_list["frames"]
        selected_imgs_list = [frame_i["file_path"] for frame_i in selected_frames]

        selected_imgs_name_list = [
            img_i.split(".JPG")[0] for img_i in selected_imgs_list
        ]
        print(
            "selected_imgs_name_list",
            selected_imgs_name_list,
            len(selected_imgs_name_list),
        )
        save_folder = os.path.join(scene_root_path, "language_features_langsplat_debug")

        # shutil.rmtree(save_folder, ignore_errors=True)
        # shutil.rmtree(os.path.join(scene_root_path, 'language_features_dim3'), ignore_errors=True)
        # if os.path.exists(save_folder):
        #     finished_imgs_list = os.listdir(save_folder)
        #     finished_names_list = [img_i.split('_')[0] for img_i in finished_imgs_list]

        #     finished_names_list = np.unique(finished_names_list)
        #     print("finished_names_list", finished_names_list)

        #     selected_imgs_list = list(set(selected_imgs_name_list) - set(finished_names_list))

        #     print("after delete selected_imgs_list", selected_imgs_list)
        #     selected_imgs_list = [img_i + '.JPG' for img_i in selected_imgs_list]
        #     selected_imgs_list = sorted(selected_imgs_list)

        # # for data_path in data_list:
        # #     if data_path.split('.JPG')[0] not in selected_imgs_list:
        # #         # print("Skip training image: ", data_path)
        # #         continue
        for data_path in selected_imgs_list:
            image_path = os.path.join(
                dataset_path, scene_i, "dslr", "undistorted_images", data_path
            )
            if f"DSC0{GLOBAL_IDX}.JPG" not in image_path:
                continue
            print("Adding image: ", data_path)
            image = cv2.imread(image_path)

            # 119 151 158

            orig_w, orig_h = image.shape[1], image.shape[0]
            target_w, target_h = 876, 584
            # [584, 876]
            resolution = (target_w, target_h)
            global_down = orig_w / target_w
            # if args.resolution == -1:
            #     if orig_h > 1080:
            #         if not WARNED:
            #             print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
            #                 "If this is not desired, please explicitly specify '--resolution/-r' as 1")
            #             WARNED = True
            #         global_down = orig_h / 1080
            #     else:
            #         global_down = 1
            # else:
            #     global_down = orig_w / args.resolution

            scale = float(global_down)
            resolution = (int(orig_w / scale), int(orig_h / scale))

            image = cv2.resize(image, resolution)
            image = torch.from_numpy(image)
            img_list.append(image)

        if len(img_list) == 0:
            print("No images left in this scene, skip")
            continue
        images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
        imgs = torch.cat(images)

        # save_folder = os.path.join(dataset_path, 'language_features')

        os.makedirs(save_folder, exist_ok=True)
        chunk_size = 1
        for chunk_start_idx in trange(0, len(imgs), chunk_size):
            print("chunk start", chunk_start_idx, "/", len(imgs))
            chunk_end_idx = min(chunk_start_idx + chunk_size, len(imgs))
            imgs_chunk = imgs[chunk_start_idx:chunk_end_idx]
            data_list_chunk = selected_imgs_list[chunk_start_idx:chunk_end_idx]

            create(imgs_chunk, data_list_chunk, save_folder)

        # if len(imgs) <= 300:
        #     create(imgs, data_list, save_folder)
        # elif len(imgs) <= 600:
        #     print("Split into two parts")
        #     # chunk half half
        #     middle_index = len(imgs) // 2
        #     imgs1 = imgs[:middle_index]
        #     imgs2 = imgs[middle_index:]
        #     data_list1 = data_list[:middle_index]
        #     data_list2 = data_list[middle_index:]
        #     create(imgs1, data_list1, save_folder)
        #     create(imgs2, data_list2, save_folder)
        # elif len(imgs) <= 900:
        #     print("Split into three parts")
        #     # chunk into three parts
        #     middle_index1 = len(imgs) // 3
        #     middle_index2 = len(imgs) // 3 * 2
        #     imgs1 = imgs[:middle_index1]
        #     imgs2 = imgs[middle_index1:middle_index2]
        #     imgs3 = imgs[middle_index2:]
        #     data_list1 = data_list[:middle_index1]
        #     data_list2 = data_list[middle_index1:middle_index2]
        #     data_list3 = data_list[middle_index2:]
        #     create(imgs1, data_list1, save_folder)
        #     create(imgs2, data_list2, save_folder)
        #     create(imgs3, data_list3, save_folder)
        # else:
        #     raise ValueError("Too many images in one scene, please split into smaller parts")
