import os
import random
import argparse
import numpy as np
import torch
import cv2
import json

from tqdm import tqdm
from model import SigLIPNetwork, SigLIPNetworkConfig, CROP_SIZE, OpenCLIPNetwork
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# ================================================================================================
# this file use the fusion strategy proposed by HOV-SG paper:

# "for each 2D segment, we crop 2 images:
# one masking the rest of the image out,
# and another one with the minimum bounding box that contains the full mask (see an example in Fig. 3 ).
# We then compute CLIP vectors for the full keyframe and for the two images that we cropped,
# and the final CLIP vector for the 2D segment is the result of fusing the 3 of them."
# --------------------------------------------------------------------


def create(image_list, data_list, save_folder, use_highlight=False):
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
            img_embed, seg_map = _embed_clip_sam_tiles(
                img.unsqueeze(0), sam_encoder, use_highlight=use_highlight
            )
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
    print(
        f"Saved {save_path_s.split('/')[-1]}, shape {data['seg_maps'].shape}, {save_path_f.split('/')[-1]}, shape {data['feature'].shape}"
    )


def _embed_clip_sam_tiles(
    image, sam_encoder_func, use_highlight=False, highlight_more=False
):
    # Concatenate input image
    aug_imgs = torch.cat([image])  # Shape: (1, 3, H, W)
    seg_images, seg_map = sam_encoder_func(aug_imgs, use_highlight=use_highlight)

    global use_clip
    # we only encode the masked crops

    # Extract full-image CLIP feature (F_g)
    if not use_clip:
        with torch.no_grad():
            F_g = model.encode_image((aug_imgs).to("cuda"))
            F_g = F_g / F_g.norm(dim=-1, keepdim=True)

    # Process crops in batches
    def process_batch(crops):
        """Helper function to process a batch of crops"""
        num_crops = crops.shape[0]
        features = []
        for i in range(0, num_crops, batch_size):
            batch = crops[i : i + batch_size].to("cuda")
            with torch.no_grad():
                batch_feats = model.encode_image(batch)
            features.append(batch_feats.cpu())
            del batch, batch_feats  # Free GPU memory
            torch.cuda.empty_cache()
        return torch.cat(features, dim=0)

    # Extract features for background and masked crops
    fm = process_batch(seg_images["default"]["masked"]).cuda()
    fm = fm / fm.norm(dim=-1, keepdim=True)
    if use_clip:
        # only process the masked crops if use clip
        # print("processing masked crops only...")
        return {"default": fm.detach().cpu().half()}, seg_map
    fl = process_batch(seg_images["default"]["background"]).cuda()
    fl = fl / fl.norm(dim=-1, keepdim=True)

    # -----------------------------------------------------------------------
    # Dynamic weighting based between fl and fm
    sim_fl_fm = torch.nn.functional.cosine_similarity(
        fl, fm, dim=-1, eps=1e-8
    )  # Shape: (num_masks,)
    # If fl and fm are very similar (sim close to 1), the contribution of fm is high.
    # If they are dissimilar (sim is lower), we put more weight on crop_w_bg.
    dynamic_masked_weight = sim_fl_fm.unsqueeze(-1)  # Shape: (num_masks, 1)
    # -----------------------------------------------------------------------

    # Fuse fl and fm using the dynamic weight
    F_l = dynamic_masked_weight * fm + (1 - dynamic_masked_weight) * fl
    F_l = torch.nn.functional.normalize(F_l, p=2, dim=-1).cuda()

    # Compute dot product between F_l and F_g for each mask
    F_g_expanded = F_g.expand(F_l.shape[0], -1)  # Shape: (num_masks, 1152)
    cos = torch.nn.CosineSimilarity(dim=-1)
    phi_l_G = cos(F_l, F_g_expanded)

    # Compute similarity scores for the full image and the crops
    # (w_i now indicates how similar the fused crop is to the full image)
    w_i = torch.softmax(phi_l_G, dim=0).unsqueeze(-1)  # Shape: (num_masks, 1)

    # Compute the three weights for fusing the features:
    # - wg for the full image,
    # - wm for the masked crop,
    # - wl for the background crop.
    wg = w_i
    wm = (1 - w_i) * dynamic_masked_weight
    wl = (1 - w_i) * (1 - dynamic_masked_weight)
    if highlight_more:
        wm, wl = wl, wm  # prioritize the background crop more

    # Fuse all three features: F_p = wg*F_g + wl*fl + wm*fm
    F_p = wg * F_g_expanded + wl * fl + wm * fm
    F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)

    # Print out diagnostic information
    print(
        f"Average weights | full_img: {wg.mean().item():.3f}, "
        f"crop_w_bg: {wl.mean().item():.3f}, "
        f"crop_masked: {wm.mean().item():.3f}"
    )
    print(
        f"Average cosine similarity between crop_w_bg and crop_masked: {sim_fl_fm.mean().item():.3f}"
    )

    # Return the fused CLIP embeddings in half precision
    clip_embeds = {"default": F_p.detach().cpu().half()}
    return clip_embeds, seg_map


def get_bbox_crop(mask, image):
    """Crop the image using the mask's bounding box, retaining background."""
    x, y, w, h = np.int32(mask["bbox"])
    return image[y : y + h, x : x + w, ...]


def get_seg_img(mask, image, mode="mask", pad=25):
    """
    Get the segmented image patch, based on the given mask and image.

    Parameters:
    - mask: A dictionary containing the segmentation mask information.
    - image: The original image.
    - mode: The mode for cropping. Options: "mask" (default), "highlight"
    - pad: The padding value for highlight mode. Default is 25.

    Returns:
    - The processed image segment
    """
    if mode == "mask":
        # Original behavior - mask out background
        image = image.copy()
        image[mask["segmentation"] == 0] = np.array([0, 0, 0], dtype=np.uint8)
        x, y, w, h = np.int32(mask["bbox"])
        seg_img = image[y : y + h, x : x + w, ...]
        return seg_img

    elif mode == "highlight":
        image = image.copy()
        # Make background slightly translucent
        image[mask["segmentation"] == 0] = (
            (image[mask["segmentation"] == 0].astype(np.float32) + 2 * 255) / 3
        ).astype(np.uint8)

        # Crop with padding
        x, y, w, h = np.int32(mask["bbox"])
        seg_img = image[
            max(0, y - pad) : min(image.shape[0], y + h + pad),
            max(0, x - pad) : min(image.shape[1], x + w + pad),
            ...,
        ]

        # Create mask for this cropped region
        seg_mask = (mask["segmentation"] == 0)[
            max(0, y - pad) : min(image.shape[0], y + h + pad),
            max(0, x - pad) : min(image.shape[1], x + w + pad),
        ]

        # Find and draw red contour
        contours, _ = cv2.findContours(
            (255 - seg_mask * 255).astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        seg_img = cv2.drawContours(
            seg_img, contours, -1, (255, 0, 0), 3
        )  # Red outline (BGR)

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


def sam_encoder(image, use_highlight=False):
    image = cv2.cvtColor(
        image[0].permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    masks_default = mask_generator.generate(image)
    masks_default = masks_update(
        masks_default, iou_thr=0.8, score_thr=0.7, inner_thr=0.5
    )[0]

    def mask2segmap(masks, image):
        seg_img_list_masked = []
        seg_img_list_background = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)

        for i, mask in enumerate(masks):
            # Masked crop (background black)
            seg_img_masked = get_seg_img(mask, image, mode="mask")
            pad_seg_img_masked = cv2.resize(
                pad_img(seg_img_masked), (CROP_SIZE, CROP_SIZE)
            )
            seg_img_list_masked.append(pad_seg_img_masked)

            # Crop with background - use highlight mode if enabled
            if use_highlight:
                seg_img_background = get_seg_img(mask, image, mode="highlight")
            else:
                seg_img_background = get_bbox_crop(mask, image)

            pad_seg_img_background = cv2.resize(
                pad_img(seg_img_background), (CROP_SIZE, CROP_SIZE)
            )
            seg_img_list_background.append(pad_seg_img_background)

            seg_map[mask["segmentation"]] = i

        # Convert to PyTorch tensors
        seg_imgs_masked = (
            torch.from_numpy(
                np.stack(seg_img_list_masked, axis=0).astype("float32")
            ).permute(0, 3, 1, 2)
        ).to("cuda")
        seg_imgs_background = (
            torch.from_numpy(
                np.stack(seg_img_list_background, axis=0).astype("float32")
            ).permute(0, 3, 1, 2)
        ).to("cuda")

        return {"masked": seg_imgs_masked, "background": seg_imgs_background}, seg_map

    # check the size each mask in masks_default
    masks_default = [m for m in masks_default if m["area"] > 10]
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


def process_single_image(
    img, data_path, save_folder, i, use_highlight=False, highlight_more=False
):
    """
    Process a single image and save the outputs (feature + seg_maps)
    to disk as .npy files.
    """
    # Update embed_size to match the SigLIP embedding dimension.
    embed_size = model.embedding_dim

    img_embed, seg_map = _embed_clip_sam_tiles(
        img, sam_encoder, use_highlight=use_highlight, highlight_more=highlight_more
    )

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
            assert (
                mask_int.max() == lengths[j] - 1
            ), f"{j}, {mask_int.max()}, {lengths[j] - 1}"
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
    parser.add_argument("--resize", type=tuple)
    parser.add_argument("--crop_edge", type=int, default=0)
    parser.add_argument("--use_clip", action="store_true")
    parser.add_argument(
        "--sam_ckpt_path",
        type=str,
        default="segment_anything/ckpt/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--sam2_ckpt_path", type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    parser.add_argument("--save_folder", type=str, default=None)
    # Add the highlight argument
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="Use highlight mode for background crops",
    )
    parser.add_argument(
        "--highlight_more", action="store_true", help="Prioritize background crop more"
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    # Define batch_size for crop processing
    batch_size = 32

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    sam2_ckpt_path = args.sam2_ckpt_path
    img_folder = os.path.join(dataset_path, "undistorted_images")
    data_list = get_selected_image_paths(dataset_path)

    # Initialize SigLIP (replacing the OpenCLIP model)
    global use_clip
    use_clip = args.use_clip
    if args.use_clip:
        print("Using OpenCLIP model")
        model = OpenCLIPNetwork(device="cuda")
    else:
        print("Using SigLIP model")
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

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    use_highlight = args.highlight
    highlight_more = args.highlight_more
    if use_highlight:
        print("Using highlight mode for background crops")

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
            process_single_image(
                image_tensor,
                data_path,
                save_folder,
                i,
                use_highlight=use_highlight,
                highlight_more=highlight_more,
            )
        except Exception as e:
            print(f"\nFailed to process image {data_path}: {str(e)}")
            if not os.path.exists("failed_images.txt"):
                with open("failed_images.txt", "w") as f:
                    f.write("Index,Data_Path,Error\n")
            with open("failed_images.txt", "a") as f:
                f.write(f"Error when processing scene {dataset_path}\n")
                f.write(f"{i},{data_path},{str(e)}\n")
            raise e

    mask_generator.predictor.model.to("cpu")
