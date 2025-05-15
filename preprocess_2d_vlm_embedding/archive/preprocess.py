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

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


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
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
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
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
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
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)





def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

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
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map


        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)
    mask_generator.predictor.model.to('cpu')

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())
    print(f"Saved {save_path_s.split('/')[-1]}, shape {data['seg_maps'].shape}, {save_path_f.split('/')[-1]}, shape {data['feature'].shape}")

def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    for mode in ['default']:
        tiles = seg_images[mode]
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
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
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
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
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default = masks_update(masks_default, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)[0]

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
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
    embed_size = 512

    # Example call - same as your `_embed_clip_sam_tiles(img, sam_encoder)`
    # But remember that `_embed_clip_sam_tiles` expects a batch dimension, so do:

    img_embed, seg_map = _embed_clip_sam_tiles(img, sam_encoder)

    # Summarize how many embeddings we have
    lengths = [len(v) for k, v in img_embed.items()]
    total_length = sum(lengths)

    # Concatenate all embedding chunks into a single tensor
    img_embed_concat = torch.cat([v for k, v in img_embed.items()], dim=0)  # shape: (total_length, embed_size)
    assert img_embed_concat.shape[0] == total_length

    # Prepare the seg_map (similar to your original code)
    seg_map_tensor_list = []
    lengths_cumsum = lengths[:]
    for j in range(1, len(lengths)):
        lengths_cumsum[j] += lengths_cumsum[j-1]

    for j, (k, v) in enumerate(seg_map.items()):
        mask_int = torch.from_numpy(v)
        if j != 0:
            # shift the mask labels so they don’t collide
            assert mask_int.max() == lengths[j] - 1, f"{j}, {mask_int.max()}, {lengths[j]-1}"
            mask_int[mask_int != -1] += lengths_cumsum[j-1]
        seg_map_tensor_list.append(mask_int)

    seg_map_tensor = torch.stack(seg_map_tensor_list, dim=0)  # shape: (num_tiles, H, W)

    # Save to disk
    if dataset_name == "scannet":
        scannet_scene_name = data_path.split('/')[-3]
        image_name = data_path.split('/')[-1].split('.')[0]
        save_folder = os.path.join(save_folder, scannet_scene_name)
    elif dataset_name == "scannetpp":
        scannetpp_scene_name = data_path.split('/')[-4]
        image_name = data_path.split('/')[-1].split('.')[0]
        save_folder = os.path.join(save_folder, scannetpp_scene_name)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, image_name)
    curr = {
        'feature': img_embed_concat,  # shape (total_length, 512) if you used 512
        'seg_maps': seg_map_tensor    # shape (num_tiles, H, W)
    }
    sava_numpy(save_path, curr)  # your existing function

def get_selected_image_paths(scene_path):
    """
    Reads the frames_list from lang_feat_selected_imgs.json in each scene folder
    and constructs a list of full image paths.

    Args:
        dataset_path (str): path to the dslr folder under a scene folder.

    Returns:
        list: Sorted list of full paths to selected images.
    """
    data_list = []
    # Path to lang_feat_selected_imgs.json
    if dataset_name == "scannetpp":
        images_path = os.path.join(scene_path, "undistorted_images")
        json_path = os.path.join(scene_path, "nerfstudio", "lang_feat_selected_imgs.json")
    elif dataset_name == "scannet":
        images_path = os.path.join(scene_path, "color_interval")
        json_path = os.path.join(scene_path, "lang_feat_selected_imgs.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    # Load the JSON file
    with open(json_path, "r") as f:
        json_data = json.load(f)
    frames_list = json_data.get("frames_list", [])

    for img_name in frames_list:
        img_path = os.path.join(images_path, img_name)
        if os.path.exists(img_path):
            data_list.append(img_path)
        else:
            print(f"Warning: Image {img_path} does not exist. Skipping this image.")

    data_list = sorted(data_list)
    return data_list

if __name__ == '__main__':
    seed_num = 709
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='sam2', help='sam2 or sam')
    parser.add_argument('--resolution', type=int, default=-1, help="target width size")
    parser.add_argument('--sam_ckpt_path', type=str,
                        default="segment_anything/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument('--sam2_ckpt_path', type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument(
        "--dataset_name", 
        type=str,
        required=True,
    )
    args = parser.parse_args()
    global dataset_name
    dataset_name = args.dataset_name
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    sam2_ckpt_path = args.sam2_ckpt_path
    img_folder = os.path.join(dataset_path, 'undistorted_images')
    data_list = get_selected_image_paths(dataset_path)

    # Initialize CLIP
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    
    sam_model_choice  = args.model
    
    if sam_model_choice == 'sam2':
        print(f"Using SAM2 model from {sam2_ckpt_path}")
        config_registry = {
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        }
        sam_2_registry = sam2_ckpt_path.replace('.pt', '').split('/')[-1]
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
    elif sam_model_choice == 'sam':
        print(f"Using SAM model from {sam_ckpt_path}")
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
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

    # Make sure the predictor is on GPU
    mask_generator.predictor.model.to('cuda')

    WARNED = False
    for i, data_path in tqdm(enumerate(data_list), desc='Processing images', total=len(data_list)):
        # --- Load and resize one image ---
        #image_path = os.path.join(img_folder, data_path)
        image_path = data_path
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ WARNING ] Could not read {image_path}, skipping.")
            continue

        orig_h, orig_w = image.shape[0], image.shape[1]
        if args.resolution == -1:
            # optional downscale to 1080p
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered a large input image (>1080P), "
                          "rescaling to 1080P. If this is not desired, "
                          "please specify `--resolution=1`.")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        if args.dataset_name == "scannet":
            resolution = (640, 480)
        elif args.dataset_name == "scannetpp":
            resolution = (876, 584)
        image = cv2.resize(image, resolution)

        # crop edge nby 12 if scannet
        if args.dataset_name == "scannet":
            image = image[12:-12, 12:-12, :]
        print(f"[ INFO ] Rescaled image to {image.shape[1]}x{image.shape[0]}")

        # Convert to tensor, add batch dimension
        # shape becomes [1, 3, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)[None, ...]

        # --- Process a single image; save results ---
        try:
            process_single_image(image_tensor, data_path, save_folder, i)
        except Exception as e:
            # If the file doesn't exist, create it and optionally add a header
            if not os.path.exists("failed_images.txt"):
                with open("failed_images.txt", "w") as f:
                    f.write("Index,Data_Path,Error\n")

            # Append the error info
            with open("failed_images.txt", "a") as f:
                f.write(f"{i},{data_path},{str(e)}\n")

            # Skip this iteration
            continue
        

    # Move model back to CPU if you like
    mask_generator.predictor.model.to('cpu')
