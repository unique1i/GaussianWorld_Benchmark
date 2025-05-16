import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import hsv_to_rgb


dataset_root = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/"

segment_class_names = np.loadtxt(
    Path(dataset_root) / "metadata" / "semantic_benchmark" / "top100.txt",
    dtype=str,
    delimiter=".",  # dummy delimiter to replace " "
)


def generate_distinct_colors(n=100, seed=42):
    np.random.seed(seed)

    # Evenly space hues, randomize saturation and value a bit
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)  # shuffle to prevent similar colors being close in order
    saturations = np.random.uniform(0.6, 0.9, n)
    values = np.random.uniform(0.7, 0.95, n)

    hsv_colors = np.stack([hues, saturations, values], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors


# Example usage
SCANNET_100_COLORS = generate_distinct_colors(100)
CLASS_LABELS_100 = segment_class_names


# from metadata.scannet200_constants import CLASS_LABELS_20

# SCANNET_20_COLORS = np.array([
#     [174, 199, 232],  # wall
#     [152, 223, 138],  # floor
#     [31, 119, 180],   # cabinet
#     [255, 187, 120],  # bed
#     [188, 189, 34],   # chair
#     [140, 86, 75],    # sofa
#     [255, 152, 150],  # table
#     [214, 39, 40],    # door
#     [197, 176, 213],  # window
#     [148, 103, 189],  # bookshelf
#     [196, 156, 148],  # picture
#     [23, 190, 207],   # counter
#     [247, 182, 210],  # desk
#     [219, 219, 141],  # curtain
#     [255, 127, 14],   # refrigerator
#     [227, 119, 194],  # shower curtain
#     [158, 218, 229],  # toilet
#     [44, 160, 44],    # sink
#     [112, 128, 144],  # bathtub
#     [82, 84, 163],    # other furniture
# ], dtype=np.uint8)  # shape: (20, 3)
# # Convert to [0,1] float range for matplotlib
# SCANNET_20_COLORS = SCANNET_20_COLORS / 255.0

import open_clip


@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,  # shape: (N, 512)
    text_feat: torch.Tensor,  # shape: (C, 512)
    canon_feat: torch.Tensor,  # shape: (K, 512)
    device: torch.device,
    use_dot_similarity: bool = False,
):
    """
    Computes predicted labels for each of the N language features using one of two
    methods:

    1. Ratio method (default, identical to original code):
       Score_c = min_i [ exp(lang . text_c) / (exp(lang . canon_i) + exp(lang . text_c)) ]
       where i indexes the canonical phrases.

    2. Dot‑similarity method (if ``use_dot_similarity`` is ``True``):
       Simply picks the text embedding with the highest CLIP dot similarity.

    Returns
    -------
    pred_label : ndarray, shape (N,)
        The predicted label indices in [0..C‑1].
    """

    # Move to device
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    canon_feat = canon_feat.to(device, non_blocking=True)

    # Fast path: plain dot‑product similarity
    if use_dot_similarity:
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        pred_label = torch.argmax(dot_lang_text, dim=1)
        return pred_label.cpu().numpy()

    # Original ratio‑based relevancy score
    dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
    dot_lang_canon = torch.matmul(lang_feat, canon_feat.t())  # (N, K)

    exp_lang_text = dot_lang_text.exp()  # (N, C)
    exp_lang_canon = dot_lang_canon.exp()  # (N, K)

    N, C = dot_lang_text.shape

    relevancy_scores = []
    for c_idx in range(C):
        text_c_exp = exp_lang_text[:, c_idx].unsqueeze(-1)  # (N,1)
        ratio_c = text_c_exp / (exp_lang_canon + text_c_exp)  # (N, K)
        score_c = torch.min(ratio_c, dim=1).values  # (N,)
        relevancy_scores.append(score_c)

    relevancy_matrix = torch.stack(relevancy_scores, dim=0).t()  # (N, C)
    pred_label = torch.argmax(relevancy_matrix, dim=1)  # (N,)
    return pred_label.cpu().numpy()


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize segmentation features")
    parser.add_argument(
        "--feat_path", type=str, required=True, help="Path to the feature files"
    )
    parser.add_argument(
        "--feature_level",
        type=str,
        required=True,
        help="Path to the autoencoder checkpoint",
    )
    # parser.add_argument('--save_feat_path', type=str, required=True, help='Path to save the features')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    feat_path = args.feat_path
    # scene_folder = "/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/0d2ee665be/dslr/"
    # feat_name = 'language_features_multi_clip_dim512_restore' #'language_features_multi_clip_dim512_restore' # 'language_features_clip_dim3'
    feat_name = feat_path.split("/")[-2]
    scene_folder = os.path.dirname(os.path.dirname(feat_path))
    print("scene_folder: ", scene_folder)
    print("feat_name: ", feat_name)
    feature_level = int(args.feature_level)
    save_name = f"{feat_name}_level_{feature_level}_seg"
    # text_emb_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/clip_text_embeddings_100.pth"
    # text_embeds = torch.load(text_emb_path)  # shape (20, 768)
    # assert text_embeds.shape[-1] == 512
    # text_embeds_np = text_embeds.detach().cpu().numpy()  # shape (100, 768)

    label_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt"
    # text_emb_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/clip_text_embeddings_100.pth"
    # text_embeds = torch.load(text_emb_path)  # shape (20, 768)
    # assert text_embeds.shape[-1] == 512
    # text_embeds_np = text_embeds.detach().cpu().numpy()  # shape (100, 768)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Canonical phrases stay fixed
    canonical_phrases = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        canon_tokens = tokenizer(canonical_phrases)
        canon_feat = model.encode_text(canon_tokens.to(device))
        canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = canon_feat.cpu()

    # ------------------------------
    # 3) Evaluate
    # ------------------------------

    ignore_classes = ["wall", "floor", "ceiling"]  # for foreground metrics

    # ---- 3.1 Load label names & encode text ----
    with open(label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]
    print("prompt_list: ", prompt_list)

    with torch.no_grad():
        text_tokens = tokenizer(prompt_list)
        text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu()

    out_folder = os.path.join(scene_folder, save_name)
    os.makedirs(out_folder, exist_ok=True)

    seg_files = sorted(glob.glob(os.path.join(scene_folder, feat_name, "*_s.npy")))
    for seg_path in seg_files:
        # Derive the matching feature path
        base_name = os.path.basename(seg_path).replace("_s.npy", "")
        feat_path = os.path.join(scene_folder, feat_name, base_name + "_f.npy")

        if not os.path.exists(feat_path):
            print(f"Feature file not found for {seg_path}, skipping.")
            continue

        # Load segmentation and features
        seg = np.load(seg_path)  # shape (1, H, W)
        # for i, seg_i in enumerate(seg):
        #     # print("i", i, "seg_i min: ", seg_i.min(), "seg_i max: ", seg_i.max())
        #     print("i", i)
        #     unique_seg_ids = np.unique(seg_i)
        #     print("number of unique seg ids: ", len(unique_seg_ids))

        print("orginal seg shape: ", seg.shape)
        seg = seg[feature_level]  # shape (H, W)
        feat = np.load(feat_path)  # shape (N_segments, 768)
        # print("feat shape: ", feat.shape)

        # Dot product: text embeddings x segment features
        # text_embeds_np: shape (20, 768)
        # feat          : shape (N_segments, 768)
        # => logits shape (20, N_segments)
        print("seg shape: ", seg.shape)
        print("feat shape: ", feat.shape)
        # logits = np.dot(text_embeds_np, feat.T)  # shape: (20, N_segments)
        # scores = 1 / (1 + np.exp(-logits))  # shape: (20, N_segments)
        # pred_labels = np.argmax(scores, axis=0)  # shape: (N_segments,)
        # print("pred_labels shape: ", pred_labels.shape)
        feat = torch.from_numpy(feat).to(device)  # shape: (N_segments, 768)
        feat = feat.to(text_feat.dtype)  # Ensure same dtype as text_feat

        pred_labels = compute_relevancy_scores(
            feat,  # shape: (N_segments, 512)
            text_feat,
            canon_feat,
            device=device,
            use_dot_similarity=True,
        )

        H, W = seg.shape
        pred_class_map = -1 * np.ones(
            (H, W), dtype=np.int32
        )  # will store 0..19 for valid classes, -1 otherwise

        unique_seg_ids = np.unique(seg)
        for s_id in unique_seg_ids:
            if s_id < 0:
                # -1 indicates background/ignored
                continue
            if s_id >= len(pred_labels):
                # Safety check if there's any mismatch
                print(f"Warning: segment ID {s_id} out of range for features.")
                continue
            s_id = int(s_id)  # Ensure s_id is an integer
            # print(f"Processing segment ID: {s_id}")
            class_idx = pred_labels[s_id]
            pred_class_map[seg == s_id] = class_idx

        # ----------------------------------------------
        # Convert predicted class map => color image
        # ----------------------------------------------
        # pred_class_map is shape (H, W), values in [0..19] or -1
        pred_color = np.zeros((H, W, 3), dtype=np.float32)
        valid_mask = pred_class_map >= 0

        # Index into palette for valid pixels
        pred_color[valid_mask] = SCANNET_100_COLORS[pred_class_map[valid_mask]]

        # If you want the ignored pixels to appear black or something else:
        # pred_color[~valid_mask] = [0.0, 0.0, 0.0]  # black

        # ----------------------------------------------
        # Visualization with overlaid text
        # ----------------------------------------------
        plt.figure(figsize=(10, 8))
        plt.imshow(pred_color)
        plt.axis("off")

        unique_segs = np.unique(seg)
        for s_id in unique_segs:
            if s_id < 0:
                # -1 indicates ignored areas
                continue
            # Gather the mask for this segment
            mask = seg == s_id
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue

            # Centroid of the segment in (y, x)
            cy, cx = coords.mean(axis=0)
            # check if cx cy fall in the mask
            if not mask[int(cy), int(cx)]:
                # instead of mean, randomly sample a point in the mask
                cy, cx = coords[np.random.randint(0, len(coords))]

            # print(pred_labels)
            # print(s_id)
            s_id = int(s_id)
            class_idx = pred_labels[s_id]
            class_name = CLASS_LABELS_100[class_idx]
            plt.text(
                x=cx,
                y=cy,
                s=class_name,
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.1"),
            )

        out_path = os.path.join(out_folder, base_name + ".jpg")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()

        print(f"Saved zero-shot seg visualization to: {out_path}")
        # /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/0d2ee665be/dslr/segmentation_2d/DSC00119.npy


if __name__ == "__main__":
    main()
