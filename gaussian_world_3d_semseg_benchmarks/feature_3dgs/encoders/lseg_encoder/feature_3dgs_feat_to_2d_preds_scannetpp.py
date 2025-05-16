import os
import glob

import numpy as np
import torch
from pathlib import Path
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
from segmentation import make_encoder


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

# import open_clip


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


import clip


def main():
    test_feat_name = "saved_feature"
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
    # model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    # model = model.eval().to(device)
    # tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_pretrained, _ = make_encoder(
        "clip_vitl16_384",
        features=256,
        groups=1,
        expand=False,
        exportable=False,
        hooks=[5, 11, 17, 23],
        use_readout="project",
    )

    # Canonical phrases stay fixed
    canonical_phrases = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        # canon_tokens = tokenizer(canonical_phrases)
        # canon_feat = model.encode_text(canon_tokens.to(device))
        # canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        # canon_feat = canon_feat.cpu()
        canon = clip.tokenize(canonical_phrases)
        canon = (
            canon.cuda()
        )  # canon = canon.to(x.device) # TODO: need use correct device
        canon_feat = clip_pretrained.encode_text(canon)  # torch.Size([4, 512])
        canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = canon_feat.cpu()  # shape: (4, 512)
    # ------------------------------
    # 3) Evaluate
    # ------------------------------

    ignore_classes = ["wall", "floor", "ceiling"]  # for foreground metrics

    # ---- 3.1 Load label names & encode text ----
    with open(label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]

    with torch.no_grad():
        # text_tokens = tokenizer(prompt_list)
        # text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
        # text_feat /= text_feat.norm(dim=-1, keepdim=True)
        # text_feat = text_feat.cpu()
        text = clip.tokenize(label_names)
        text = text.cuda()  # text = text.to(x.device) # TODO: need use correct device
        text_feat = clip_pretrained.encode_text(text)  # torch.Size([150, 512])
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu()  # shape: (150, 512)

    print("text_feat shape: ", text_feat.shape)

    ignore_mask = np.array([name in ignore_classes for name in label_names], dtype=bool)

    num_classes = len(label_names)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for scene_name in os.listdir(
        "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannetpp/"
    )[:2]:
        for split in ["train", "test"]:
            print("Processing scene: ", scene_name, " split: ", split)
            # scene_name = '09c1414f1b'
            # split = 'train'
            scene_folder = f"/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/subset_sequences/{scene_name}/dslr"

            feature_3dgs_path = f"/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannetpp/{scene_name}"
            if not os.path.exists(feature_3dgs_path):
                print(f"feature_3dgs_path not exists: {feature_3dgs_path}")
                continue

            out_folder = os.path.join(scene_folder, "zero_shot_semseg")
            os.makedirs(out_folder, exist_ok=True)

            feat_files = sorted(
                glob.glob(
                    os.path.join(
                        feature_3dgs_path, split, "ours_7000", test_feat_name, "*.pt"
                    )
                )
            )
            # print("feat_files: ", feat_files, os.path.join(feature_3dgs_path, split, 'ours_7000', test_feat_name))

            mean_iou_list = []
            mean_class_acc_list = []
            fg_miou_list = []
            fg_macc_list = []
            global_acc_list = []

            print(
                "feat_files: ",
                len(feat_files),
                "in",
                os.path.join(feature_3dgs_path, split, "ours_7000", test_feat_name),
            )
            assert (
                len(feat_files) > 0
            ), f"No feature files found in {os.path.join(feature_3dgs_path, split, 'ours_7000', test_feat_name)}"

            for feat_path in tqdm(feat_files):
                # Derive the matching feature path
                base_name = os.path.basename(feat_path).replace("_fmap_CxHxW.pt", "")

                feat = torch.load(feat_path).to(device)  # shape (N_segments, 768)
                # print("feat shape: ", feat.shape)

                # Dot product: text embeddings x segment features
                # text_embeds_np: shape (20, 768)
                # feat          : shape (N_segments, 768)
                # => logits shape (20, N_segments)
                # print("seg shape: ", seg.shape)
                # print("feat shape: ", feat.shape)
                # logits = np.dot(text_embeds_np, feat.T)  # shape: (20, N_segments)
                # scores = 1 / (1 + np.exp(-logits))  # shape: (20, N_segments)
                # pred_labels = np.argmax(scores, axis=0)  # shape: (N_segments,)
                # print("pred_labels shape: ", pred_labels.shape)
                # feat = torch.from_numpy(feat).to(device)  # shape: (N_segments, 768)
                feat = feat.to(text_feat.dtype)  # Ensure same dtype as text_feat
                C, H, W = feat.shape

                feat_flatten = feat.permute(1, 2, 0).reshape(
                    -1, C
                )  # shape: (N_segments, 768)
                pred_labels = compute_relevancy_scores(
                    feat_flatten,  # shape: (N_segments, 512)
                    text_feat,
                    canon_feat,
                    device=device,
                    use_dot_similarity=True,
                )

                pred_labels = pred_labels.reshape(H, W)  # shape: (H, W)
                pred_class_map = pred_labels.astype(np.int64)  # shape: (H, W)
                # print("pred_class_map shape: ", pred_class_map.shape, pred_class_map.min(), pred_class_map.max())

                # saving labels
                label_save_path = feat_path.replace(
                    "_fmap_CxHxW.pt", "_pred_labels.npy"
                ).replace(test_feat_name, "pred_labels")
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                # print("save pred_labels to: ", label_save_path)
                np.save(label_save_path, pred_labels)

                # unique_seg_ids = np.unique(seg)
                # for s_id in unique_seg_ids:
                #     if s_id < 0:
                #         # -1 indicates background/ignored
                #         continue
                #     if s_id >= len(pred_labels):
                #         # Safety check if th
                #
                # ere's any mismatch
                #         print(f"Warning: segment ID {s_id} out of range for features.")
                #         continue
                #     s_id = int(s_id)  # Ensure s_id is an integer
                #     # print(f"Processing segment ID: {s_id}")
                #     class_idx = pred_labels[s_id]
                #     pred_class_map[seg == s_id] = class_idx

                # print("pred_class_map shape: ", pred_class_map.shape, pred_class_map.min(), pred_class_map.max())
                gt_2d_seg = np.load(
                    os.path.join(scene_folder, "segmentation_2d", base_name + ".npy")
                )
                # interpolate to shape (H, W)
                gt_2d_seg = gt_2d_seg.astype(np.float32)  # shape: (H, W)
                gt_2d_seg = (
                    torch.nn.functional.interpolate(
                        torch.from_numpy(gt_2d_seg).unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="nearest",
                    )
                    .squeeze()
                    .numpy()
                )
                gt_2d_seg = gt_2d_seg.astype(np.int64)  # shape: (H, W)

                for height_i in range(H):
                    for width_i in range(W):
                        gt_c = gt_2d_seg[height_i, width_i]
                        pr_c = pred_class_map[height_i, width_i]
                        if gt_c < num_classes and pr_c < num_classes:
                            if gt_c >= 0:
                                confusion_mat[gt_c, pr_c] += 1
                            else:
                                # Ignore the pixel if gt_c is -1
                                continue

    ious = []
    per_class_acc = []
    gt_class_counts = np.sum(confusion_mat, axis=1)

    for c in range(num_classes):
        tp = confusion_mat[c, c]
        fn = gt_class_counts[c] - tp
        fp = np.sum(confusion_mat[:, c]) - tp
        denom = tp + fp + fn
        iou_c = tp / denom if denom > 0 else 0.0
        ious.append(iou_c)

        acc_c = tp / gt_class_counts[c] if gt_class_counts[c] > 0 else 0.0
        per_class_acc.append(acc_c)

    valid_mask = gt_class_counts > 0
    mean_iou = np.mean(np.array(ious)[valid_mask]) if valid_mask.any() else 0.0
    mean_class_acc = (
        np.mean(np.array(per_class_acc)[valid_mask]) if valid_mask.any() else 0.0
    )

    total_correct = np.trace(confusion_mat)
    total_count = confusion_mat.sum()
    global_acc = total_correct / (total_count + 1e-12)

    # Foreground metrics (exclude ignore classes)
    fg_mask = valid_mask & (~ignore_mask)
    fg_miou = np.mean(np.array(ious)[fg_mask]) if fg_mask.any() else 0.0
    fg_macc = np.mean(np.array(per_class_acc)[fg_mask]) if fg_mask.any() else 0.0

    # ------------------------------
    # 5) Print final results
    # ------------------------------
    print("\n======== RESULTS ========")
    print("Per‑class IoU:")
    # for c, name in enumerate(label_names):
    #     print(f"  {name:24s}: {ious[c]:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Mean Class Accuracy: {mean_class_acc:.4f}")
    print(f"Foreground mIoU: {fg_miou:.4f}")
    print(f"Foreground mAcc: {fg_macc:.4f}")


if __name__ == "__main__":
    main()
