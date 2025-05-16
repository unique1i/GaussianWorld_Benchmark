import os
import glob

import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
from segmentation import make_encoder

from metadata.scannet200_constants import CLASS_LABELS_20, CLASS_LABELS_200
import clip

SCANNET_20_COLORS = np.array(
    [
        [174, 199, 232],  # wall
        [152, 223, 138],  # floor
        [31, 119, 180],  # cabinet
        [255, 187, 120],  # bed
        [188, 189, 34],  # chair
        [140, 86, 75],  # sofa
        [255, 152, 150],  # table
        [214, 39, 40],  # door
        [197, 176, 213],  # window
        [148, 103, 189],  # bookshelf
        [196, 156, 148],  # picture
        [23, 190, 207],  # counter
        [247, 182, 210],  # desk
        [219, 219, 141],  # curtain
        [255, 127, 14],  # refrigerator
        [227, 119, 194],  # shower curtain
        [158, 218, 229],  # toilet
        [44, 160, 44],  # sink
        [112, 128, 144],  # bathtub
        [82, 84, 163],  # other furniture
    ],
    dtype=np.uint8,
)  # shape: (20, 3)
# Convert to [0,1] float range for matplotlib
# SCANNET_20_COLORS = SCANNET_20_COLORS / 255.0


# dataset_root = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/'


# segment_class_names = np.loadtxt(
#     Path(dataset_root) / "metadata" / "semantic_benchmark" / "top100.txt",
#     dtype=str,
#     delimiter=".",  # dummy delimiter to replace " "
# )
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


# # Example usage
SCANNET_200_COLORS = generate_distinct_colors(200)


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


def load_scene_list(val_split_path):
    """
    Reads a .txt file listing validation scenes (one per line).
    Returns a list of scene IDs (strings).
    """
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if len(line.strip()) > 0]
    return scene_ids


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 3DGS Gradient Backprojection Benchmark"
    )
    parser.add_argument(
        "--split", type=str, default="/splits/scannet_mini_val.txt", help="Split name"
    )
    parser.add_argument("--rescale", type=int, default=0, help="rescale custom")
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/scannet_mini_val_set_suite/original_data",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--ply_root_path",
        type=str,
        default="/scannet_mini_val_set_suite/mcmc_3dgs",
        help="Path to the ply files",
    )
    parser.add_argument(
        "--results_root_dir",
        type=str,
        default="./results/scannet/",
        help="Path to the results directory",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    scene_ids = load_scene_list(args.split)
    iteration = 7000

    for scene_name in os.listdir(
        "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannet"
    ):
        for split in ["train", "test"]:
            print("Processing scene: ", scene_name, " split: ", split)
            # scene_name = '09c1414f1b'
            # split = 'train'
            # /srv/beegfs02/scratch/qimaqi_data/data/scannet_subset/subset_sequences/scene0011_00/
            scene_folder = f"/srv/beegfs02/scratch/qimaqi_data/data/scannet_subset/subset_sequences/{scene_name}/"

            feature_3dgs_path = f"/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannet/{scene_name}"
            if not os.path.exists(feature_3dgs_path):
                print(f"feature_3dgs_path not exists: {feature_3dgs_path}")
                continue

            test_feat_name = "saved_feature"
            # text_emb_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/clip_text_embeddings_100.pth"
            # text_embeds = torch.load(text_emb_path)  # shape (20, 768)
            # assert text_embeds.shape[-1] == 512
            # text_embeds_np = text_embeds.detach().cpu().numpy()  # shape (100, 768)

            # label_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt'
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
            # ------------------------------
            # 3) Evaluate
            # ------------------------------

            ignore_classes = ["wall", "floor", "ceiling"]  # for foreground metrics

            # ---- 3.1 Load label names & encode text ----
            # with open(label_path, "r") as f:
            #     label_names = [line.strip() for line in f if len(line.strip()) > 0]
            # prompt_list = ["this is a " + name for name in label_names]
            label_names_20 = CLASS_LABELS_20
            label_names_200 = CLASS_LABELS_200

            with torch.no_grad():
                # text_tokens = tokenizer(prompt_list)
                # text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
                # text_feat /= text_feat.norm(dim=-1, keepdim=True)
                # text_feat = text_feat.cpu()
                text = clip.tokenize(label_names_20)
                text = (
                    text.cuda()
                )  # text = text.to(x.device) # TODO: need use correct device
                text_feat = clip_pretrained.encode_text(text)  # torch.Size([150, 512])
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat.cpu()  # shape: (150, 512)
            text_feat_20 = text_feat

            with torch.no_grad():
                # text_tokens = tokenizer(prompt_list)
                # text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
                # text_feat /= text_feat.norm(dim=-1, keepdim=True)
                # text_feat = text_feat.cpu()
                text = clip.tokenize(label_names_200)
                text = (
                    text.cuda()
                )  # text = text.to(x.device) # TODO: need use correct device
                text_feat = clip_pretrained.encode_text(text)  # torch.Size([150, 512])
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat.cpu()  # shape: (150, 512)
            text_feat_200 = text_feat

            print("text_feat shape: ", text_feat.shape)

            ignore_mask_20 = np.array(
                [name in ignore_classes for name in label_names_20], dtype=bool
            )
            ignore_mask_200 = np.array(
                [name in ignore_classes for name in label_names_200], dtype=bool
            )

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
                pred_labels_20 = compute_relevancy_scores(
                    feat_flatten,  # shape: (N_segments, 512)
                    text_feat_20,  # shape: (20, 512)
                    text_feat_20,
                    device=device,
                    use_dot_similarity=True,
                )

                pred_labels_200 = compute_relevancy_scores(
                    feat_flatten,  # shape: (N_segments, 512)
                    text_feat_200,  # shape: (20, 512)
                    text_feat_200,
                    device=device,
                    use_dot_similarity=True,
                )

                print("pred_labels_20 shape: ", pred_labels_20.shape)
                print("pred_labels_200 shape: ", pred_labels_200.shape)

                pred_labels_20 = pred_labels_20.reshape(H, W)  # shape: (H, W)
                pred_labels_200 = pred_labels_200.reshape(H, W)  # shape: (H, W)

                pred_labels_20 = pred_labels_20.astype(np.int64)  # shape: (H, W)
                pred_labels_200 = pred_labels_200.astype(np.int64)  # shape: (H, W)

                print(
                    "pred_labels_20 shape: ",
                    pred_labels_20.shape,
                    pred_labels_20.min(),
                    pred_labels_20.max(),
                )
                print(
                    "pred_labels_200 shape: ",
                    pred_labels_200.shape,
                    pred_labels_200.min(),
                    pred_labels_200.max(),
                )

                # saving labels
                label_save_path = feat_path.replace(
                    "_fmap_CxHxW.pt", "_pred_labels.npy"
                ).replace(test_feat_name, "pred_labels_20")
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                print("save pred_labels to: ", label_save_path)
                np.save(label_save_path, pred_labels_20)

                label_save_path = feat_path.replace(
                    "_fmap_CxHxW.pt", "_pred_labels.npy"
                ).replace(test_feat_name, "pred_labels_200")
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                print("save pred_labels to: ", label_save_path)
                np.save(label_save_path, pred_labels_200)

            # raise NotImplementedError("pred_labels_200 not implemented")

            #     # print("pred_class_map shape: ", pred_class_map.shape, pred_class_map.min(), pred_class_map.max())
            #     gt_2d_seg = np.load(os.path.join(scene_folder, 'segmentation_2d', base_name + ".npy"))
            #     # interpolate to shape (H, W)
            #     gt_2d_seg = gt_2d_seg.astype(np.float32)  # shape: (H, W)
            #     gt_2d_seg = torch.nn.functional.interpolate(
            #         torch.from_numpy(gt_2d_seg).unsqueeze(0).unsqueeze(0),
            #         size=(H, W),
            #         mode='nearest'
            #     ).squeeze().numpy()
            #     gt_2d_seg = gt_2d_seg.astype(np.int64)  # shape: (H, W)

            #     print("gt_2d_seg shape: ", gt_2d_seg.shape, gt_2d_seg.min(), gt_2d_seg.max())

            #     # calculate accuracy and mIoU

            #     # ------------------------------
            #     # 4) Compute metrics
            #         # ------------------------------
            #     num_classes = len(label_names)
            #     confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

            #     # ---- 3.2.e Accumulate confusion ----
            #     # for gt_c, pr_c in zip(gt_val, pred_label):
            #     #     if gt_c < num_classes and pr_c < num_classes:  # guard against idx mismatch
            #     #         confusion_mat[gt_c, pr_c] += 1

            #     for height_i in range(H):
            #         for width_i in range(W):
            #             gt_c = gt_2d_seg[height_i, width_i]
            #             pr_c = pred_class_map[height_i, width_i]
            #             if gt_c < num_classes and pr_c < num_classes:
            #                 if gt_c >= 0:
            #                     confusion_mat[gt_c, pr_c] += 1
            #                 else:
            #                     # Ignore the pixel if gt_c is -1
            #                     continue

            #     ious = []
            #     per_class_acc = []
            #     gt_class_counts = np.sum(confusion_mat, axis=1)

            #     for c in range(num_classes):
            #         tp = confusion_mat[c, c]
            #         fn = gt_class_counts[c] - tp
            #         fp = np.sum(confusion_mat[:, c]) - tp
            #         denom = tp + fp + fn
            #         iou_c = tp / denom if denom > 0 else 0.0
            #         ious.append(iou_c)

            #         acc_c = tp / gt_class_counts[c] if gt_class_counts[c] > 0 else 0.0
            #         per_class_acc.append(acc_c)

            #     valid_mask = gt_class_counts > 0
            #     mean_iou = np.mean(np.array(ious)[valid_mask]) if valid_mask.any() else 0.0
            #     mean_class_acc = np.mean(np.array(per_class_acc)[valid_mask]) if valid_mask.any() else 0.0

            #     total_correct = np.trace(confusion_mat)
            #     total_count = confusion_mat.sum()
            #     global_acc = total_correct / (total_count + 1e-12)

            #     # Foreground metrics (exclude ignore classes)
            #     fg_mask = valid_mask & (~ignore_mask)
            #     fg_miou = np.mean(np.array(ious)[fg_mask]) if fg_mask.any() else 0.0
            #     fg_macc = np.mean(np.array(per_class_acc)[fg_mask]) if fg_mask.any() else 0.0

            #     # ------------------------------
            #     # 5) Print final results
            #     # ------------------------------
            #     print("\n======== RESULTS ========")
            #     print("Per‑class IoU:")
            #     # for c, name in enumerate(label_names):
            #     #     print(f"  {name:24s}: {ious[c]:.4f}")
            #     print(f"Mean IoU: {mean_iou:.4f}")
            #     print(f"Global Accuracy: {global_acc:.4f}")
            #     print(f"Mean Class Accuracy: {mean_class_acc:.4f}")
            #     print(f"Foreground mIoU: {fg_miou:.4f}")
            #     print(f"Foreground mAcc: {fg_macc:.4f}")

            #     mean_iou_list.append(mean_iou)
            #     mean_class_acc_list.append(mean_class_acc)
            #     fg_miou_list.append(fg_miou)
            #     fg_macc_list.append(fg_macc)
            #     global_acc_list.append(global_acc)

            # # summary
            # print("\n======== SUMMARY ========")
            # print(f"Mean IoU: {np.mean(mean_iou_list):.4f}")
            # print(f"Mean Class Accuracy: {np.mean(mean_class_acc_list):.4f}")
            # print(f"Foreground mIoU: {np.mean(fg_miou_list):.4f}")
            # print(f"Foreground mAcc: {np.mean(fg_macc_list):.4f}")
            # print(f"Global Accuracy: {np.mean(global_acc_list):.4f}")

        # # ----------------------------------------------
        # # Convert predicted class map => color image
        # # ----------------------------------------------
        # # pred_class_map is shape (H, W), values in [0..19] or -1
        # pred_color = np.zeros((H, W, 3), dtype=np.float32)
        # valid_mask = (pred_class_map >= 0)

        # # Index into palette for valid pixels
        # pred_color[valid_mask] = SCANNET_100_COLORS[pred_class_map[valid_mask]]

        # # If you want the ignored pixels to appear black or something else:
        # # pred_color[~valid_mask] = [0.0, 0.0, 0.0]  # black

        # # ----------------------------------------------
        # # Visualization with overlaid text
        # # ----------------------------------------------
        # plt.figure(figsize=(10, 8))
        # plt.imshow(pred_color)
        # plt.axis("off")

        # unique_segs = np.unique(seg)
        # for s_id in unique_segs:
        #     if s_id < 0:
        #         # -1 indicates ignored areas
        #         continue
        #     # Gather the mask for this segment
        #     mask = (seg == s_id)
        #     coords = np.argwhere(mask)
        #     if coords.size == 0:
        #         continue

        #     # Centroid of the segment in (y, x)
        #     cy, cx = coords.mean(axis=0)

        #     # print(pred_labels)
        #     # print(s_id)
        #     s_id = int(s_id)
        #     class_idx = pred_labels[s_id]
        #     class_name = CLASS_LABELS_100[class_idx]
        #     plt.text(
        #         x=cx, y=cy,
        #         s=class_name,
        #         color="white",
        #         fontsize=8,
        #         ha="center",
        #         va="center",
        #         bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.1")
        #     )

        # out_path = os.path.join(out_folder, base_name + ".jpg")
        # plt.savefig(out_path, bbox_inches="tight", dpi=200)
        # plt.close()

        # print(f"Saved zero-shot seg visualization to: {out_path}")


if __name__ == "__main__":
    main()
