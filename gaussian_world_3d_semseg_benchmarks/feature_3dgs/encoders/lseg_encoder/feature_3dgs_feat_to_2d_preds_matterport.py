import os
import glob

import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
from segmentation import make_encoder

from metadata.matterport3d import MATTERPORT_LABELS_21
import clip


def load_scene_list(val_split_path):
    """
    Reads a .txt file listing validation scenes (one per line).
    Returns a list of scene IDs (strings).
    """
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if len(line.strip()) > 0]
    return scene_ids


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


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 3DGS Gradient Backprojection Benchmark"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="/splits/matterport3d_mini_test.txt",
        help="Split name",
    )
    parser.add_argument("--rescale", type=int, default=0, help="rescale custom")
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/matterport3d_region_mini_test_set_suite/original_data",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--ply_root_path",
        type=str,
        default="/matterport3d_region_mini_test_set_suite/mcmc_3dgs",
        help="Path to the ply files",
    )
    parser.add_argument(
        "--results_root_dir",
        type=str,
        default="./results/matterport/",
        help="Path to the results directory",
    )

    return parser.parse_args()


def main():
    args = get_arguments()
    scene_ids = load_scene_list(args.split)
    iteration = 7000

    for scene_name in scene_ids:
        for split in ["train", "test"]:
            print("Processing scene: ", scene_name, " split: ", split)

            scene_folder = os.path.join(args.data_root_path, scene_name)

            feature_3dgs_path = f"feature-3dgs_Qi/output/matterport/{scene_name}"
            # f'/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/matterport/{scene_name}'
            if not os.path.exists(feature_3dgs_path):
                print(f"feature_3dgs_path not exists: {feature_3dgs_path}")
                continue

            test_feat_name = "saved_feature"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            clip_pretrained, _ = make_encoder(
                "clip_vitl16_384",
                features=256,
                groups=1,
                expand=False,
                exportable=False,
                hooks=[5, 11, 17, 23],
                use_readout="project",
            )

            label_names_21 = MATTERPORT_LABELS_21
            text_prompts_21 = ["this is a " + label for label in MATTERPORT_LABELS_21]

            with torch.no_grad():
                text = clip.tokenize(text_prompts_21)
                text = (
                    text.cuda()
                )  # text = text.to(x.device) # TODO: need use correct device
                text_feat = clip_pretrained.encode_text(text)  # torch.Size([150, 512])
                text_feat /= text_feat.norm(dim=-1, keepdim=True)
                text_feat = text_feat.cpu()  # shape: (150, 512)
            text_feat_21 = text_feat

            ignore_classes = ["other furniture", "wall", "floor", "ceiling"]

            ignore_mask_21 = np.array(
                [name in ignore_classes for name in label_names_21], dtype=bool
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

                feat = feat.to(text_feat.dtype)  # Ensure same dtype as text_feat
                C, H, W = feat.shape

                feat_flatten = feat.permute(1, 2, 0).reshape(
                    -1, C
                )  # shape: (N_segments, 768)
                pred_labels_21 = compute_relevancy_scores(
                    feat_flatten,  # shape: (N_segments, 512)
                    text_feat_21,  # shape: (20, 512)
                    text_feat_21,
                    device=device,
                    use_dot_similarity=True,
                )

                pred_labels_21 = pred_labels_21.reshape(H, W)  # shape: (H, W)

                pred_labels_21 = pred_labels_21.astype(np.int64)  # shape: (H, W)

                print(
                    "pred_labels_21 shape: ",
                    pred_labels_21.shape,
                    pred_labels_21.min(),
                    pred_labels_21.max(),
                )

                # saving labels
                label_save_path = feat_path.replace(
                    "saved_feature", "pred_labels"
                ).replace("_fmap_CxHxW.pt", "_pred_labels.npy")
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                print("save pred_labels to: ", label_save_path)
                np.save(label_save_path, pred_labels_21)


if __name__ == "__main__":
    main()
