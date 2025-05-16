import os
import argparse
import numpy as np
import torch
import open_clip
import sys

from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

###################################
# 1. Basic I/O Utilities
###################################


def save_results_to_file(log_path, results_str, args):
    """Save the results to a text file with relevant parameters in the filename."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as f:
        f.write("Command line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        f.write(results_str)

    print(f"\nResults saved to: {log_path}")


def load_scene_list(val_split_path):
    """
    Reads a .txt file listing validation scenes (one per line).
    Returns a list of scene IDs (strings).
    """
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if len(line.strip()) > 0]
    return scene_ids


def read_ply_file_3dgs(file_path):
    """
    Reads the 3D Gaussian ply (e.g. point_cloud_30000.ply).
    Returns xyz and opacity.
    """
    ply_data = PlyData.read(file_path)
    vertex = ply_data["vertex"]
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    opacity = vertex["opacity"]
    xyz = np.stack([x, y, z], axis=-1)
    return xyz, opacity


###################################
# 2. CLIP Relevancy Scoring
###################################


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

    # Fast path: plain dot‑product similarity
    if use_dot_similarity:
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        pred_label = torch.argmax(dot_lang_text, dim=1)
        return pred_label.cpu().numpy()

    # Original ratio‑based relevancy score
    canon_feat = canon_feat.to(device, non_blocking=True)
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


###################################
# 3. Main Evaluation Script
###################################


def parse_args():
    parser = argparse.ArgumentParser(description="Open‑Vocal 3DGS semantic evaluation")
    parser.add_argument(
        "--val_split_path",
        type=str,
        default="/home/yli7/projects/yue/language_feat_exps/splits/scannetpp_mini_val.txt",
    )
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        default="/home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v2_preprocessed",
    )
    parser.add_argument(
        "--gs_root",
        type=str,
        default="/home/yli7/scratch/datasets/gaussian_world/outputs/ludvig/scannetpp",
    )
    parser.add_argument(
        "--langfeat_root",
        type=str,
        default="/home/yli7/scratch/datasets/gaussian_world/outputs/ludvig/scannetpp",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="/home/yli7/scratch2/datasets/scannetpp_v2/metadata/semantic_benchmark/top100.txt",
    )
    parser.add_argument(
        "--use_dot_similarity",
        action="store_true",
        help="If set, use plain CLIP dot similarity instead of ratio scoring.",
    )
    parser.add_argument(
        "--save_pred",
        action="store_true",
        help="Save predictions for all coord as <scene_id>_<benchmark>_semseg_pred.npy",
    )
    parser.add_argument(
        "--model_name", type=str, default="clip", choices=["clip", "siglip2"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_dot_similarity = True
    model_name = args.model_name
    split_name = args.val_split_path.split("/")[-1].split(".")[0]

    print(f"Using {model_name.upper()} language features from {args.langfeat_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Capture all printed output
    stdout_original = sys.stdout
    results_capture = []

    class CaptureOutput:
        def write(self, text):
            results_capture.append(text)
            stdout_original.write(text)

        def flush(self):
            stdout_original.flush()

    sys.stdout = CaptureOutput()

    # ------------------------------
    # 1) Load validation scenes
    # ------------------------------
    scene_ids = load_scene_list(args.val_split_path)
    # scene_ids = [scene_ids[0]]
    print(f"Found {len(scene_ids)} validation scenes.")

    # ------------------------------
    # 2) Load CLIP model
    # ------------------------------
    if model_name == "clip":
        CLIP_MODEL = "ViT-B-16"
        CLIP_PRETRAIN = "laion2b_s34b_b88k"
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAIN
        )
        model = model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    elif model_name == "siglip2":
        siglip_spec = "siglip2-base-patch16-512"
        model = AutoModel.from_pretrained(f"google/{siglip_spec}").eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"google/{siglip_spec}")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # # Canonical phrases stay fixed
    # canonical_phrases = ["object", "things", "stuff", "texture"]
    # with torch.no_grad():
    #     canon_tokens = tokenizer(canonical_phrases)
    #     canon_feat = model.encode_text(canon_tokens.to(device))
    #     canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
    #     canon_feat = canon_feat.cpu()

    # ------------------------------
    # 3) Evaluate
    # ------------------------------

    ignore_classes = ["wall", "floor", "ceiling"]  # for foreground metrics

    # ---- 3.1 Load label names & encode text ----
    with open(args.label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]

    def prepare_text_features(text_prompts):
        if model_name == "clip":
            text_tokens = tokenizer(text_prompts)
            text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu()
        elif model_name == "siglip2":
            inputs = tokenizer(
                text_prompts, padding="max_length", max_length=64, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_feat = model.get_text_features(**inputs)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat.cpu()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        return text_feat

    text_feat = prepare_text_features(prompt_list)

    num_classes = len(label_names)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Build ignore mask once
    ignore_mask = np.array([name in ignore_classes for name in label_names], dtype=bool)

    # ---- 3.2 Loop over scenes ----
    for scene_id in tqdm(scene_ids, desc="Scene", dynamic_ncols=True):
        # Paths to data
        scene_preproc_folder = os.path.join(args.preprocessed_root, "val", scene_id)
        if not os.path.isdir(scene_preproc_folder):
            print(f"[Warning] Preprocessed folder not found: {scene_preproc_folder}")
            continue

        coord_path = os.path.join(scene_preproc_folder, "coord.npy")
        segment_path = os.path.join(scene_preproc_folder, "segment.npy")
        if not (os.path.isfile(coord_path) and os.path.isfile(segment_path)):
            print(
                f"[Warning] Missing coord.npy or segment.npy at {scene_preproc_folder}"
            )
            continue

        coord = np.load(coord_path)  # (N, 3)
        segment = np.load(segment_path)  # (N, 3) first col => label index
        labeled_gt = segment[
            :, 0
        ]  # scannetpp save top-3 lables as some points have multiple labels
        valid_mask = (
            labeled_gt >= 0
        )  # in the preprocessing, -1 is used for ignore labels
        if valid_mask.sum() == 0:
            continue

        xyz_val = coord[valid_mask]
        gt_val = labeled_gt[valid_mask].astype(np.int64)

        # ---- 3.2.b Load 3DGS & CLIP feats ----
        scene_3dgs_folder = os.path.join(args.gs_root, scene_id)
        ply_path = os.path.join(scene_3dgs_folder, model_name, "gaussians.ply")
        if not os.path.isfile(ply_path):
            print(f"[Warning] 3DGS .ply not found for scene {scene_id} at {ply_path}")
            continue
        gauss_xyz, _ = read_ply_file_3dgs(ply_path)

        langfeat_path = os.path.join(
            args.langfeat_root, scene_id, model_name, "features.npy"
        )
        if not os.path.isfile(langfeat_path):
            print(f"[Warning] Language feature not found at {langfeat_path}")
            continue
        gauss_lang_feat = torch.from_numpy(np.load(langfeat_path)).float()  # (G, 512)

        norms = gauss_lang_feat.norm(dim=1)
        keep_mask_gs = norms > 0
        gauss_xyz = gauss_xyz[keep_mask_gs.numpy()]
        gauss_lang_feat = gauss_lang_feat[keep_mask_gs]
        if gauss_xyz.shape[0] == 0:
            print(f"[Warning] All 3DGS zero feats in {scene_id}")
            continue

        # ---- 3.2.c KD‑Tree NN search ----
        kd_tree = KDTree(gauss_xyz)
        _, nn_idx = kd_tree.query(xyz_val)
        nn_lang_feat = gauss_lang_feat[nn_idx]

        # ---- 3.2.d Predict labels ----
        pred_label = compute_relevancy_scores(
            nn_lang_feat,
            text_feat,
            None,
            device=device,
            use_dot_similarity=args.use_dot_similarity,
        )

        if args.save_pred:
            _, nn_idx_all = kd_tree.query(coord)  # (N,)
            nn_lang_feat_all = gauss_lang_feat[nn_idx_all]
            pred_label_all = compute_relevancy_scores(
                nn_lang_feat_all,
                text_feat,
                None,
                device=device,
                use_dot_similarity=args.use_dot_similarity,
            )
            save_fname = f"ludvig_{model_name}_{scene_id}_semseg_pred.npy"
            save_path = os.path.join(f"output/ludvig/{split_name}", save_fname)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, pred_label_all.astype(np.int32))
            # print(f"[Info] Saved predictions for scene {scene_id} ({bench_name}) to {save_path}")

        # ---- 3.2.e Accumulate confusion ----
        for gt_c, pr_c in zip(gt_val, pred_label):
            if gt_c < num_classes and pr_c < num_classes:  # guard against idx mismatch
                confusion_mat[gt_c, pr_c] += 1

    # ------------------------------
    # 4) Compute metrics
    # ------------------------------
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
    for c, name in enumerate(label_names):
        print(f"  {name:24s}: {ious[c]:.4f}")
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Class Accuracy: {mean_class_acc:.4f}")
    print(f"Foreground mIoU: {fg_miou:.4f}")
    print(f"Foreground mAcc: {fg_macc:.4f}")

    sys.stdout = stdout_original
    results_str = "".join(results_capture)
    log_path = f"logs/ludvig_3d_semseg_eval_{split_name}_{model_name}.txt"
    save_results_to_file(log_path, results_str, args)


if __name__ == "__main__":
    main()
