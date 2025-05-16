import os
import numpy as np
import torch
import argparse
import gc
import open_clip
import sys

from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
from transformers import AutoModel, AutoTokenizer

"""
Scanet++ Zero-shot Semantic Segmentation Evaluation Script

Official Criteria:
"Top-1 evaluation considers the top prediction for each vertex, and considers a prediction correct if it matches any ground truth label for that vertex;
Top-3 evaluation considers the top 3 predictions for each vertex, and considers a prediction correct if any of the top 3 predictions matches the ground truth. 
For multilabeled vertices, all labels in the ground truth must be present in the top 3 predictions for the prediction to be considered correct."
"""

###################################
# 1. Basic I/O Utilities
###################################


def load_scene_list(val_split_path):
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if line.strip()]
    return scene_ids


def read_ply_file_3dgs(file_path):
    ply_data = PlyData.read(file_path)
    vertex = ply_data["vertex"]
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    opacity = vertex["opacity"]
    xyz = np.stack([x, y, z], axis=-1)
    return xyz, opacity


def save_results_to_file(log_path, results_str, args):
    """Save the results to a text file with relevant parameters in the filename."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Write results to the file
    with open(log_path, "w") as f:
        # Write command line arguments first
        f.write("Command line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")

        # Write experiment results
        f.write(results_str)

    print(f"\nResults saved to: {log_path}")


###################################
# 2. SigLIP Relevancy Scoring
###################################


@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,
    text_feat: torch.Tensor,
    canon_feat: torch.Tensor,
    device: torch.device,
    use_siglip_probabilities: bool = True,
):
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    # print("lang_feat.norm:", lang_feat.norm(dim=-1, keepdim=True).mean())
    # print("text_feat.norm:", text_feat.norm(dim=-1, keepdim=True).mean())
    if use_siglip_probabilities:
        logits = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        if model_name == "siglip2":
            probs = torch.sigmoid(logits)  # (N, C)
            top1_probs, top1_indices = torch.topk(probs, k=1, dim=1)  # (N, 1)
            mask = top1_probs[:, 0] >= 0.1
            top1_indices[~mask] = IGNORE_INDEX
        else:
            probs = torch.softmax(logits, dim=1)  # (N, C) for other models
            top1_indices = torch.argmax(probs, dim=1, keepdim=True)
        return top1_indices.cpu().numpy()  # shape (N, 1)
    else:
        canon_feat = canon_feat.to(device, non_blocking=True)
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())
        dot_lang_canon = torch.matmul(lang_feat, canon_feat.t())
        N, C = dot_lang_text.shape
        relevancy_scores = []
        for c_idx in range(C):
            text_c_exp = dot_lang_text[:, c_idx].exp().unsqueeze(-1)
            ratio_c = text_c_exp / (dot_lang_canon.exp() + text_c_exp)
            score_c = torch.min(ratio_c, dim=1).values
            relevancy_scores.append(score_c)
        relevancy_matrix = torch.stack(relevancy_scores, dim=0).t()
        _, top1_indices = torch.topk(relevancy_matrix, k=1, dim=1)
        return top1_indices.cpu().numpy()


###################################
# 3. Main Evaluation Script
###################################

model_name = "siglip2"  # override with the model name


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--val_split_path", type=str)
    argparser.add_argument("--scannetpp_preprocessed_root", type=str)
    argparser.add_argument("--scannetpp_3dgs_root", type=str)
    argparser.add_argument(
        "--nn_num",
        type=int,
        help="Number of nearest neighbors to consider for semseg voting.",
    )
    argparser.add_argument(
        "--print_class_iou",
        action="store_true",
        help="If true, print Top-1 IoU for each class.",
    )
    argparser.add_argument(
        "--ignore_index", type=int, default=-1, help="Index to ignore in the confusion"
    )
    argparser.add_argument(
        "--ignore_classes",
        nargs="+",
        default=["wall", "floor", "ceiling"],
        help="Comma-separated list of classes to ignore in the evaluation.",
    )
    argparser.add_argument(
        "--model_spec",
        type=str,
        default="siglip2-base-patch16-512",
        help="SigLIP model specification.",
    )
    argparser.add_argument(
        "--eval_top3",
        action="store_true",
        help="Enable Top-3 evaluation. If not set, only Top-1 evaluation is performed.",
    )
    args = argparser.parse_args()
    eval_top3 = args.eval_top3
    args.print_class_iou = True

    global IGNORE_INDEX
    IGNORE_INDEX = args.ignore_index

    # Paths & Setup
    global model_name
    version = "scannetpp_v2"
    val_split_path = (
        args.val_split_path
        or "/home/yli7/projects/yue/language_feat_exps/splits/scannetpp_val.txt"
    )
    scannetpp_preprocessed_root = (
        args.scannetpp_preprocessed_root
        or f"/home/yli7/scratch2/datasets/ptv3_preprocessed/{version}_preprocessed"
    )
    scannetpp_3dgs_root = (
        args.scannetpp_3dgs_root
        or "/home/yli7/scratch2/datasets/gaussian_world/scannetpp_v2_mcmc_3dgs"
    )  # scannetpp_v1_default_fix_xyz_gs # scannetpp_v1_mcmc_3dgs
    text_path = (
        f"/home/yli7/scratch2/datasets/{version}/metadata/semantic_benchmark/top100.txt"
    )
    scannetpp_langfeat_root = os.path.join(
        scannetpp_3dgs_root, f"language_features_{model_name}"
    )
    print(
        f"Using {model_name.upper()} language features from {scannetpp_langfeat_root}"
    )

    EXCLUDED_CLASS_NAMES = args.ignore_classes
    NN_NUM = args.nn_num if args.nn_num else 25
    print(f"Using {NN_NUM} nearest neighbors for voting.")
    device = torch.device("cuda")
    use_siglip_probabilities = True if not model_name == "clip" else False

    # used for logging
    split_name = val_split_path.split("/")[-1].split(".")[0]
    method_name = "oracle"
    gs_folder_name = os.path.basename(os.path.normpath(scannetpp_3dgs_root))
    log_path = f"logs/{method_name}_{gs_folder_name}_nn_num_{NN_NUM}_{split_name}_{model_name}.txt"

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

    # Load validation scenes
    scene_ids = load_scene_list(val_split_path)
    # scene_ids = [scene_ids[-1]] # for debugging
    print("Evaluating: ", scene_ids)
    print(f"Found {len(scene_ids)} validation scenes.")

    # Load SigLIP model & text embeddings
    # siglip_spec = args.model_spec
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

    with open(text_path, "r") as f:
        lines = f.readlines()
    top100_label = [line.strip() for line in lines]

    def prepare_text_features(text_prompts):
        text_prompts = ["this is a " + name for name in text_prompts]
        if model_name == "clip":
            text_tokens = tokenizer(text_prompts).to(device)
            with torch.no_grad():
                text_feat = model.encode_text(text_tokens)  # (C, 512)
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

    text_feat = prepare_text_features(top100_label)  # (C, 512)

    canon_feat = None
    canonical_phrases = ["object", "things", "stuff", "texture"]
    if model_name == "clip":
        canon_tokens = tokenizer(canonical_phrases)
        canon_feat = model.encode_text(canon_tokens.to(device))
        canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = canon_feat.cpu()
    else:
        with torch.no_grad():
            canon_inputs = tokenizer(
                canonical_phrases,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            canon_inputs = {k: v.to(device) for k, v in canon_inputs.items()}
            canon_feat = model.get_text_features(**canon_inputs)
            canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
            canon_feat = canon_feat.cpu()
    num_classes = len(top100_label)
    print(f"num classes: {num_classes}")

    # Prepare Metrics
    total_points = 0
    top1_correct = 0
    # Only allocate top-3 metrics if eval_top3 is enabled.
    if eval_top3:
        top3_correct = 0
        TP3 = np.zeros(num_classes, dtype=np.int64)
        FP3 = np.zeros(num_classes, dtype=np.int64)
        FN3 = np.zeros(num_classes, dtype=np.int64)
        confusion_mat_top3 = np.zeros((num_classes, num_classes), dtype=np.int64)
        fn_ignore_top3 = np.zeros(num_classes, dtype=np.int64)
    TP1 = np.zeros(num_classes, dtype=np.int64)
    FP1 = np.zeros(num_classes, dtype=np.int64)
    FN1 = np.zeros(num_classes, dtype=np.int64)
    confusion_mat_top1 = np.zeros((num_classes, num_classes), dtype=np.int64)
    fn_ignore_top1 = np.zeros(num_classes, dtype=np.int64)

    # Loop over scenes
    for scene_id in tqdm(scene_ids, desc="Scene", dynamic_ncols=True):
        scene_preproc_folder = os.path.join(
            scannetpp_preprocessed_root, "val", scene_id
        )
        if not os.path.isdir(scene_preproc_folder):
            scene_preproc_folder = os.path.join(
                scannetpp_preprocessed_root, "train", scene_id
            )
        if not os.path.isdir(scene_preproc_folder):
            print(
                f"Skipping {scene_id}: preprocessed data not found at {scene_preproc_folder}."
            )
            continue

        # Load scene data
        coord = np.load(os.path.join(scene_preproc_folder, "coord.npy"))
        segment = np.load(os.path.join(scene_preproc_folder, "segment.npy"))
        gt_labels = [row[row >= 0] for row in segment]
        valid_mask = np.array([len(labels) > 0 for labels in gt_labels], dtype=bool)
        if np.sum(valid_mask) == 0:
            continue
        xyz_val = coord[valid_mask]
        gt_val = [gt_labels[i] for i in np.where(valid_mask)[0]]
        gt_first = [g[0] for g in gt_val]

        # Load 3DGS and language features
        three_dgs_ckpt = os.path.join(
            scannetpp_3dgs_root, scene_id, "ckpts", "point_cloud_30000.ply"
        )
        if not os.path.isfile(three_dgs_ckpt):
            continue
        gauss_xyz, _ = read_ply_file_3dgs(three_dgs_ckpt)
        langfeat_path = os.path.join(scannetpp_langfeat_root, scene_id, "langfeat.pth")
        if not os.path.isfile(langfeat_path):
            print(
                f"Skipping {scene_id}: language features not found at {langfeat_path}\n"
            )
            continue
        gauss_lang_feat = torch.load(langfeat_path, map_location="cpu")[0]
        # debug
        # print(f"text_feat.shape:", text_feat.shape)
        # print(f"gauss_lang_feat.shape:", gauss_lang_feat.shape)
        norms = gauss_lang_feat.norm(dim=1)
        keep_mask = norms > 0
        gauss_xyz = gauss_xyz[keep_mask.numpy()]
        gauss_lang_feat = gauss_lang_feat[keep_mask]
        # print(f"gauss_lang_feat.shape after filtering:", gauss_lang_feat.shape)

        # Precompute initial labels for all Gaussians in the scene
        max_batch_size = 192000
        num_gauss = gauss_lang_feat.shape[0]
        all_gauss_labels = []
        for start in range(0, num_gauss, max_batch_size):
            end = min(start + max_batch_size, num_gauss)
            batch_features = gauss_lang_feat[start:end]
            batch_labels = compute_relevancy_scores(
                batch_features, text_feat, canon_feat, device, use_siglip_probabilities
            )
            all_gauss_labels.append(batch_labels)
        gauss_labels = np.concatenate(all_gauss_labels, axis=0)  # (M, 1)

        # KDTree search for NN_NUM nearest neighbors
        kd_tree = KDTree(gauss_xyz)
        _, nn_idx = kd_tree.query(xyz_val, k=NN_NUM)  # (N_val, NN_NUM)

        # Retrieve precomputed labels for neighbors
        pred_labels_per_neighbor = gauss_labels[nn_idx].squeeze(-1)  # (N_val, NN_NUM)

        # Aggregate predictions via voting (remainder of the code remains unchanged)
        topk_labels = []
        for row in pred_labels_per_neighbor:
            valid_preds = row[row != -1]
            if len(valid_preds) == 0:
                topk = [-1, -1, -1]
            else:
                unique, counts = np.unique(valid_preds, return_counts=True)
                if len(unique) == 1:
                    topk = [unique[0]] * 3
                elif len(unique) == 2:
                    idx_sorted = np.argsort(-counts)
                    sorted_labels = unique[idx_sorted]
                    topk = [sorted_labels[0], sorted_labels[1], sorted_labels[0]]
                else:
                    idx_sorted = np.argsort(-counts)
                    sorted_labels = unique[idx_sorted]
                    topk = sorted_labels[:3].tolist()
            topk_labels.append(topk)
        topk_labels = np.array(topk_labels)

        # Global vertex-level accuracy and per-class updates (Top-1 always, Top-3 only if enabled)
        top1_pred = topk_labels[:, 0]
        for i in range(len(gt_val)):
            total_points += 1
            g = gt_val[i]
            if top1_pred[i] in g:
                top1_correct += 1
            # Update Top-3 if evaluation is enabled.
            if eval_top3 and all(lbl in topk_labels[i] for lbl in g):
                top3_correct += 1

            G = set(g.tolist())
            P1 = {top1_pred[i]} if top1_pred[i] != -1 else set()
            # Update Top-1 per-class metrics.
            for pred in P1:
                if pred in G:
                    TP1[pred] += 1
                else:
                    FP1[pred] += 1
            for label in G:
                if label not in P1:
                    FN1[label] += 1

            # Update Top-3 per-class metrics if enabled.
            if eval_top3:
                P3 = set(topk_labels[i])
                P3.discard(-1)
                for pred in P3:
                    if pred in G:
                        TP3[pred] += 1
                    else:
                        FP3[pred] += 1
                for label in G:
                    if label not in P3:
                        FN3[label] += 1

            gt_c = gt_first[i]
            pr1 = top1_pred[i]
            if pr1 == -1:
                fn_ignore_top1[gt_c] += 1
            else:
                confusion_mat_top1[gt_c, pr1] += 1

            if eval_top3:
                if all(lbl in topk_labels[i] for lbl in g):
                    pr3 = gt_c
                else:
                    pr3 = topk_labels[i, 0]
                if pr3 == -1:
                    fn_ignore_top3[gt_c] += 1
                else:
                    confusion_mat_top3[gt_c, pr3] += 1

        # Clean up large temporary variables for this scene
        del coord, segment, gt_labels, xyz_val, gt_val, gt_first
        del gauss_xyz, gauss_lang_feat, nn_idx
        del pred_labels_per_neighbor, topk_labels
        gc.collect()

    # Compute Metrics from Confusion Matrices
    def compute_confmat_metrics(confmat, fn_ignore):
        per_class_iou = []
        per_class_acc = []
        total_points_cm = np.sum(confmat) + np.sum(fn_ignore)
        for c in range(num_classes):
            tp = confmat[c, c]
            fn = np.sum(confmat[c, :]) - tp + fn_ignore[c]
            fp = np.sum(confmat[:, c]) - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            per_class_iou.append(iou)
            gt_count = tp + fn
            per_class_acc.append(tp / gt_count if gt_count > 0 else 0.0)
        global_acc = (
            np.sum(np.diag(confmat)) / total_points_cm if total_points_cm > 0 else 0.0
        )
        present_classes = [
            c for c in range(num_classes) if (np.sum(confmat[c, :]) + fn_ignore[c]) > 0
        ]
        mean_class_acc = (
            np.mean([per_class_acc[c] for c in present_classes])
            if present_classes
            else 0.0
        )
        mIoU = (
            np.mean([per_class_iou[c] for c in present_classes])
            if present_classes
            else 0.0
        )
        freq_weighted_iou = sum(
            ((np.sum(confmat[c, :]) + fn_ignore[c]) / total_points_cm)
            * per_class_iou[c]
            for c in range(num_classes)
        )
        return global_acc, mean_class_acc, mIoU, freq_weighted_iou

    gacc1, macc1, miou1, fiou1 = compute_confmat_metrics(
        confusion_mat_top1, fn_ignore_top1
    )
    if eval_top3:
        gacc3, macc3, miou3, fiou3 = compute_confmat_metrics(
            confusion_mat_top3, fn_ignore_top3
        )
    else:
        gacc3 = macc3 = miou3 = fiou3 = None

    excluded_indices = [
        i for i, name in enumerate(top100_label) if name in EXCLUDED_CLASS_NAMES
    ]

    def compute_mean_acc_excluded(confmat, fn_ignore, excluded_indices):
        """
        Mean per‑class accuracy ignoring the classes in `excluded_indices`.
        Mirrors the logic used for overall mAcc, but filters the unwanted classes.
        """
        acc_vals = []
        for c in range(num_classes):
            if c in excluded_indices:
                continue
            tp = confmat[c, c]
            fn = np.sum(confmat[c, :]) - tp + fn_ignore[c]
            gt = tp + fn  # points of this gt‑class
            if gt > 0:
                acc_vals.append(tp / gt)
        return np.mean(acc_vals) if acc_vals else 0.0

    def compute_excluded(confmat, fn_ignore, excluded_indices):
        acc_list, iou_list, total_points_ex = [], [], 0
        for c in range(num_classes):
            if c in excluded_indices:
                continue
            pts = np.sum(confmat[c, :]) + fn_ignore[c]
            if pts > 0:
                total_points_ex += pts
                tp = confmat[c, c]
                fn = np.sum(confmat[c, :]) - tp + fn_ignore[c]
                fp = np.sum(confmat[:, c]) - tp
                denom = tp + fp + fn
                iou = tp / denom if denom > 0 else 0.0
                gt_count = tp + fn
                acc_list.append(tp / gt_count if gt_count > 0 else 0.0)
                iou_list.append(iou)
        mean_acc_ex = np.mean(acc_list) if acc_list else 0.0
        miou_ex = np.mean(iou_list) if iou_list else 0.0
        freq_iou_ex = 0.0
        if total_points_ex > 0:
            for c in range(num_classes):
                if c in excluded_indices:
                    continue
                pts = np.sum(confmat[c, :]) + fn_ignore[c]
                if pts > 0:
                    tp = confmat[c, c]
                    fn = np.sum(confmat[c, :]) - tp + fn_ignore[c]
                    fp = np.sum(confmat[:, c]) - tp
                    denom = tp + fp + fn
                    iou = tp / denom if denom > 0 else 0.0
                    freq_iou_ex += (pts / total_points_ex) * iou
        return miou_ex, freq_iou_ex

    miou1_ex, fiou1_ex = compute_excluded(
        confusion_mat_top1, fn_ignore_top1, excluded_indices
    )
    macc1_ex = compute_mean_acc_excluded(
        confusion_mat_top1, fn_ignore_top1, excluded_indices
    )
    if eval_top3:
        miou3_ex, fiou3_ex = compute_excluded(
            confusion_mat_top3, fn_ignore_top3, excluded_indices
        )
        macc3_ex = compute_mean_acc_excluded(
            confusion_mat_top3, fn_ignore_top3, excluded_indices
        )
    else:
        miou3_ex = fiou3_ex = None

    # Print Results
    print("total valid points:", total_points)
    print("\n======== RESULTS ========")
    print("Top-1 Evaluation:")
    print(
        f"  Global Accuracy:           {gacc1:.4f}  (or {top1_correct / total_points:.4f} by vertex-level check)"
    )
    print(f"  Mean Per-Class Accuracy:     {macc1:.4f}")
    print(f"  Mean IoU (mIoU):             {miou1:.4f}")
    print(f"  Foreground excluding classes: {EXCLUDED_CLASS_NAMES}")
    print(f"  Foreground mIoU:             {miou1_ex:.4f}")
    print(f"  Foreground mAcc:              {macc1_ex:.4f}")

    # If requested, print per-class Top-1 IoU
    if args.print_class_iou:
        print("\nPer-Class Top-1 IoU:")
        for c in range(num_classes):
            tp = confusion_mat_top1[c, c]
            fn = np.sum(confusion_mat_top1[c, :]) - tp + fn_ignore_top1[c]
            fp = np.sum(confusion_mat_top1[:, c]) - tp
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            print(f"{top100_label[c]:<25}: {iou:.4f}")

    if eval_top3:
        print("\nTop-3 Evaluation:")
        print(
            f"  Global Accuracy:           {gacc3:.4f}  (or {top3_correct / total_points:.4f} by vertex-level check)"
        )
        print(f"  Mean Per-Class Accuracy:     {macc3:.4f}")
        print(f"  Mean IoU (mIoU):             {miou3:.4f}")
        print(f"  Foreground excluding classes: {EXCLUDED_CLASS_NAMES}")
        print(f"  Foreground mIoU: {miou3_ex:.4f}")
        print(f"  Foreground mAcc:             {macc3_ex:.4f}")

    print("\nDone!")
    gc.collect()

    # Restore stdout and save results to file
    sys.stdout = stdout_original
    results_str = "".join(results_capture)
    save_results_to_file(log_path, results_str, args)


if __name__ == "__main__":
    main()
