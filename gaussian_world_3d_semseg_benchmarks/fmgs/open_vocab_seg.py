import os
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
import open_clip
from tqdm import tqdm

###################################
# 1. Basic I/O Utilities
###################################


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
):
    """
    Computes the "relevancy score" for each of the C text classes
    for each of the N 3DGS embeddings, returning the argmax along classes.

    Score_c = min_i [ exp(lang . text_c) / (exp(lang . canon_i) + exp(lang . text_c)) ].

    Returns predicted_label : (N,) in [0..C-1]
    """
    # Move to device
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    canon_feat = canon_feat.to(device, non_blocking=True)

    dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
    dot_lang_canon = torch.matmul(lang_feat, canon_feat.t())  # (N, K)

    exp_lang_text = dot_lang_text.exp()  # (N, C)
    exp_lang_canon = dot_lang_canon.exp()  # (N, K)

    N, C = dot_lang_text.shape
    K = dot_lang_canon.shape[1]

    relevancy_scores = []
    for c_idx in range(C):
        # ratio_c = exp_lang_text[:, c_idx] / (exp_lang_canon + exp_lang_text[:, c_idx])
        # Then take min over i in [0..K-1].
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


def main():
    # ------------------------------
    # Paths & Setup
    # ------------------------------
    # val_split_path = "/home/yli7/scratch2/datasets/scannetpp_v1/splits/nvs_sem_val.txt"
    val_split_path = "temp.txt"
    scannetpp_preprocessed_root = "/home/yli7/scratch2/datasets/scannetpp_preprocessed"

    # Root where your 3DGS + CLIP feats are stored
    # scannetpp_3dgs_root = "/home/yli7/scratch2/datasets/gaussian_world/scannetpp_3dgs_default_depth_true"
    scannetpp_3dgs_root = "/home/yli7/scratch2/outputs/scannetpp_v1_default_fix_xyz_gs"
    # scannetpp_langfeat_root = "/home/yli7/scratch2/datasets/gaussian_world/scannetpp_lang_feat"
    scannetpp_langfeat_root = (
        "/home/yli7/scratch2/outputs/scannetpp_v1_default_fix_xyz_gs/language_features"
    )

    # The top-100 classes text file
    text_path = "/home/yli7/scratch2/datasets/scannetpp_v1/metadata/semantic_benchmark/top100.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # 1) Load validation scenes
    # ------------------------------
    scene_ids = load_scene_list(val_split_path)
    print(f"Found {len(scene_ids)} validation scenes.")

    # ------------------------------
    # 2) Load CLIP model & text embeddings
    # ------------------------------
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    model = model.eval().to(device)

    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Read top-100 label names
    with open(text_path, "r") as f:
        lines = f.readlines()
    top100_label_init = [
        line.strip() for line in lines
    ]  # e.g. ["wall", "floor", "chair", ...]
    top100_label_prompt = ["this is a " + label for label in top100_label_init]

    # Encode top-100 text
    with torch.no_grad():
        text_tokens = tokenizer(top100_label_prompt)
        text_feat = model.encode_text(text_tokens.to(device))  # (100, 512)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu()

    # Encode canonical phrases
    canonical_phrases = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        canon_tokens = tokenizer(canonical_phrases)
        canon_feat = model.encode_text(canon_tokens.to(device))
        canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = canon_feat.cpu()

    num_classes = len(top100_label_init)

    # ------------------------------
    # 3) Prepare confusion matrix
    # ------------------------------
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ------------------------------
    # 4) Loop over scenes
    # ------------------------------
    for scene_id in tqdm(scene_ids, desc="Scene", dynamic_ncols=True):
        # Our official preprocessed data is in scannetpp_preprocessed_root/<split>/<scene_id>/
        # Here we are evaluating the "val" split. So:
        scene_preproc_folder = os.path.join(
            scannetpp_preprocessed_root, "val", scene_id
        )
        if not os.path.isdir(scene_preproc_folder):
            print(f"[Warning] Preprocessed folder not found: {scene_preproc_folder}")
            continue

        # (a) Load coordinate + label from segment.npy
        coord_path = os.path.join(scene_preproc_folder, "coord.npy")
        segment_path = os.path.join(scene_preproc_folder, "segment.npy")

        if not os.path.isfile(coord_path) or not os.path.isfile(segment_path):
            print(f"[Warning] Missing coord.npy or segment.npy for scene {scene_id}")
            continue

        coord = np.load(coord_path)  # shape (N, 3)
        segment = np.load(segment_path)  # shape (N, 3); first column => top-100 index
        labeled_gt = segment[:, 0]  # int16 in [0..99] or ignore_index (<0)

        # Keep only valid labels
        valid_mask = labeled_gt >= 0
        if np.sum(valid_mask) == 0:
            continue

        xyz_val = coord[valid_mask]  # shape (M, 3)
        gt_val = labeled_gt[valid_mask].astype(np.int64)  # (M,)

        # (b) Load 3DGS & CLIP feats
        scene_3dgs_folder = os.path.join(scannetpp_3dgs_root, scene_id)
        three_dgs_ckpt = os.path.join(
            scene_3dgs_folder, "ckpts", "point_cloud_30000.ply"
        )
        if not os.path.isfile(three_dgs_ckpt):
            print(f"[Warning] 3DGS .ply not found for scene {scene_id}")
            continue
        gauss_xyz, _ = read_ply_file_3dgs(three_dgs_ckpt)

        lang_feat_folder = os.path.join(scannetpp_langfeat_root, scene_id)
        langfeat_path = os.path.join(lang_feat_folder, "langfeat.pth")
        if not os.path.isfile(langfeat_path):
            print(f"[Warning] Language feature .pth not found for scene {scene_id}")
            continue
        gauss_lang_feat = torch.load(langfeat_path)[0].cpu()  # (G, 512)
        print(
            f"\nLoaded {gauss_xyz.shape[0]} 3DGS and {gauss_lang_feat.shape[0]} language features for {scene_id}"
        )

        # Filter out zero vectors if needed
        norms = gauss_lang_feat.norm(dim=1)
        keep_mask = norms > 0
        gauss_xyz = gauss_xyz[keep_mask.numpy()]
        gauss_lang_feat = gauss_lang_feat[keep_mask]
        if gauss_xyz.shape[0] == 0:
            print(f"[Warning] All 3DGS zero feats in {scene_id}")
            continue

        # (c) Build KDTree => NN search
        kd_tree = KDTree(gauss_xyz)
        _, nn_idx = kd_tree.query(xyz_val)  # shape (M,)
        nn_lang_feat = gauss_lang_feat[nn_idx]  # (M, 512)

        # (d) Relevancy scoring => predicted label
        pred_label = compute_relevancy_scores(
            nn_lang_feat,
            text_feat,  # (100, 512)
            canon_feat,  # (4, 512)
            device=device,
        )  # shape (M,)

        # (e) Accumulate confusion
        for gt_c, pr_c in zip(gt_val, pred_label):
            confusion_mat[gt_c, pr_c] += 1

    # ------------------------------
    # 5) Compute IoU and Accuracy
    # ------------------------------
    ious = []
    for c in range(num_classes):
        tp = confusion_mat[c, c]
        fn = np.sum(confusion_mat[c, :]) - tp
        fp = np.sum(confusion_mat[:, c]) - tp
        denom = tp + fp + fn
        if denom > 0:
            iou_c = tp / denom
        else:
            iou_c = 0.0
        ious.append(iou_c)
    # get the mask of classes that has non-zero gt count
    mask = (np.sum(confusion_mat, axis=1) > 0).astype(np.int32)
    mean_iou = np.mean([iou for iou, m in zip(ious, mask) if m == 1])

    # Global accuracy
    total_correct = np.sum(np.diag(confusion_mat))
    total_count = np.sum(confusion_mat)
    global_acc = total_correct / (total_count + 1e-12)

    # Mean class accuracy
    per_class_acc = []
    for c in range(num_classes):
        gt_count_c = np.sum(confusion_mat[c, :])
        if gt_count_c > 0:
            acc_c = confusion_mat[c, c] / gt_count_c
            per_class_acc.append(acc_c)
        else:
            acc_c = 0.0
    mean_class_acc = np.mean(per_class_acc)

    # Print final
    print("\n======== RESULTS ========")
    print(
        "Per-class IoU:",
        {top100_label_init[c]: round(ious[c], 4) for c in range(num_classes)},
    )
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Mean Class Accuracy: {mean_class_acc:.4f}")


if __name__ == "__main__":
    main()
