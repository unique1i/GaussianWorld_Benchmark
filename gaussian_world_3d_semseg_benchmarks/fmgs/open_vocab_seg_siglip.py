import os
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree

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
# 2. SigLIP Relevancy Scoring
###################################


@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,  # shape: (N, feature_dim)
    text_feat: torch.Tensor,  # shape: (C, feature_dim)
    canon_feat: torch.Tensor,  # shape: (K, feature_dim); only used if flag is False
    device: torch.device,
    use_siglip_probabilities: bool = True,
):
    """
    Computes the predicted label for each of the N language feature vectors.

    Two modes are supported:

    (a) fllow equation in LERF, use canonical phrases:
       For each text class c, it computes:
         Score_c = min_i [ exp(lang · text_c) / (exp(lang · canon_i) + exp(lang · text_c)) ]
       and then returns the argmax over classes.

    (b) SigLIP mode (use_siglip_probabilities=True):
       It computes the dot product between lang_feat and each text_feat to obtain logits,
       applies a sigmoid to obtain probabilities (as done in SigLIP),
       and then returns the argmax over the resulting probability scores.

    Returns:
      pred_label : numpy array of shape (N,) with predicted class indices.
    """
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)

    print("use siglip probabilities: ", use_siglip_probabilities)

    if use_siglip_probabilities:
        # Directly compute logits, then apply sigmoid.
        logits = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        probs = torch.sigmoid(logits)  # (N, C)
        # For each language feature, find the maximum probability and its index.
        max_probs, max_indices = torch.max(probs, dim=1)
        # Set predictions with max probability below a threshold to the ignore index.
        pred_label = max_indices.clone()
        pred_label[max_probs < 0.1] = -1
        return pred_label.cpu().numpy()
    else:
        # using canonical features.
        canon_feat = canon_feat.to(device, non_blocking=True)
        dot_lang_text = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        dot_lang_canon = torch.matmul(lang_feat, canon_feat.t())  # (N, K)
        exp_lang_text = dot_lang_text.exp()  # (N, C)
        exp_lang_canon = dot_lang_canon.exp()  # (N, K)
        N, C = dot_lang_text.shape

        relevancy_scores = []
        for c_idx in range(C):
            text_c_exp = exp_lang_text[:, c_idx].unsqueeze(-1)  # (N, 1)
            ratio_c = text_c_exp / (exp_lang_canon + text_c_exp)  # (N, K)
            score_c = torch.min(ratio_c, dim=1).values  # (N,)
            relevancy_scores.append(score_c)

        relevancy_matrix = torch.stack(relevancy_scores, dim=0).t()  # (N, C)
        pred_label = torch.argmax(relevancy_matrix, dim=1)
        return pred_label.cpu().numpy()


###################################
# 3. Main Evaluation Script
###################################


def main():
    # ------------------------------
    # Paths & Setup
    # ------------------------------
    val_split_path = "/home/yli7/scratch2/datasets/scannetpp_v1/splits/nvs_sem_val.txt"
    val_split_path = "temp.txt"
    scannetpp_preprocessed_root = (
        "/home/yli7/scratch2/datasets/ptv3_preprocessed/scannetpp_v1_preprocessed"
    )

    # 3DGS data and language features
    scannetpp_3dgs_root = (
        "/home/yli7/scratch2/datasets/gaussian_world/scannetpp_v1_mcmc_3dgs"
    )
    scannetpp_langfeat_root = "/home/yli7/scratch2/datasets/gaussian_world/scannetpp_v1_mcmc_3dgs/language_features"

    # top-100 classes text file
    text_path = "/home/yli7/scratch2/datasets/scannetpp_v1/metadata/semantic_benchmark/top100.txt"

    EXCLUDED_CLASS_NAMES = ["wall", "floor", "ceiling"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Option to choose computation method: set to True to use direct SigLIP probabilities.
    use_siglip_method = True

    # ------------------------------
    # 1) Load validation scenes
    # ------------------------------
    scene_ids = load_scene_list(val_split_path)
    print(f"Found {len(scene_ids)} validation scenes.")

    # ------------------------------
    # 2) Load SigLIP model & text embeddings
    # ------------------------------
    from transformers import AutoModel, AutoTokenizer

    CROP_SIZE = 512
    siglip_spec = "siglip-base-patch16-512"
    model = AutoModel.from_pretrained(
        f"google/{siglip_spec}"
    )  # siglip-so400m-patch14-384, siglip-base-patch16-224, siglip-base-patch16-384
    tokenizer = AutoTokenizer.from_pretrained(f"google/{siglip_spec}")
    model = model.eval().to(device)

    # Read top-100 label names
    with open(text_path, "r") as f:
        lines = f.readlines()
    top100_label_init = [
        line.strip() for line in lines
    ]  # e.g. ["wall", "floor", "chair", ...]
    top100_label_prompt = ["this is a " + label for label in top100_label_init]

    # Encode top-100 text using SigLIP.
    with torch.no_grad():
        text_inputs = tokenizer(
            top100_label_prompt, padding="max_length", return_tensors="pt"
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_feat = model.get_text_features(**text_inputs)  # (100, feature_dim)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu()

    # Encode canonical phrases (only used if not using the direct SigLIP mode)
    canonical_phrases = ["object", "things", "stuff", "texture"]
    with torch.no_grad():
        canon_inputs = tokenizer(
            canonical_phrases, padding="max_length", return_tensors="pt"
        )
        canon_inputs = {k: v.to(device) for k, v in canon_inputs.items()}
        canon_feat = model.get_text_features(**canon_inputs)
        canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
        canon_feat = canon_feat.cpu()

    num_classes = len(top100_label_init)

    # ------------------------------
    # 3) Prepare confusion matrix
    # ------------------------------
    confusion_mat = np.zeros(
        (num_classes, num_classes), dtype=np.int64
    )  # used to log our predictions for each class

    # ------------------------------
    # 4) Loop over scenes
    # ------------------------------
    for scene_id in tqdm(scene_ids, desc="Scene", dynamic_ncols=True):
        scene_preproc_folder = os.path.join(
            scannetpp_preprocessed_root, "val", scene_id
        )
        if not os.path.isdir(scene_preproc_folder):
            print(f"[Warning] Preprocessed folder not found: {scene_preproc_folder}")
            continue

        # (a) Load coordinate and label from segment.npy
        coord_path = os.path.join(scene_preproc_folder, "coord.npy")
        segment_path = os.path.join(scene_preproc_folder, "segment.npy")
        if not os.path.isfile(coord_path) or not os.path.isfile(segment_path):
            print(f"[Warning] Missing coord.npy or segment.npy for scene {scene_id}")
            continue
        coord = np.load(coord_path)  # shape (N, 3)
        segment = np.load(segment_path)  # shape (N, 3); first column => top-100 index
        labeled_gt = segment[:, 0]  # int16 in [0..99] or ignore_index (<0)

        valid_mask = labeled_gt >= 0
        if np.sum(valid_mask) == 0:
            continue

        xyz_val = coord[valid_mask]  # shape (M, 3)
        gt_val = labeled_gt[valid_mask].astype(np.int64)  # (M,)

        # (b) Load 3DGS and language features
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
        gauss_lang_feat = torch.load(langfeat_path)[0].cpu()  # shape (G, feature_dim)
        print(
            f"\nLoaded {gauss_xyz.shape[0]} 3DGS and {gauss_lang_feat.shape[0]} language features for {scene_id}"
        )

        # Filter out zero vectors
        norms = gauss_lang_feat.norm(dim=1)
        keep_mask = norms > 0
        gauss_xyz = gauss_xyz[keep_mask.numpy()]
        gauss_lang_feat = gauss_lang_feat[keep_mask]
        if gauss_xyz.shape[0] == 0:
            print(f"[Warning] All 3DGS zero feats in {scene_id}")
            continue

        # (c) Build KDTree for NN search
        kd_tree = KDTree(gauss_xyz)
        _, nn_idx = kd_tree.query(xyz_val)  # shape (M,)
        nn_lang_feat = gauss_lang_feat[nn_idx]  # (M, feature_dim)

        # (d) Compute relevancy scores; choose the method based on use_siglip_method flag.
        pred_label = compute_relevancy_scores(
            nn_lang_feat,
            text_feat,  # (100, feature_dim)
            canon_feat,  # (4, feature_dim); will be ignored if use_siglip_probabilities=True
            device=device,
            use_siglip_probabilities=use_siglip_method,
        )  # shape (M,)

        # (e) Accumulate confusion matrix
        fn_ignore = np.zeros(num_classes, dtype=np.int64)
        for gt_c, pr_c in zip(gt_val, pred_label):
            if pr_c == -1:
                fn_ignore[gt_c] += 1  # track ignored predictions
            else:
                confusion_mat[gt_c, pr_c] += 1

    # ------------------------------
    # 5) Compute IoU, fIoU, and Accuracy
    # ------------------------------

    # (a) Per-class IoU
    ious = []
    for c in range(num_classes):
        tp = confusion_mat[c, c]
        fn = np.sum(confusion_mat[c, :]) - tp + fn_ignore[c]  # total misses + ignored
        fp = np.sum(confusion_mat[:, c]) - tp
        denom = tp + fp + fn
        iou_c = tp / denom if denom > 0 else 0.0
        ious.append(iou_c)

    # (b) Identify which classes are actually present in the dataset
    #     (i.e., have at least one ground-truth sample).
    #     We'll also build a set of excluded indices based on EXCLUDED_CLASS_NAMES.
    present_mask = (np.sum(confusion_mat, axis=1) + fn_ignore) > 0
    excluded_indices = [
        i for i, name in enumerate(top100_label_init) if name in EXCLUDED_CLASS_NAMES
    ]

    # (c) Mean IoU (mIoU) over all present classes
    present_classes = [c for c in range(num_classes) if present_mask[c]]
    mean_iou = (
        np.mean([ious[c] for c in present_classes]) if len(present_classes) > 0 else 0.0
    )

    # (d) Frequency-weighted IoU (fIoU) over all present classes
    #     freq_c = (# of GT samples in class c) / (total # of GT samples)
    #     fIoU = sum_c(freq_c * iou_c)
    total_gt = 0
    for c in range(num_classes):
        # ground-truth count for class c includes confusion_mat[c, :].sum() + fn_ignore[c]
        total_gt += np.sum(confusion_mat[c, :]) + fn_ignore[c]

    fiou = 0.0
    if total_gt > 0:
        for c in range(num_classes):
            gt_count_c = np.sum(confusion_mat[c, :]) + fn_ignore[c]
            freq_c = gt_count_c / total_gt
            fiou += freq_c * ious[c]

    # (e) Excluding certain classes from the metrics
    included_indices = [
        c for c in range(num_classes) if present_mask[c] and (c not in excluded_indices)
    ]
    mean_iou_excl = (
        np.mean([ious[c] for c in included_indices])
        if len(included_indices) > 0
        else 0.0
    )

    # Frequency-weighted IoU ignoring excluded classes
    total_gt_incl = 0
    for c in included_indices:
        total_gt_incl += np.sum(confusion_mat[c, :]) + fn_ignore[c]

    fiou_excl = 0.0
    if total_gt_incl > 0:
        for c in included_indices:
            gt_count_c = np.sum(confusion_mat[c, :]) + fn_ignore[c]
            freq_c = gt_count_c / total_gt_incl
            fiou_excl += freq_c * ious[c]

    # (f) Global accuracy & mean class accuracy
    total_correct = np.sum(np.diag(confusion_mat))
    total_count = np.sum(confusion_mat)
    global_acc = total_correct / (total_count + 1e-12)

    per_class_acc = []
    for c in range(num_classes):
        gt_count_c = np.sum(confusion_mat[c, :])
        if gt_count_c > 0:
            acc_c = confusion_mat[c, c] / gt_count_c
            per_class_acc.append(acc_c)
    mean_class_acc = np.mean(per_class_acc) if len(per_class_acc) > 0 else 0.0

    # ------------------------------
    # Print results
    # ------------------------------
    print("\n======== RESULTS ========")
    print("Present classes:", [top100_label_init[c] for c in present_classes])
    missing_classes = [
        top100_label_init[i] for i in range(num_classes) if not present_mask[i]
    ]
    print("Missing classes:", missing_classes)
    print("\n--- Per-class IoU (all classes) ---")
    for c in range(num_classes):
        if present_mask[c]:
            print(f"  {top100_label_init[c]:20s} IoU={ious[c]:.4f}")

    print("\n--- Metrics (ALL present classes) ---")
    print(f"Mean IoU (mIoU)      : {mean_iou:.4f}")
    print(f"Frequency-weighted IoU (fIoU) : {fiou:.4f}")
    print(f"Global Accuracy      : {global_acc:.4f}")
    print(f"Mean Class Accuracy  : {mean_class_acc:.4f}")

    print("\n--- Metrics (EXCLUDING: {}) ---".format(EXCLUDED_CLASS_NAMES))
    print(f"Mean IoU (excl)      : {mean_iou_excl:.4f}")
    print(f"Frequency-weighted IoU (excl) : {fiou_excl:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
