import os
import numpy as np
import torch
import sys
import argparse
import open_clip

from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
from metadata.scannet200_constants import CLASS_LABELS_20, CLASS_LABELS_200
from transformers import AutoModel, AutoTokenizer


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


def clustering_voting(pred, instance_labels, ignore_index):
    """
    Args:
        pred (np.ndarray): Predicted semantic labels for each point, shape (N,)
        instance_labels (np.ndarray): Instance ID for each point, shape (N,)
        ignore_index (int): Instance ID value to ignore (e.g., -1 for background)
    Returns:
        np.ndarray: Updated semantic predictions with consistent labels per instance
    """
    # Ensure inputs have the same shape
    if pred.shape != instance_labels.shape:
        print(
            "clustering_voting: prediction and instance arrays must have the same shape, got {} and {}".format(
                pred.shape, instance_labels.shape
            )
        )
        return pred

    updated_pred = pred.copy()
    unique_instances = np.unique(instance_labels)
    valid_instances = unique_instances[unique_instances != ignore_index]

    for instance_id in valid_instances:
        instance_mask = instance_labels == instance_id
        instance_preds = pred[instance_mask]
        unique_classes, counts = np.unique(instance_preds, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        updated_pred[instance_mask] = majority_class

    return updated_pred


@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,
    text_feat: torch.Tensor,
    canon_feat: torch.Tensor,
    device: torch.device,
    use_siglip_probabilities: bool = True,
    bench_name: str = "",
):
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)
    if use_siglip_probabilities:
        logits = torch.matmul(lang_feat, text_feat.t())  # (N, C)
        probs = torch.sigmoid(logits)  # (N, C)
        top1_probs, top1_indices = torch.topk(probs, k=1, dim=1)  # (N, 1)
        mask = top1_probs[:, 0] > 0.0
        top1_indices[~mask] = -1  # if lang_feat is all zeros
        # if bench_name == "scannet20":
        #     top1_indices[~mask] = -1 # use "other" class
        # else:
        #     top1_indices[~mask] = -1  # Use -1 as ignore index
        return top1_indices.cpu().numpy()
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


def save_results_to_file(log_path, results_str, args, enable_clustering):
    """Save the results to a text file with relevant parameters in the filename."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Write results to the file
    with open(log_path, "w") as f:
        # Write command line arguments first
        f.write("Command line arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"enable_clustering: {enable_clustering}\n")
        f.write("\n")

        # Write experiment results
        f.write(results_str)

    print(f"\nResults saved to: {log_path}")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--val_split_path", type=str)
    argparser.add_argument("--scannet_preprocessed_root", type=str)
    argparser.add_argument("--scannet_3dgs_root", type=str)
    argparser.add_argument("--nn_num", type=int)
    argparser.add_argument("--print_class_iou", action="store_true")
    argparser.add_argument(
        "--ignore_classes", nargs="+", default=["wall", "floor", "ceiling"]
    )
    argparser.add_argument("--model_spec", type=str, default="siglip2-base-patch16-512")
    argparser.add_argument(
        "--save_results", action="store_true", help="Save results to a file"
    )
    args = argparser.parse_args()

    model_name = "clip"
    val_split_path = (
        args.val_split_path
        or "/home/yli7/projects/yue/language_feat_exps/splits/scannet_val.txt"
    )
    scannet_preprocessed_root = (
        args.scannet_preprocessed_root
        or "/home/yli7/scratch2/datasets/ptv3_preprocessed/scannet_preprocessed"
    )
    scannet_3dgs_root = (
        args.scannet_3dgs_root
        or "/home/yli7/scratch2/datasets/gaussian_world/scannet_mcmc_3dgs"
    )  # scannet_default_fix_xyz_gs, scannet_mcmc_3dgs
    scannet_langfeat_root = os.path.join(
        scannet_3dgs_root, f"language_features_{model_name}"
    )  # language_features_siglip2, language_features_instance

    nn_num = args.nn_num or 25
    args.print_class_iou = True
    args.ignore_classes = ["wall", "floor", "ceiling"]

    enable_clustering = True
    # used for logging
    split_name = val_split_path.split("/")[-1].split(".")[0]
    method_name = "occam"
    gs_folder_name = os.path.basename(os.path.normpath(scannet_3dgs_root))
    log_path = f"logs/{method_name}_{gs_folder_name}_nn_num_{nn_num}__clustering_{enable_clustering}_{split_name}_{model_name}.txt"

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_siglip_probabilities = True

    # Load validation scenes
    scene_ids = load_scene_list(val_split_path)
    # scene_ids = [scene_ids[827]]
    print(f"Using {model_name.upper()} language features from {scannet_langfeat_root}")
    print(f"Found {len(scene_ids)} validation scenes.")
    print(f"use nn_num: {nn_num}")

    # Load SigLIP model & prepare text embeddings for both benchmarks
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

    # Generate text embeddings for ScanNet20 and ScanNet200
    def prepare_text_features(text_prompts):
        text_prompts = ["this is a " + name for name in text_prompts]
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

    text_feat_20 = prepare_text_features(CLASS_LABELS_20)
    text_feat_200 = prepare_text_features(CLASS_LABELS_200)

    # # Prepare canonical features
    # canonical_phrases = ["object", "things", "stuff", "texture"]
    # canon_inputs = tokenizer(canonical_phrases, padding="max_length", max_length=64, return_tensors="pt")
    # canon_inputs = {k: v.to(device) for k, v in canon_inputs.items()}
    # canon_feat = model.get_text_features(**canon_inputs)
    # canon_feat /= canon_feat.norm(dim=-1, keepdim=True)
    # canon_feat = canon_feat.cpu()

    # Initialize metrics for both benchmarks
    benchmarks = [
        {
            "name": "scannet20",
            "text_feat": text_feat_20,
            "class_labels": CLASS_LABELS_20,
            "confusion_mat": np.zeros(
                (len(CLASS_LABELS_20), len(CLASS_LABELS_20)), dtype=np.int64
            ),
            "fn_ignore": np.zeros(len(CLASS_LABELS_20), dtype=np.int64),
            "total_points": 0,
            "top1_correct": 0,
        },
        {
            "name": "scannet200",
            "text_feat": text_feat_200,
            "class_labels": CLASS_LABELS_200,
            "confusion_mat": np.zeros(
                (len(CLASS_LABELS_200), len(CLASS_LABELS_200)), dtype=np.int64
            ),
            "fn_ignore": np.zeros(len(CLASS_LABELS_200), dtype=np.int64),
            "total_points": 0,
            "top1_correct": 0,
        },
    ]

    # Process each scene
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        scene_folder = os.path.join(scannet_preprocessed_root, "val", scene_id)
        if not os.path.exists(scene_folder):
            scene_folder = os.path.join(scannet_preprocessed_root, "train", scene_id)
        if not os.path.exists(scene_folder):
            raise ValueError(
                f"Scene {scene_id} not found in {scannet_preprocessed_root}"
            )

        try:
            coord = np.load(os.path.join(scene_folder, "coord.npy"))
            instance = np.load(os.path.join(scene_folder, "instance.npy"))
            segment20 = np.load(os.path.join(scene_folder, "segment20.npy"))
            segment200 = np.load(os.path.join(scene_folder, "segment200.npy"))
        except:
            raise ValueError(
                f"Error loading data for scene {scene_id} in {scene_folder}"
            )

        gs_path = os.path.join(
            scannet_3dgs_root, scene_id, "ckpts", "point_cloud_30000.ply"
        )
        if not os.path.exists(gs_path):
            raise ValueError(
                f"Error loading Gaussian data for scene {scene_id} in {gs_path}"
            )

        gauss_xyz, _ = read_ply_file_3dgs(gs_path)
        lang_feat_path = os.path.join(scannet_langfeat_root, scene_id, "langfeat.pth")
        if os.path.exists(lang_feat_path):
            gauss_lang_feat = torch.load(lang_feat_path, map_location="cpu")[0]
        elif os.path.exists(
            os.path.join(scannet_langfeat_root, scene_id, "langfeat.npy")
        ):
            gauss_lang_feat = np.load(
                os.path.join(scannet_langfeat_root, scene_id, "langfeat.npy")
            )
            gauss_lang_feat = torch.from_numpy(gauss_lang_feat).float()
        else:
            # raise ValueError(f"Path error for scene {scene_id} in {lang_feat_path}")
            print(f"Path error for scene {scene_id} in {lang_feat_path}")
            continue

        for bench in benchmarks:
            # Select current benchmark data
            current_segment = segment20 if bench["name"] == "scannet20" else segment200
            text_feat = bench["text_feat"]
            class_labels = bench["class_labels"]
            num_classes = len(class_labels)

            # Each element in current_segment is an array of valid GT class indices for that point
            gt_labels = [row[row >= 0] for row in current_segment]
            valid_mask = np.array([len(labels) > 0 for labels in gt_labels], dtype=bool)
            if not np.any(valid_mask):
                continue

            xyz_val = coord[valid_mask]
            gt_val = [gt_labels[i] for i in np.where(valid_mask)[0]]
            # For simplicity, pick the "first" ground-truth label as the canonical GT
            gt_first = [g[0] for g in gt_val]

            # Compute predictions for the Gaussian samples
            batch_size = 128000
            gauss_labels = []
            for i in range(0, len(gauss_lang_feat), batch_size):
                batch_feat = gauss_lang_feat[i : i + batch_size].to(device)
                batch_pred = compute_relevancy_scores(
                    batch_feat,
                    text_feat,
                    None,
                    device,
                    use_siglip_probabilities,
                    bench["name"],
                )
                gauss_labels.append(batch_pred)
            gauss_labels = np.concatenate(gauss_labels, axis=0)

            # KDTree search and voting
            kd_tree = KDTree(gauss_xyz)
            _, nn_indices = kd_tree.query(xyz_val, k=nn_num)
            neighbor_labels = gauss_labels[nn_indices].squeeze(-1)

            # Voting logic
            top1_preds = []
            for neighbors in neighbor_labels:
                valid_neighbors = neighbors[
                    neighbors != -1
                ]  # ignore "no confident prediction"
                if len(valid_neighbors) == 0:
                    top1_preds.append(-1)
                else:
                    counts = np.bincount(valid_neighbors)
                    top1_preds.append(np.argmax(counts))
            top1_preds = np.array(top1_preds)

            # clustering voting
            if enable_clustering:
                top1_preds = clustering_voting(top1_preds, instance[valid_mask], -1)

            # Update metrics
            bench["total_points"] += len(gt_val)
            for i, (g, pred) in enumerate(zip(gt_val, top1_preds)):
                gt_c = gt_first[i]
                # Update confusion matrix
                if pred == -1:
                    bench["fn_ignore"][gt_c] += 1
                else:
                    if 0 <= gt_c < num_classes and 0 <= pred < num_classes:
                        bench["confusion_mat"][gt_c, pred] += 1

                # Update top-1 "correct" if the predicted label is in the GT set
                if pred in g:
                    bench["top1_correct"] += 1

    # Compute and print results for each benchmark
    for bench in benchmarks:
        print(f"\n=== Results for {bench['name'].upper()} ===")
        num_classes = len(bench["class_labels"])
        cm = bench["confusion_mat"]
        fn_ignore = bench["fn_ignore"]

        # Global accuracy
        global_acc = (
            bench["top1_correct"] / bench["total_points"]
            if bench["total_points"] > 0
            else 0
        )

        # Arrays to store IoU and Acc for each class (indexed by class ID)
        iou_array = np.full(num_classes, np.nan, dtype=float)
        acc_array = np.full(num_classes, np.nan, dtype=float)

        # Compute per-class metrics
        for c in range(num_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp + fn_ignore[c]
            # If no points of this class exist at all, skip
            if (tp + fn) == 0:
                continue

            acc = tp / (tp + fn)  # class accuracy
            denom = tp + fp + fn
            iou = (tp / denom) if denom > 0 else 0.0

            iou_array[c] = iou
            acc_array[c] = acc

        # Mean class accuracy / IoU over classes that are not NaN
        valid_mask = ~np.isnan(iou_array)
        valid_acc_vals = acc_array[valid_mask]
        valid_iou_vals = iou_array[valid_mask]
        macc = np.mean(valid_acc_vals) if len(valid_acc_vals) > 0 else 0
        miou = np.mean(valid_iou_vals) if len(valid_iou_vals) > 0 else 0

        # Foreground metrics: exclude user-specified classes (e.g. wall, floor, ceiling)
        excluded_indices = [
            i
            for i, name in enumerate(bench["class_labels"])
            if name in args.ignore_classes
        ]
        print(f"Excluded indices: {excluded_indices}")
        # We only consider classes that are valid and not in the excluded set
        fg_mask = valid_mask.copy()
        fg_mask[excluded_indices] = False

        fg_acc_vals = acc_array[fg_mask]
        fg_iou_vals = iou_array[fg_mask]
        fg_macc = np.mean(fg_acc_vals) if len(fg_acc_vals) > 0 else 0
        fg_miou = np.mean(fg_iou_vals) if len(fg_iou_vals) > 0 else 0

        print(f"Global Accuracy: {global_acc:.4f}")
        print(f"Mean Class Accuracy: {macc:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"Foreground excluding classes: {args.ignore_classes}")
        print(f"Foreground mIoU: {fg_miou:.4f}")
        print(f"Foreground mAcc: {fg_macc:.4f}")

        # Print per-class IoU if requested
        if args.print_class_iou:
            print("\nPer-class IoU:")
            for c in range(num_classes):
                if not np.isnan(iou_array[c]):
                    class_name = bench["class_labels"][c]
                    print(f"{class_name:<20}: {iou_array[c]:.4f}")

    # Restore stdout and save results to file
    sys.stdout = stdout_original
    results_str = "".join(results_capture)
    save_results_to_file(log_path, results_str, args, enable_clustering)


if __name__ == "__main__":
    main()
