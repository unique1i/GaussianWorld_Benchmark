from plyfile import PlyData
import torch
import numpy as np
import os
from argparse import ArgumentParser
import open_clip
from tqdm import tqdm
from scipy.spatial import cKDTree as KDTree
import sys
from scene import SemanticModel
from torch.nn.functional import softmax
from metadata.matterport3d import MATTERPORT_LABELS_21, MATTERPORT_LABELS_160
import datetime


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


def load_scene_list(gt_scene_dir):
    # Check if the directory exists
    if not os.path.isdir(gt_scene_dir):
        raise ValueError(f"Directory {gt_scene_dir} does not exist")

    # Get all entries in the directory
    all_entries = os.listdir(gt_scene_dir)

    folder_names = [
        entry.replace(".cache", "")
        for entry in all_entries
        if os.path.isdir(os.path.join(gt_scene_dir, entry))
    ]

    return folder_names


def load_ply(path):
    semantic_dim = 10

    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    sem_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("sem_")
    ]
    sem_names = sorted(sem_names, key=lambda x: int(x.split("_")[-1]))
    sems = np.zeros((xyz.shape[0], len(sem_names) or semantic_dim))
    if len(sem_names) == semantic_dim:
        for idx, attr_name in enumerate(sem_names):
            sems[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, sems


@torch.no_grad()
def compute_relevancy_scores(
    lang_feat: torch.Tensor,
    text_feat: torch.Tensor,
    device: torch.device,
    bench_name: str = "",
):
    lang_feat = lang_feat.to(device, non_blocking=True)
    text_feat = text_feat.to(device, non_blocking=True)

    logits = torch.matmul(lang_feat, text_feat.t())  # (N, C)
    probs = torch.sigmoid(logits)  # (N, C)
    top1_probs, top1_indices = torch.topk(probs, k=1, dim=1)  # (N, 1)
    mask = top1_probs[:, 0] > 0.0
    top1_indices[~mask] = -1  # if lang_feat is all zeros
    return top1_indices.cpu().numpy()


def parse_args():
    parser = ArgumentParser(description="Open‑Vocal 3DGS semantic evaluation")
    parser.add_argument(
        "--gt_scene_dir",
        type=str,
        default="/scratch/joanna_cheng/matterport3d_test_set_preprocessed",
    )
    parser.add_argument(
        "--gs_root",
        type=str,
        default="/scratch/joanna_cheng/matterport3d_region_test_set_suite/mcmc_3dgs",
    )
    parser.add_argument(
        "--pred_scene_dir", type=str, default="/scratch/joanna_cheng/matterport3d_pred"
    )
    parser.add_argument(
        "--use_dot_similarity",
        action="store_true",
        help="If set, use plain CLIP dot similarity instead of ratio scoring.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_dot_similarity = True
    print_class_iou = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    nn_num = 25
    ignore_classes = ["other furniture", "wall", "floor", "ceiling"]

    # Setup for results logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"results_full_matterport3d_nn_num_{nn_num}_clip.txt"
    log_path = os.path.join("logs", log_filename)

    # Capture all printed output
    stdout_original = sys.stdout
    results_capture = []

    class CaptureOutput:
        def write(self, text):
            results_capture.append(text)
            stdout_original.write(text)

        def flush(self):
            stdout_original.flush()

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
    scene_ids = load_scene_list(args.gt_scene_dir)
    print(f"Found {len(scene_ids)} validation scenes.")

    # ------------------------------
    # 2) Load CLIP model (once)
    # ------------------------------
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Generate text embeddings for ScanNet20 and ScanNet200
    def prepare_text_features(class_labels):
        inputs = tokenizer(class_labels)
        text_feat = model.encode_text(inputs.to(device))
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat.cpu()

    text_prompts_21 = ["this is a " + label for label in MATTERPORT_LABELS_21]
    text_prompts_160 = ["this is a " + label for label in MATTERPORT_LABELS_160]
    text_feat_21 = prepare_text_features(text_prompts_21)
    text_feat_160 = prepare_text_features(text_prompts_160)

    # ------------------------------
    # 3) Evaluate
    # ------------------------------

    # Initialize metrics for both benchmarks
    benchmarks = [
        {
            "name": "matterport3d_semseg_21",
            "text_feat": text_feat_21,
            "class_labels": MATTERPORT_LABELS_21,
            "confusion_mat": np.zeros(
                (len(MATTERPORT_LABELS_21), len(MATTERPORT_LABELS_21)), dtype=np.int64
            ),
            "fn_ignore": np.zeros(len(MATTERPORT_LABELS_21), dtype=np.int64),
            "total_points": 0,
            "top1_correct": 0,
        }
    ]

    # ---- 3.1 Load label names & encode text ----
    for scene_id in tqdm(scene_ids, desc="Processing scenes"):
        scene_preproc_folder = os.path.join(args.gt_scene_dir, scene_id)
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
        segment21 = np.load(segment_path)

        # ---- 3.2.b Load 3DGS & CLIP feats ----
        scene_3dgs_folder = os.path.join(args.pred_scene_dir, scene_id)
        ply_path = os.path.join(
            scene_3dgs_folder, "point_cloud/iteration_1500_lvl_0", "point_cloud.ply"
        )
        if not os.path.isfile(ply_path):
            print(f"[Warning] 3DGS .ply not found for scene {scene_id}")
            continue

        with torch.no_grad():
            gauss_xyz, gauss_lang_feat = load_ply(ply_path)
            gauss_lang_feat = (
                torch.from_numpy(gauss_lang_feat).float().to(device)
            )  # (G, 10)

            lut_model_path = os.path.join(
                scene_3dgs_folder, "point_cloud/iteration_1500_lvl_0", "LUT.pt"
            )
            LUT = torch.load(lut_model_path).to(device)

            mlp_model_path = os.path.join(
                scene_3dgs_folder, "point_cloud/iteration_1500_lvl_0", "semantic_MLP.pt"
            )
            MLP = SemanticModel.load(mlp_model_path).to(device)

            gauss_lang_label = MLP(gauss_lang_feat)  # (G, 512)
            sem_logit = softmax(gauss_lang_label * 10, dim=-1).argmax(
                dim=-1
            )  # (G, 512)
            gauss_lang_feat = LUT[sem_logit]  # (G, 512)

            norms = gauss_lang_feat.norm(dim=1)
            gauss_lang_feat = gauss_lang_feat / gauss_lang_feat.norm(
                dim=-1, keepdim=True
            )

            keep_mask_gs = norms > 0
            gauss_xyz = gauss_xyz[keep_mask_gs.cpu().numpy()]
            gauss_lang_feat = gauss_lang_feat[keep_mask_gs]

        if gauss_xyz.shape[0] == 0:
            print(f"[Warning] All 3DGS zero feats in {scene_id}")
            continue

        for bench in benchmarks:
            # Select current benchmark data
            current_segment = segment21
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
                    batch_feat, text_feat, device, bench["name"]
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
            i for i, name in enumerate(bench["class_labels"]) if name in ignore_classes
        ]
        print(f"Excluded indices: {excluded_indices}, classes: {ignore_classes}")
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
        print(f"Foreground excluding classes: {ignore_classes}")
        print(f"Foreground mIoU: {fg_miou:.4f}")
        print(f"Foreground mAcc: {fg_macc:.4f}")

        # Print per-class IoU if requested
        if print_class_iou:
            print("\nPer-class IoU:")
            for c in range(num_classes):
                if not np.isnan(iou_array[c]):
                    # only print class that presents during evaluation
                    class_name = bench["class_labels"][c]
                    print(f"{class_name:<20}: {iou_array[c]:.4f}")

    # Restore stdout and save results to file
    sys.stdout = stdout_original
    results_str = "".join(results_capture)

    save_results_to_file(log_path, results_str, args)


if __name__ == "__main__":
    main()
