#!/usr/bin/env python3
import os
import shutil
import argparse

def load_keep_list(split_path):
    """Read scene names (one per line) from a split file."""
    with open(split_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def infer_dataset_name(filename):
    """Take 'holicity_mini_val.txt' → 'holicity'."""
    return filename.split('_')[0]

def process_dataset(data_root, splits_dir, split_file, dry_run):
    dataset = infer_dataset_name(split_file)
    data_dir = os.path.join(data_root, dataset)
    if not os.path.isdir(data_dir):
        print(f"⚠️ Skipping '{dataset}': directory not found at {data_dir}")
        return

    keep = load_keep_list(os.path.join(splits_dir, split_file))
    all_scenes = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))]
    to_remove = [s for s in all_scenes if s not in keep]

    if not to_remove:
        print(f"[{dataset}] No extra scenes to remove.")
        return

    print(f"[{dataset}] Scenes to {'simulate removal' if dry_run else 'delete'} ({len(to_remove)}):")
    for scene in sorted(to_remove):
        scene_path = os.path.join(data_dir, scene, "clip")
        print("  ", scene_path)
        if not dry_run:
            shutil.rmtree(scene_path)

def main():
    parser = argparse.ArgumentParser(
        description="Prune scene folders not listed in split files."
    )
    parser.add_argument(
        "--data-root", "-d", required=True,
        help="Root directory containing holicity/, matterport3d/, scannet/, scannetpp/"
    )
    parser.add_argument(
        "--splits-dir", "-s", required=True,
        help="Directory containing *_mini_*.txt split files."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't actually delete, just print what would be removed."
    )
    args = parser.parse_args()

# holicity_mini_val.txt  matterpor3d_mini_test.txt  scannet_mini_val.txt    scannetpp_mini_val.txt  
    splits = ["holicity_mini_val.txt", "matterport3d_mini_test.txt",
              "scannet_mini_val.txt", "scannetpp_mini_val.txt"]
    if not splits:
        print("❌  No .txt split files found in", args.splits_dir)
        return

    for split in splits:
        process_dataset(args.data_root, args.splits_dir, split, args.dry_run)

if __name__ == "__main__":
    main()

# python scripts/remove_results.py \
#   --data-root /home/yli7/scratch/datasets/gaussian_world/outputs/ludvig \
#   --splits-dir /home/yli7/projects/yue/language_feat_exps/splits \
#   --dry-run

