import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def find_npy_files(root: Path):
    """Yield all .npy files ending with _f.npy or _s.npy under root."""
    for path in root.rglob("*.npy"):
        if path.name.endswith("_f.npy") or path.name.endswith("_s.npy"):
            yield path

def main():
    parser = argparse.ArgumentParser(
        description="Count—and optionally delete—.npy files with shape (456, 616)."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to traverse (e.g. /home/.../language_features_siglip2)"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete any .npy files matching the target shape after counting"
    )
    args = parser.parse_args()

    target_shape = (456, 616)
    total_files = 0
    matching_files = []

    for npy_path in tqdm(find_npy_files(args.root), desc="Processing .npy files", unit=" file"):
        total_files += 1
        # mmap_mode='r' loads only the header and memory-maps the data
        arr = np.load(npy_path, mmap_mode='r')
        if 456 in arr.shape:
            matching_files.append(npy_path)

    print(f"Scanned {total_files} .npy files.")
    print(f"Found {len(matching_files)} files with shape {target_shape}.")

    if args.delete and matching_files:
        confirm = input(f"Are you sure you want to delete these {len(matching_files)} files? [y/N]: ")
        if confirm.lower() == "y":
            for path in matching_files:
                try:
                    os.remove(path)
                    print(f"Deleted {path}")
                except Exception as e:
                    print(f"❌ Failed to delete {path}: {e}")
        else:
            print("Deletion aborted by user.")
    elif args.delete:
        print("No matching files to delete.")

if __name__ == "__main__":
    main()