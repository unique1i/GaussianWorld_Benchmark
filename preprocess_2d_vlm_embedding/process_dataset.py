import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(
        description="Batch run preprocess.py on multiple scenes."
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        required=True,
        help="Path to the text file listing scene folder names (one per line).",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the root ScanNetpp folder (e.g. '/data/work2-gcp-europe-west4-a/GSWorld/ScanNetpp').",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="scannet or scannetpp"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
    )
    parser.add_argument("--split", type=str, default="")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting line index in the txt file (inclusive).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="Ending line index in the txt file (exclusive). ",
    )
    parser.add_argument("--use_clip", action="store_true")
    args = parser.parse_args()

    with open(args.txt_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if args.end_idx == -1:
        selected_scenes = lines[args.start_idx :]
    else:
        selected_scenes = lines[args.start_idx : args.end_idx]

    print(f"Processing dataset: {args.dataset_name}")
    print(f"Found {len(lines)} total scene folders in '{args.txt_file}'.")
    print(
        f"Processing scenes from line {args.start_idx} to "
        f"{args.end_idx if args.end_idx != -1 else (len(lines) - 1)} (inclusive)."
    )
    print(f"Number of scenes to process: {len(selected_scenes)}\n")

    for idx, scene_id in enumerate(selected_scenes):
        if args.dataset_name == "scannet":
            # split ["scans", "scans_test"]
            if os.path.exists(os.path.join(args.dataset_folder, "scans", scene_id)):
                dataset_path = os.path.join(args.dataset_folder, "scans", scene_id)
            elif os.path.exists(
                os.path.join(args.dataset_folder, "scans_test", scene_id)
            ):
                dataset_path = os.path.join(args.dataset_folder, "scans_test", scene_id)
            else:
                raise FileNotFoundError(f"Scene folder not found: {scene_id}")

            cmd = [
                "python",
                "preprocess_scannet.py",
                "--dataset_path",
                dataset_path,
                "--save_folder",
                (
                    args.save_folder
                    if args.save_folder
                    else os.path.join(args.dataset_folder, "language_features")
                ),
                "--zipped",
            ]
        elif args.dataset_name == "scannetpp":
            # split ["data", "sem_test"]
            if os.path.exists(
                os.path.join(args.dataset_folder, "data", scene_id, "dslr")
            ):
                dataset_path = os.path.join(
                    args.dataset_folder, "data", scene_id, "dslr"
                )
            elif os.path.exists(
                os.path.join(args.dataset_folder, "sem_test", scene_id, "dslr")
            ):
                dataset_path = os.path.join(
                    args.dataset_folder, "sem_test", scene_id, "dslr"
                )
            else:
                raise FileNotFoundError(
                    f"Scene folder not found: {scene_id} at {args.dataset_folder}"
                )
            cmd = [
                "python",
                "preprocess_scannetpp.py",
                "--dataset_path",
                dataset_path,
                "--save_folder",
                (
                    args.save_folder
                    if args.save_folder
                    else os.path.join(args.dataset_folder, "language_features_siglip2")
                ),
                "--resolution",
                "876",  # for scannetpp, we downscale by 2
                # "--highlight", # test highlight crop
                # "--highlight_more", # test more weight on highlight
            ]
        elif args.dataset_name == "matterport3d":
            dataset_path = os.path.join(args.dataset_folder, scene_id)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Scene folder not found: {scene_id} at {args.dataset_folder}"
                )
            cmd = [
                "python",
                "preprocess_matterport3d.py",
                "--dataset_path",
                dataset_path,
                "--save_folder",
                (
                    args.save_folder
                    if args.save_folder
                    else os.path.join(args.dataset_folder, "language_features_siglip2")
                ),
            ]
        elif args.dataset_name == "holicity":
            dataset_path = os.path.join(args.dataset_folder, scene_id)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Scene folder not found: {scene_id} at {args.dataset_folder}"
                )
            cmd = [
                "python",
                "preprocess_holicity.py",
                "--dataset_path",
                dataset_path,
                "--save_folder",
                (
                    args.save_folder
                    if args.save_folder
                    else os.path.join(args.dataset_folder, "language_features_siglip2")
                ),
            ]
        else:
            raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
        if args.use_clip:
            cmd += ["--use_clip"]

        print(
            f"Processing scene {scene_id} ({idx + 1}/{len(selected_scenes)}), {dataset_path}"
        )
        print("Running command:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        print("Completed scene:", scene_id)
        print("-" * 60)


if __name__ == "__main__":
    main()
