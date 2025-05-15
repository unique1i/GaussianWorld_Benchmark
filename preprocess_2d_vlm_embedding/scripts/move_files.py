import os
import glob
import shutil


def move_npy_files(base_dir=".", dry_run=True):
    """
    Iterate over scene folders inside `data/` folder, find any files matching
    '*_s.npy' or '*_f.npy' in 'dslr/undistorted_images/', and move them into
    scene-specific subdirectories under `language_features/`.

    :param base_dir: The base directory containing 'data' and 'language_features'.
    :param dry_run: If True, only print the intended moves without executing them.
    """
    data_dir = os.path.join(base_dir, "data")
    lang_feat_dir = os.path.join(base_dir, "language_features")

    # List all directories under `data`
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    scene_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]

    for scene in scene_dirs:
        scene_img_dir = os.path.join(data_dir, scene, "dslr", "undistorted_images")
        if not os.path.exists(scene_img_dir):
            # If there's no `dslr/undistorted_images` folder, skip
            continue

        # Look for *_s.npy and *_f.npy in that directory
        s_npy_files = glob.glob(os.path.join(scene_img_dir, "*_s.npy"))
        f_npy_files = glob.glob(os.path.join(scene_img_dir, "*_f.npy"))
        npy_files = s_npy_files + f_npy_files

        if not npy_files:
            # No matching files, skip
            continue

        # Prepare the destination scene folder under language_features
        dest_scene_dir = os.path.join(lang_feat_dir, scene)
        if dry_run:
            print(f"[dry_run] Would create folder (if not exists): {dest_scene_dir}")
        else:
            os.makedirs(dest_scene_dir, exist_ok=True)

        # Move the files
        for npy_file in npy_files:
            file_name = os.path.basename(npy_file)
            dest_path = os.path.join(dest_scene_dir, file_name)

            if dry_run:
                print(f"[dry_run] Would move {npy_file} --> {dest_path}")
            else:
                shutil.move(npy_file, dest_path)
                print(f"Moved {npy_file} --> {dest_path}")


def copy_json_files(base_dir=".", dry_run=True):
    """
    Iterate over scene folders in `data/` and look for 
    'dslr/nerfstudio/lang_feat_selected_imgs.json'.
    Copy the JSON file (if found) to scene-specific subdirectories under `language_features/`.

    """
    data_dir = os.path.join(base_dir, "data") # 'data' and 'sem_test'
    lang_feat_dir = os.path.join(base_dir, "language_features_siglip2_w_highlight_mainly_crop_w_bg")

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    scene_dirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]

    for scene in scene_dirs:
        nerfstudio_dir = os.path.join(data_dir, scene, "dslr", "nerfstudio")
        json_file = os.path.join(nerfstudio_dir, "lang_feat_selected_imgs.json")

        if os.path.exists(json_file):
            dest_scene_dir = os.path.join(lang_feat_dir, scene)
            dest_json_file = os.path.join(dest_scene_dir, "lang_feat_selected_imgs.json")

            if dry_run:
                print(f"[dry_run] Would create folder (if not exists): {dest_scene_dir}")
                print(f"[dry_run] Would copy {json_file} --> {dest_json_file}")
            else:
                os.makedirs(dest_scene_dir, exist_ok=True)
                shutil.copy2(json_file, dest_json_file)
                print(f"Copied {json_file} --> {dest_json_file}")

if __name__ == "__main__":
    # Set dry_run=True to test the script without moving files.

    # move_npy_files(base_dir="/home/yli7/scratch2/datasets/scannetpp", dry_run=False)
    copy_json_files(base_dir="/home/yli7/scratch2/datasets/scannetpp_v2", dry_run=False)
