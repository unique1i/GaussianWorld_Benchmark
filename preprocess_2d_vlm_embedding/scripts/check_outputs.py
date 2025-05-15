import os
import json
from tqdm import tqdm
import argparse

############ Configuration ############
dataset_name = "scannetpp"
splits_dict = {"scannetpp": ['data', 'sem_test'], "scannet": ['scans', 'scans_test']}
splits = splits_dict[dataset_name]
dataset_folder = '/home/yli7/scratch2/datasets/scannet' if dataset_name == "scannet" else '/home/yli7/scratch2/datasets/scannetpp_v2'
dataset_lang_feat = f'/home/yli7/scratch2/datasets/scannet/language_features_siglip2' if dataset_name == "scannet" \
    else f'/home/yli7/scratch2/datasets/scannetpp_v2/language_features_siglip2'

def get_selected_image_paths(scene_path, dataset_name):
    data_list = []
    if dataset_name == "scannetpp":
        json_path = os.path.join(scene_path, "dslr", "nerfstudio", "lang_feat_selected_imgs.json")
        undistorted_images_path = os.path.join(scene_path, "dslr", "undistorted_images")
    elif dataset_name == "scannet":
        json_path = os.path.join(scene_path, "lang_feat_selected_imgs.json")
        undistorted_images_path = os.path.join(scene_path, "color_interval.zip")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Json file not found: {json_path}")
    # Print the modification time of the JSON file
    print(f"Last modified of {json_path}: {os.path.getmtime(json_path)}")

    with open(json_path, "r") as f:
        json_data = json.load(f)

    frames_list = json_data.get("frames_list", [])
    
    for img_name in frames_list:
        img_path = os.path.join(undistorted_images_path, img_name)
        if os.path.exists(img_path):
            data_list.append(img_name)
        elif dataset_name == "scannet":
            data_list.append(img_name) # scannet has zipped images
        else:
            print(f"Warning: Image {img_path} does not exist. Skipping this image.")

    return sorted(data_list)

def main(delete=False, dry_run=True, dataset_name="scannetpp"):
    total_missing_images = {}
    total_extra_images = 0
    for split in splits:
        split_path = os.path.join(dataset_folder, split)
        scene_folders = os.listdir(split_path)
        for scene_folder in tqdm(scene_folders, desc=f"Processing scenes in {split}"):
            scene_path = os.path.join(split_path, scene_folder)
            if not os.path.isdir(scene_path):
                raise NotADirectoryError(f"Path is not a directory: {scene_path}")
            print(f"\nChecking outputs for {scene_folder} in split {split}...")
            
            selected_image_names = get_selected_image_paths(scene_path, dataset_name)
            # Remove extension (e.g., '.jpg') from each image name
            selected_image_names = [f[:-4] for f in selected_image_names]
            print(f"Selected image number: {len(selected_image_names)}")

            lang_feat_folder = os.path.join(dataset_lang_feat, scene_folder)
            if not os.path.exists(lang_feat_folder):
                raise FileNotFoundError(f"lang_feat_folder not found: {lang_feat_folder}")
            lang_feat_files = sorted(os.listdir(lang_feat_folder))

            # Assume each feature file name ends with 6 extra characters (e.g., '_feat' + extension)
            lang_feat_files_imgs = list(set([f[:-6] for f in lang_feat_files])) # '_s.npy'
            if "lang_feat_selected_img" in lang_feat_files_imgs:
                lang_feat_files_imgs.remove("lang_feat_selected_img")  # Remove the JSON file
            
            # Compare selected image names and lang_feat file base names
            missing_images = set(selected_image_names) - set(lang_feat_files_imgs)
            extra_images = set(lang_feat_files_imgs) - set(selected_image_names)
            total_extra_images += len(extra_images)
            print("Missing images: ", len(missing_images))
            print("Extra images: ", len(extra_images))

            # Delete extra feature files if requested
            if delete and extra_images:
                # Gather the files to delete by matching the base name
                files_to_delete = []
                for f in lang_feat_files:
                    base_name = f[:-6]
                    if base_name in extra_images:
                        files_to_delete.append(os.path.join(lang_feat_folder, f))
                
                if dry_run:
                    print(f"Dry-run: Would delete {len(files_to_delete)} extra file(s) in {scene_folder}:")
                    for file_path in files_to_delete:
                        print(" ", file_path)
                else:
                    for file_path in files_to_delete:
                        try:
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")
                        except Exception as e:
                            print(f"Error deleting {file_path}: {e}")
                    print(f"Deleted {len(files_to_delete)} extra file(s) in {scene_folder}.")
            
            total_missing_images[scene_folder] = len(missing_images)
    print("\nTotal scene number: ", len(total_missing_images))
    # Print per-scene missing features
    print("\nScene-wise missing features report:")
    any_missing = False
    for scene in sorted(total_missing_images.keys()):
        count = total_missing_images[scene]
        if count > 0:
            print(f"{scene}: {count} missing features")
            any_missing = True
    if not any_missing:
        print("All scenes have no missing features.")
    print("Total missing features for selected images: ", sum(total_missing_images.values()))
    print("Total extra images: ", total_extra_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check and optionally delete extra language feature files."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete extra feature files that do not correspond to selected images."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run deletion: list files to be deleted without actually deleting them."
    )
    args = parser.parse_args()

    if args.delete:
        print("Extra files deletion is ENABLED.")
        if args.dry_run:
            print("Dry-run mode: Files will not be actually deleted.")
    else:
        print("Deletion not enabled. Only checking differences.")

    main(delete=args.delete, dry_run=args.dry_run, dataset_name=dataset_name)