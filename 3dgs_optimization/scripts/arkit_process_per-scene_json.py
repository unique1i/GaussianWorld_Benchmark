# """
# This script is used to write the json file of the "wide" assets from the dataset for 3DGS training.
# "wide" images and depths are downloaded using the script from scripts_OpenSun3D.
# """
import os
import json
import glob
import bisect
import numpy as np
import cv2
from plyfile import PlyData
from tqdm import tqdm


def convert_angle_axis_to_matrix3(angle_axis):
    """
    Convert angle-axis to rotation matrix using OpenCV.
    """
    angle_axis = np.array(angle_axis, dtype=np.float64).reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(angle_axis)
    return rotation_matrix


def TrajStringToMatrix(traj_str):
    """
    Convert a trajectory string to timestamp and inverse extrinsic matrix.
    """
    tokens = traj_str.strip().split()
    assert len(tokens) == 7, f"Trajectory string does not have 7 tokens: {traj_str}"

    timestamp = tokens[0]
    angle_axis = [float(tok) for tok in tokens[1:4]]
    translation = np.array([float(tok) for tok in tokens[4:7]], dtype=np.float64)

    rotation_matrix = convert_angle_axis_to_matrix3(angle_axis)

    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = translation

    # Invert to get camera-to-world
    Rt = np.linalg.inv(extrinsics)

    return timestamp, Rt.tolist()


def find_closest_pose_from_timestamp(
    image_timestamps, pose_timestamps, time_dist_limit=0.3
):
    """
    For each image timestamp, find the closest pose timestamp within the time distance limit.
    """
    closest_poses = []
    new_frame_ids = []

    # Convert pose timestamps to float and sort
    pose_timestamps_sorted = sorted(pose_timestamps, key=lambda x: float(x))
    pose_timestamps_floats = [float(ts) for ts in pose_timestamps_sorted]

    for image_ts in image_timestamps:
        image_ts_float = float(image_ts)
        index = bisect.bisect_left(pose_timestamps_floats, image_ts_float)

        if index == 0:
            closest_pose = pose_timestamps_sorted[0]
        elif index == len(pose_timestamps_sorted):
            closest_pose = pose_timestamps_sorted[-1]
        else:
            prev_pose = pose_timestamps_sorted[index - 1]
            next_pose = pose_timestamps_sorted[index]
            diff_prev = abs(image_ts_float - float(prev_pose))
            diff_next = abs(float(next_pose) - image_ts_float)
            if diff_prev < diff_next:
                closest_pose = prev_pose
            else:
                closest_pose = next_pose

        if abs(float(closest_pose) - image_ts_float) < time_dist_limit:
            closest_poses.append(closest_pose)
            new_frame_ids.append(image_ts)
        else:
            print(
                f"Warning: Closest pose for image timestamp {image_ts} is {closest_pose}, which exceeds the time distance limit. Skipping."
            )

    return new_frame_ids, closest_poses


def st2_camera_intrinsics(filename):
    """
    Read camera intrinsics from a .pincam file.
    """
    w, h, fx, fy, cx, cy = np.loadtxt(filename)
    return {
        "width": int(w),
        "height": int(h),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


def read_ply(ply_file):
    """
    Read a PLY file and extract the number of points and bounding box.
    """
    plydata = PlyData.read(ply_file)
    vertices = plydata["vertex"]
    num_points = len(vertices)

    x = vertices["x"]
    y = vertices["y"]
    z = vertices["z"]

    bbox_min = [float(np.min(x)), float(np.min(y)), float(np.min(z))]
    bbox_max = [float(np.max(x)), float(np.max(y)), float(np.max(z))]

    return num_points, bbox_min, bbox_max


def process_scene(scene_path, frames_min=500):
    """
    Process a single scene folder to generate transforms_train.json.
    """
    # Paths
    traj_file = os.path.join(scene_path, "lowres_wide.traj")
    wide_folder = os.path.join(scene_path, "wide")
    intrinsics_folder = os.path.join(scene_path, "wide_intrinsics")
    mesh_file = None
    # Find the .ply mesh file
    for file in os.listdir(scene_path):
        if file.endswith(".ply"):
            mesh_file = os.path.join(scene_path, file)
            break
    if mesh_file is None:
        print(f"No .ply file found in {scene_path}. Skipping.")
        return False

    if not os.path.exists(traj_file):
        print(f"Trajectory file {traj_file} does not exist. Skipping.")
        return False

    # Read and parse trajectory
    poses_from_traj = {}
    with open(traj_file, "r") as f:
        traj_lines = f.readlines()
    for line in traj_lines:
        timestamp, Rt = TrajStringToMatrix(line)
        poses_from_traj[timestamp] = Rt

    # Collect frame IDs from wide folder
    image_paths = sorted(glob.glob(os.path.join(wide_folder, "*.png")))
    frame_ids = [
        os.path.basename(p).split(".png")[0].split("_")[1] for p in image_paths
    ]  # e.g., "41048190_3419.682"
    frame_ids = sorted(frame_ids, key=lambda x: float(x))

    # checck total frames num
    if len(frame_ids) < frames_min:
        print(f"Less than {frames_min} frames in {scene_path}. Skipping.")
        return False

    # Find closest poses
    pose_timestamps = list(poses_from_traj.keys())
    new_frame_ids, closest_pose_ids = find_closest_pose_from_timestamp(
        frame_ids, pose_timestamps, time_dist_limit=0.3
    )

    if len(new_frame_ids) == 0:
        print(f"No matching poses found for frames in {scene_path}. Skipping.")
        return False

    # Prepare frames with individual intrinsics
    frames = []
    for idx, (frame_id, pose_id) in enumerate(zip(new_frame_ids, closest_pose_ids)):
        # Define relative file path, e.g., "wide/41048190_3419.682.png"
        image_filename = f"{os.path.basename(scene_path)}_{frame_id}.png"
        relative_path = os.path.join("wide", image_filename)

        # Get the transform matrix
        transform_matrix = poses_from_traj[pose_id]

        # Locate the corresponding .pincam file
        pincam_file = os.path.join(
            intrinsics_folder, f"{os.path.basename(scene_path)}_{frame_id}.pincam"
        )
        if not os.path.exists(pincam_file):
            # Try with +/- 0.001 adjustment
            adjusted_id_minus = f"{float(frame_id) - 0.001:.3f}"
            pincam_file = os.path.join(
                intrinsics_folder,
                f"{os.path.basename(scene_path)}_{adjusted_id_minus}.pincam",
            )
            if not os.path.exists(pincam_file):
                adjusted_id_plus = f"{float(frame_id) + 0.001:.3f}"
                pincam_file = os.path.join(
                    intrinsics_folder,
                    f"{os.path.basename(scene_path)}_{adjusted_id_plus}.pincam",
                )
                if not os.path.exists(pincam_file):
                    print(
                        f"Intrinsics file for frame {frame_id} not found in {scene_path}. Skipping frame."
                    )
                    continue  # Skip this frame if intrinsics are not found

        # Read intrinsics for the current frame
        intrinsics = st2_camera_intrinsics(pincam_file)
        frame = {
            "file_path": relative_path,
            "transform_matrix": transform_matrix,
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
        }
        frames.append(frame)

    if not frames:
        print(f"No frames with valid intrinsics found in {scene_path}. Skipping.")
        return

    # Read mesh information
    init_point_num, bbox_min, bbox_max = read_ply(mesh_file)

    # randomly select 50 frames as test frames
    test_frames = np.random.choice(frames, 50, replace=False)

    # json format
    transforms = {
        "share_intrinsics": False,
        "fx": 0,  # Placeholder since intrinsics are per-frame
        "fy": 0,
        "cx": 0,
        "cy": 0,
        "width": intrinsics["width"],  # Assuming width and height are consistent
        "height": intrinsics["height"],
        "zipped": False,
        "crop_edge": 0,
        "resize": [960, 720],
        "frames_num": len(frames),
        "init_point_num": init_point_num,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "frames": frames,
        "test_frames": test_frames,
    }

    output_file = os.path.join(scene_path, "transforms_train.json")
    with open(output_file, "w") as f:
        json.dump(transforms, f, indent=4)
    print(f"Written {output_file}")

    return True


def main():
    """
    Main function to process all scene folders in the dataset path.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate transforms_train.json for each scene folder."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/yli7/scratch/repos/dataset_tools/ARKitScenes/input",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path

    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return

    # List all scene directories (assuming each subdirectory is a scene)
    scene_dirs = [
        os.path.join(dataset_path, d)
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if not scene_dirs:
        print(f"No scene directories found in {dataset_path}.")
        return
    else:
        print(f"Found {len(scene_dirs)} scene folders.")

    valid_scene_num = 0
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes..."):
        _ = process_scene(scene_dir)
        if _:
            valid_scene_num += 1
    print(f"Processed {valid_scene_num}/{len(scene_dirs)} valid scenes.")


if __name__ == "__main__":
    main()
