import os
import numpy as np

# validation_dir =
# validation_set = np.loadtxt(validation_dir, dtype=str)

import time


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 3DGS Gradient Backprojection Benchmark"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="/insait/qimaqi/workspace/3dgs-gradient-backprojection-benchmark/splits/scannet_mini_val.txt",
        help="Split name",
    )
    parser.add_argument("--rescale", type=int, default=0, help="rescale custom")
    parser.add_argument(
        "--result_root", type=str, default="./results", help="result root"
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="/scannet_mini_val_set_suite/original_data",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--ply_root_path",
        type=str,
        default="/scannet_mini_val_set_suite/mcmc_3dgs",
        help="Path to the ply files",
    )
    parser.add_argument(
        "--results_root_dir",
        type=str,
        default="./results/scannet/",
        help="Path to the results directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    split = args.split
    rescale = args.rescale
    result_root = args.result_root

    validation_set = np.loadtxt(args.split, dtype=str)
    validation_set = sorted(validation_set)
    if args.end_idx == -1:
        args.end_idx = len(validation_set)
    validation_set = validation_set[args.start_idx : args.end_idx]

    start_time = time.time()
    for scene in validation_set:
        scene = scene.split("/")[-1]
        if args.rescale == 0:
            potential_results_dir = os.path.join(
                result_root, scene, "features_lseg_480_640.pt"
            )
        elif args.rescale == 1:
            potential_results_dir = os.path.join(
                result_root, scene, "features_lseg_320_480.pt"
            )

        if os.path.exists(potential_results_dir):
            print("Found potential results dir: ", potential_results_dir)
            continue
        print(scene)
        # os.system("python backproject_ply_scannet.py --scene_name {} --rescale {}".format(scene, rescale))
        os.system(
            "python backproject_ply_scannet.py --data_root_path {} --ply_root_path {} --results_root_dir {} \
            --scene_name {} --rescale {}".format(
                args.data_root_path,
                args.ply_root_path,
                args.results_root_dir,
                scene,
                rescale,
            )
        )

    end_time = time.time()
    print("Total time: ", end_time - start_time)
