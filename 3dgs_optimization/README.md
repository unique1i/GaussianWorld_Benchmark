# gsplat

3DGS optimization based on Gsplat 1.3.0 version.

### Environment Setup

```bash
conda env create -n gsplat python=3.10
conda activate gsplat
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```

Note, we made changes to gsplat, please install the gsplat from the source code in this repo.

#### What changes have we made?

- support custom dataset, e.g., we add `ARKitScenesDataset` and `ScannetppDataset` which support both unzipped and zipped dataset loading (to avoid the large number of files). Please zip the image/depth folder inside each scene folder. The single function `initialize_datasets` takes care of all the dataloading.
- add "bbox_min" and "bbox_max" keys in the per-scene `transforms_train.json` file for the bounding box. It is recommended to add the bbox of the initializing point cloud when processing the dataset, format: [minx, miny, minz], [maxx, maxy, maxz]. If provided, when pruning the Gaussians and saving the ply files, we will only keep the Gaussians within the bounding box + 1m.
- use `err_mask` to filter depth error, to exclude large error pixels, if depth loss is enabled in the training
- add `eval_depth` in the evaluation step
- add `disable_growing` and `disable_pruning` strategy for default 3DGS strategy

#### What are the recommended training settings?

In short, we use the settings:

```python
MCMCStrategy
cap_max: int = 1500_000 # 1.5M Gaussians
depth_loss = True
scale_reg = 0.02
opacity_reg = 0

# the arguments for the above
# we provide these in batch_train.sh for convenience
python examples/simple_trainer.py "mcmc" "--strategy.cap-max" "1500000" "--depth_loss" "--scale_reg" "0.02" "--opacity_reg" "0"
```

Here is an training ablation on a Scannet++ scnene with defualt strategy.

Highest psnr is achieved with only rgb_loss, and scale_reg=0.02, opacity_reg=0, however, using depth_loss is recommended as it improves the rendered depth a lot, thus align the Gaussians better with the scene geometry. This leads to a psnr drop of about 0.5, which is acceptable.

MCMC startegy trains faster and has slightly higher PSNR, we can set a fixed target number of Gaussians for MCMC, e.g., scannet is trained to fit 1.5M Gaussians using MCMC.

<details>
<summary>Ablation Table</summary>

| Name             | depth_loss | scale_reg | opacity_reg | disable_growing | disable_pruning | strategy.reset | num_GS  | psnr     | depth_loss | State    | Notes                 |
| ---------------- | ---------- | --------- | ----------- | --------------- | --------------- | -------------- | ------- | -------- | ---------- | -------- | --------------------- |
| 281bc17764_9880b | false      | 0.02      | 0           | false           | false           | 30000          | 1891340 | 34.42562 | 0.073324   | Finished | Add notes...          |
| 281bc17764_a7651 | false      | 0         | 0           | false           | false           | 30000          | 1815413 | 34.37514 | 0.073707   | Finished | disable opacity reset |
| 281bc17764_2a72d | false      | 0.02      | 0           | false           | false           | 3000           | 1411055 | 33.99352 | 0.073012   | Finished | Add notes...          |
| 281bc17764_7cca4 | false      | 0         | 0           | false           | false           | 3000           | 1371814 | 33.9646  | 0.073835   | Finished | Add notes...          |
| 281bc17764_63f0f | true       | 0.02      | 0           | false           | false           | 30000          | 1950583 | 33.89177 | 0.010965   | Finished | Add notes...          |
| 281bc17764_99f83 | true       | 0         | 0           | false           | false           | 3000           | 1769958 | 33.65436 | 0.011429   | Finished | Add notes...          |
| 281bc17764_7f5c5 | true       | 0         | 0           | false           | false           | 3000           | 1791170 | 33.64742 | 0.011331   | Finished | Add notes...          |
| 281bc17764_0851d | false      | 0         | 0           | true            | true            | 30000          | 1358855 | 33.7135  | 0.076832   | Finished | Add notes...          |
| 281bc17764_b2deb | false      | 0         | 0           | true            | true            | 30000          | 1339764 | 33.29958 | 0.075381   | Finished | Add notes...          |
| 281bc17764_71d78 | true       | 0         | 0           | true            | true            | 30000          | 1338232 | 33.2579  | 0.081101   | Finished | Add notes...          |
| 281bc17764_46993 | true       | 0         | 0           | true            | true            | 30000          | 1329827 | 32.92927 | 0.081321   | Finished | Add notes...          |

</details>

#### How to add a new dataset?

- We require processing scenes first and obtain the per-scene metadata saved in `transforms_train.json` in each scene folder. Each json should be in the format of the following with example in `scripts/transforms_train.json`. An example script used to process the ARKitScenes dataset is provided in `scripts/arkit_process_per-scene_json.py`.

  ```python
  {
      "share_intrinsics": False,     # usually True, if not, please add per-frame intrinsics in frame
      "fx": 0,                       # placeholder if share_intrinsics is False
      "fy": 0,
      "cx": 0,
      "cy": 0,
      "width": intrinsics["width"],  # original image size
      "height": intrinsics["height"],
      "zipped": False,      # if zipped, we will load later from zip files
      "crop_edge": 0,       # crop edge of the image if needed
      "resize": [960, 720], # target size of the image, if image is too large
      "frames_num": len(frames),
      "init_point_num": init_point_num, # point clouds size used for 3DGS initialization
      "bbox_min": bbox_min, # bounding box min of the point clouds
      "bbox_max": bbox_max, # bounding box max of the point clouds
      "frames": frames,     # see below
      "test_frames": test_frames # if not provided by dataset, random select 50 from frames
  }

  # frame format: a list of dictionaries, each dict has the format:
  [
      {
          "file_path": relative_path, # image path relative to the scene folder
          "transform_matrix": transform_matrix, # list of 4 x 4, use array.tolist(), camera to world pose
          "fx": intrinsics["fx"], # needed if share_intrinsics is False
          "fy": intrinsics["fy"],
          "cx": intrinsics["cx"],
          "cy": intrinsics["cy"],
      },
      ...... # more frames
  ]
  ```

- Please add new dataset class resemble the exsiting ones, e.g., `ARKitScenesDataset`.
- Note all the dataset class should have the same `__getitem__` format, i.e.:
  ```python
  data = {
          "image": img_tensor,  # 3 x H x W, range [0, 255]
          "camtoworld": torch.from_numpy(pose),  # 4 x 4, camera to world pose
          "K": K,  # 3 x 3 tensor
          "image_id": idx,
          "image_name": image_rel_path, # image path relative to the scene folder
          }
  if self.load_depth:
      data["depth"] = depth_tensor  # H x W
  return data
  ```
- Link the dataset class in `examples/datasets/dataset_factory.py`, add per dataset loading process in `initialize_datasets` function.
- If we use mesh for 3DGS initialization, and the dataset has depth input for training, the script `scripts/fuse_rgbd_pcl.py` helps obtain the fused RGBD point clouds from the depth, which can be used for inspection if the point clouds align with the mesh. The script supports our datset class format.

#### How to submit 3DGS training jobs of a whole dataset?

The script `batch_train.sh` is added to help submit training jobs for a whole dataset. It requires the dataset class to be added first.

The following syntax is used to submit the training jobs:

```bash
usage() {
    echo "Usage: $0 -i <input_root_folder> -o <output_root_folder> -g <gpu_ids> [-s <scene_list_file>] [-h]"
    echo ""
    echo "Options:"
    echo "  -i, --input       Path to the input root folder containing all the scene folders."
    echo "  -o, --output      Path to the output root folder where results will be saved."
    echo "  -g, --gpus        Comma-separated list of GPU IDs to use (e.g., 0,1,2,3)."
    echo "  -s, --scene-list  (Optional) Path to a text file listing scene folder names to train, one folder name per line."
    echo "  -h, --help        Display this help message and exit."
    exit 1
}

# example command used to train the arkitscenes dataset
# bash bash_train.sh -i /nvmestore/yli7/datasets/ARKitScenes/wide -o /nvmestore/yli7/outputs/arkit_gs -g 0,1 -s /nvmestore/yli7/datasets/ARKitScenes/scripts_OpenSun3D/valid_scenes.txt
```

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

```bash
pip install -r requirements.txt
```

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
# under examples/
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)
