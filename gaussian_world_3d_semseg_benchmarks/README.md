### How We Run GaussianWorld Benchmark

We follow the instructions of each baseline method and have adapted their code to run on the ScanNet, ScanNet++, Matterport3D, and HoliCity benchmarks.

Each folder contains the files we used to run the zero-shot 3D semantic segmentation benchmark using the results of each method.


## Overview

The scripts for each method evaluate zero-shot semantic segmentation performance on scenes using the language features outputted by the method. The evaluation is done in following steps:    

1. Loading 3D Gaussian splatting point clouds and their language features
2. Matching scene points to nearest Gaussian features using KD-tree
3. Computing relevancy scores between features and semantic labels
4. Calculating average IoU, accuracy, and foreground mIoU/mAcc.

## Example Usage

```bash
python open_vocab_seg_ludvig_scannet.py \
    --val_split_path ./splits/scannetpp_mini_val.txt \
    --preprocessed_root /home/yli7/scratch2/datasets/ptv3_preprocessed/scannet_preprocessed \
    --gs_root /home/yli7/scratch/datasets/gaussian_world/outputs/ludvig/scannet \
    --label20_path /home/yli7/scratch2/datasets/scannet/metadata/semantic_benchmark/label20.txt \
    --model_name clip \
    --save_pred
```

### Key Arguments

- `--val_split_path`: Path to validation scene list file
- `--preprocessed_root`: Directory containing preprocessed scene data
- `--gs_root`: Root directory for Gaussian splatting outputs
- `--langfeat_root`: Directory containing language features
- `--label_path`: Path to semantic label definitions
- `--model_name`: Language model to use (`clip` or `siglip2`)
- `--save_pred`: Save prediction results as .npy files

## Input Data Structure

```
preprocessed_root/
├── val/
│   └── <scene_id>/
│       ├── coord.npy      # Point coordinates
│       └── segment.npy    # Ground truth labels

gs_root/
└── <scene_id>/
    └── <model_name>/
        └── gaussians.ply  # 3DGS file

langfeat_root/
└── <scene_id>/
    └── <model_name>/
        └── features.npy   # Outputed language features
```

## Output

- **Log File**: Saved to `logs/ludvig_3d_semseg_eval_<split>_<model>.txt`
- **Predictions** (optional): Saved as `.npy` files when `--save_pred` is used

## Relevancy Scoring

**Dot Similarity**:
   ```
   Score_c = max(lang·text_c)
   ```

## Metrics

- **Global Accuracy**: Overall point-wise accuracy
- **Mean IoU**: Average Intersection over Union across all classes
- **Mean Class Accuracy**: Average per-class accuracy
- **Foreground mIoU/mAcc**: Metrics excluding wall/floor/ceiling classes