# Process ScanNetpp Images to Language Features

## Clone the Repository

```bash
git clone git@github.com:RunyiYang/CLIP-SAM-process.git
```

## Download Checkpoints

1. **Option 1: Download all SAM2 checkpoints (optional):**

   ```bash
   cd sam2/checkpoints
   ./download_ckpts.sh
   cd ../..
   ```

2. **Option 2: Download only the required SAM2 checkpoint:**

   ```bash
   cd sam2/checkpoints
   wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   cd ../..
   ```

3. **Segment Anything v1 checkpoint (SAM v1)**
   ```bash
   cd segment_anything/ckpt
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   cd ../..
   ```

## Environment Setup

> **Prerequisite**: CUDA >= 12.1 (tested on CUDA 12.1, CUDA 12.4 should work too)

Create and activate a new environment (using micromamba as an example):

```bash
micromamba create -n lang_feat -y python=3.12
micromamba activate lang_feat
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx tqdm open-clip-torch IPython
pip install accelerate==1.4.0 flash_attn transformers==4.50.1

# on snellius: module load 2023; module load CUDA/12.4.0
```

## Run the Processing Script

```bash
python process_dataset_scannetpp.py \
  --txt_file runyi_splits.txt \
  --scannetpp_folder /data/work2-gcp-europe-west4-a/GSWorld/ScanNetpp/ \
  --start_idx 0 \
  --end_idx 100
```

- `--txt_file`: Path to your splits file.
- `--scannetpp_folder`: Provide the correct path to the ScanNet++ data directory.
