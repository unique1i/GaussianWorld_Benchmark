"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch
import torch.utils.data

# from scipy.spatial import cKDTree
import argparse

# from pointcept.utils.misc import (
#     AverageMeter,
#     intersection_and_union,
#     intersection_and_union_gpu,
#     make_dirs,
#     neighbor_voting,
#     clustering_voting
# )

import open_clip

from autoencoder.model import Autoencoder

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def get_argparse():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--feat_path",
        type=str,
        default=".",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        default=".",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--save_feat_path",
        type=str,
        default=None,
        help="Path to the output directory",
    )
    return parser.parse_args()


# 09c1414f1b
# 0d2ee665be
# 38d58a7a31
# 3db0a1c8f3
# 5ee7c22ba0
# 5f99900f09
# a8bf42d646
# a980334473
# c5439f4607
# cc5237fd77

if __name__ == "__main__":
    args = get_argparse()
    label_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt"
    # feat_path = f'/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/{data_name}/dslr/language_features_multi_clip_dim3/'
    # save_feat_path = f'/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannetpp_mini_val_set_suite/original_data/{data_name}/dslr/language_features_multi_clip_dim512_restore/'

    feat_path = args.feat_path
    assert os.path.exists(feat_path), f"Feature path not found at {feat_path}"

    ae_checkpoint = args.ae_checkpoint
    save_feat_path = args.save_feat_path
    if save_feat_path is None:
        feat_name = feat_path.split("/")[-1]
        feat_root_path = "/".join(feat_path.split("/")[:-1])
        save_feat_path = os.path.join(feat_root_path, f"{feat_name}_dim512_restore")
        print(f"save_feat_path: {save_feat_path}")
    os.makedirs(save_feat_path, exist_ok=True)

    # ae_checkpoint = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/autoencoder/ckpt', data_name+'_clip', 'best_ckpt.pth')

    # ------------------------------
    # data_name = '0d2ee665be'
    import glob
    import shutil

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------------------------------
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # ---- 3.1 Load label names & encode text ----
    with open(label_path, "r") as f:
        label_names = [line.strip() for line in f if len(line.strip()) > 0]
    prompt_list = ["this is a " + name for name in label_names]

    with torch.no_grad():
        text_tokens = tokenizer(label_names)
        text_feat = model.encode_text(text_tokens.to(device))  # (C, 512)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        text_embeds_np = text_feat.cpu().numpy()  # (C, 512)

    # val_list = ['0d2ee665be']
    # for data_name in tqdm(val_list):
    #     # os.listdir('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/subset_sequences/')
    #     # feat_path = f'/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/subset_sequences//{data_name}/dslr/language_features_dim3/'

    # if not os.path.exists(feat_path):
    #     print(f"feat_path not exists: {feat_path}")
    #     continue

    assert os.path.exists(
        ae_checkpoint
    ), f"Autoencoder checkpoint not found at {ae_checkpoint}"

    # autoencder back to 512 dimension
    encoder_hidden_dims = [256, 128, 64, 32, 3]
    decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]

    ae_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    ae_checkpoint_load = torch.load(ae_checkpoint)
    ae_model.load_state_dict(ae_checkpoint_load)
    ae_model.eval()

    seg_files = sorted(glob.glob(os.path.join(feat_path, "*_s.npy")))
    for seg_path in seg_files:
        # Derive the matching feature path
        base_name = os.path.basename(seg_path).replace("_s.npy", "")
        feat_path_i = os.path.join(feat_path, base_name + "_f.npy")
        # Load segmentation and features
        shutil.copyfile(seg_path, os.path.join(save_feat_path, base_name + "_s.npy"))
        feat_load = np.load(feat_path_i)  #  N, 3
        print("feat_load.shape", feat_load.shape)

        with torch.no_grad():
            feat_load = torch.from_numpy(feat_load).to("cuda:0")  # H, W, 3
            restored_feat_load = ae_model.decode(feat_load)  # H*W, 512
        restored_feat_load = restored_feat_load.cpu().numpy()
        # restored_feat_load = restored_feat_load / np.linalg.norm(restored_feat_load, axis=1, keepdims=True)  # Normalize features
        print("restored_feat_load.shape", restored_feat_load.shape)
        np.save(os.path.join(save_feat_path, base_name + "_f.npy"), restored_feat_load)
