import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
from torch.nn import functional as F


def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--feat_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument(
        "--encoder_dims",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32, 3],
    )
    parser.add_argument(
        "--decoder_dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 256, 512],
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    feat_path = args.feat_path
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{feat_path}/"
    if args.save_name is None:
        output_dir = f"{dataset_path}/language_features_clip_dim3"
    else:
        output_dir = f"{dataset_path}/{args.save_name}"

    os.makedirs(output_dir, exist_ok=True)

    # copy the segmentation map
    for filename in os.listdir(data_dir):
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            if os.path.exists(target_path):
                print("File already exists:", target_path)
                continue
            shutil.copy(source_path, target_path)
    print("Segmentation map copied to", output_dir)

    print("Loading model from", ckpt_path)
    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir)

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        drop_last=False,
    )

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    eval_loss = 0.0

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0").to(torch.float32)
        # print("Data", data.dtype, data.shape)
        with torch.no_grad():
            outputs = model.encode(data).to("cpu").numpy()
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

        with torch.no_grad():
            restore_data = model.decode(torch.tensor(outputs).cuda())
            cosine_loss = cos_loss(restore_data, data).mean()
            eval_loss += cosine_loss.item()

    print("Eval cosine loss", eval_loss / len(test_loader))

    os.makedirs(output_dir, exist_ok=True)
    start = 0

    for k, v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start : start + v])
        start += v
