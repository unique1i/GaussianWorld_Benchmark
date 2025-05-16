"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
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
from dataclasses import dataclass, field
from typing import Tuple, Type

import torchvision
import open_clip
import torch.nn as nn

from autoencoder.model import Autoencoder

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.negatives]
            ).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(
            self.config.clip_model_type, self.config.clip_model_pretrained
        )

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
            ).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(
            softmax,
            1,
            best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2),
        )[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

    def encode_scannetpp_text(self, text_path, save_path):
        with open(text_path, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        tok_phrases = torch.cat(
            [self.tokenizer(phrase) for phrase in self.class_names]
        ).to("cuda")
        text_embedding = self.model.encode_text(tok_phrases)
        text_embedding = text_embedding.float()
        # save text embedding
        text_embedding = text_embedding.cpu()
        print("text_embedding", text_embedding.shape)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        torch.save(text_embedding, save_path)


def get_argparse():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--gs_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default=None,
        help="Path to the autoencoder checkpoint",
    )

    args = parser.parse_args()
    return args


# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/09c1414f1b/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/0d2ee665be/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/38d58a7a31/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/3db0a1c8f3/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/5ee7c22ba0/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/5f99900f09/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/a8bf42d646/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/a980334473/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/c5439f4607/feature_level_3/chkpnt30000.pth
# python langsplat_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/cc5237fd77/feature_level_3/chkpnt30000.pth

# python langsplat_3D_feat_to_eval_feat.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/holicity/ytwUEEljP6RgoV0MviqvsQ_LD/feature_level_-1/chkpnt30000.pth

# /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/matterport/q9vSo1VnCiC_05/feature_level_-1/chkpnt30000.pth


if __name__ == "__main__":
    args = get_argparse()
    # model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    model_params, first_iter = torch.load(args.gs_path)
    (
        active_sh_degree,
        _xyz,
        _features_dc,
        _features_rest,
        _scaling,
        _rotation,
        _opacity,
        _language_feature,
        max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        spatial_lr_scale,
    ) = model_params

    data_name = os.path.basename(os.path.dirname(os.path.dirname(args.gs_path)))
    save_path = os.path.dirname(args.gs_path)

    if args.autoencoder_path is None:
        ae_checkpoint = os.path.join("autoencoder/ckpt", data_name, "best_ckpt.pth")
    else:
        ae_checkpoint = args.autoencoder_path

    assert os.path.exists(
        ae_checkpoint
    ), f"Autoencoder checkpoint not found at {ae_checkpoint}"

    os.makedirs(save_path, exist_ok=True)
    pred_save_path = os.path.join(save_path, "features.npy")
    # feat_save_path = os.path.join(save_path, "feat", f"{data_name}_feat.pth")

    print("_language_feature", _language_feature.shape)

    pred_part_feat = _language_feature.cuda()
    # find pred_part_feat with 0,0,0
    # autoencder back to 512 dimension
    encoder_hidden_dims = [256, 128, 64, 32, 3]
    decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]

    ae_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    ae_checkpoint_load = torch.load(ae_checkpoint)
    ae_model.load_state_dict(ae_checkpoint_load)
    ae_model.eval()

    with torch.no_grad():
        # lvl, h, w, _ = sem_feat.shape
        restored_feat = ae_model.decode(pred_part_feat)
        # restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512

    # pred_part_feat_decode = ae_model.encode(pred_part_feat).to("cpu").numpy()
    print("restored_feat", restored_feat[:3])
    # renormalize
    restored_feat = F.normalize(restored_feat, p=2, dim=-1)
    # pred_part_feat = out_dict["point_feat"]["feat"]  # shape [M, feat_dim]
    # max_probs, argmax_indices = torch.max(logits, dim=1)
    # argmax_indices[max_probs < confidence_threshold] = ignore_index
    # pred = argmax_indices.cpu().numpy()
    # norms = restored_feat.norm(dim=-1, keepdim=True)

    restored_feat = restored_feat.cpu().numpy()
    print("restored_feat", restored_feat.shape)
    np.save(pred_save_path, restored_feat)

    _xyz_np = _xyz.cpu().numpy()
    xyz_save_path = os.path.join(save_path, "xyz.npy")
    np.save(xyz_save_path, _xyz_np)
    print("_xyz_np", _xyz_np.shape)
    # restored_feat_mask = restored_feat[pred_part_feat_mask]
    # gt_mask = semantic_gt[pred_part_feat_mask]
    # logits = torch.mm(restored_feat_mask, text_embeddings.t()).softmax(dim=-1)

    # # pred = logits.topk(3, dim=1)[1].cpu().numpy()  # shape => [N, 3]
    # pred = logits.topk(1, dim=1)[1].cpu().numpy()  # shape => [N, 3]

    # coord = coord[pred_part_feat_mask]
    # pred = neighbor_voting(coord, pred, vote_k, ignore_index, num_classes, valid_mask=None)

    # print("after voting pred", pred.shape)
    # print("gt_mask", gt_mask.shape)
    # gt_mask = gt_mask.cpu().numpy()

    # intersection, union, target = intersection_and_union(
    #     pred, gt_mask, num_classes, ignore_index
    # )
    # record = {}
    # record[data_name] = dict(
    #     intersection=intersection,
    #     union=union,
    #     target=target
    # )

    # # Per‐scene IoU & accuracy
    # mask = union != 0
    # iou_class = intersection / (union + 1e-10)
    # iou = np.mean(iou_class[mask])
    # acc = sum(intersection) / (sum(target) + 1e-10)

    # print("iou_class", iou_class)
    # print("acc", acc)
    # print("iou", iou)

    # acc 0.10824689671320656
    # iou 0.012283376296118304

    # # Running average across scenes so far
    # mask_union = union_meter.sum != 0
    # mask_target = target_meter.sum != 0
    # m_iou = np.mean((intersection_meter.sum / (union_meter.sum + 1e-10))[mask_union])
    # m_acc = np.mean((intersection_meter.sum / (target_meter.sum + 1e-10))[mask_target])

    # if "origin_coord" in data_dict:
    #     coords = data_dict["origin_coord"]

    # else:
    #     logger.warning("Neighbor voting requires 'origin_coord' in data_dict, skipped..")
    # if "origin_instance" in data_dict:
    #     pred = clustering_voting(pred, data_dict["origin_instance"], ignore_index)

# intersection, union, target = intersection_and_union(
#     pred, segment, self.num_classes, self.ignore_index
# )
# intersection_meter.update(intersection)
# union_meter.update(union)
# target_meter.update(target)
