import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree as KDTree
import clip
from segmentation import make_encoder
import trimesh

###################################
# 1. Basic I/O Utilities
###################################


def load_scene_list(val_split_path):
    """
    Reads a .txt file listing validation scenes (one per line).
    Returns a list of scene IDs (strings).
    """
    with open(val_split_path, "r") as f:
        lines = f.readlines()
    scene_ids = [line.strip() for line in lines if len(line.strip()) > 0]
    return scene_ids


def read_ply_file_3dgs(file_path):
    """
    Reads the 3D Gaussian ply (e.g. point_cloud_30000.ply).
    Returns xyz and opacity.
    """
    ply_data = PlyData.read(file_path)
    vertex = ply_data["vertex"]
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    opacity = vertex["opacity"]
    xyz = np.stack([x, y, z], axis=-1)
    return xyz, opacity

    # parser.add_argument("--output_root", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannetpp")
    # parser.add_argument("--label_path", type=str, default="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt")


clip_pretrained, _ = make_encoder(
    "clip_vitl16_384",
    features=256,
    groups=1,
    expand=False,
    exportable=False,
    hooks=[5, 11, 17, 23],
    use_readout="project",
)


#
# label_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt'
# gs_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetpp_v1_mcmc_1.5M_3dgs/09c1414f1b/ckpts/point_cloud_30000.ply'
# labeled_pc_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannetpp/09c1414f1b/semantic_point_clouds_no_filter.ply'
# gs_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetpp_v1_mcmc_1.5M_3dgs/0d2ee665be/ckpts/point_cloud_30000.ply'
# labeled_pc_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannetpp/0d2ee665be/semantic_point_clouds_no_filter.ply'

# gs_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetpp_v1_mcmc_1.5M_3dgs/38d58a7a31/ckpts/point_cloud_30000.ply'
# labeled_pc_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannetpp/38d58a7a31/semantic_point_clouds_no_filter.ply'

# fname = 'scene0329_01'
# benchname = 'scannet20'
# label_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/encoders/lseg_encoder/metadata/scannet_label20.txt'
# gs_path = '/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/mcmc_3dgs/scene0329_01/ckpts/point_cloud_30000.ply'
# labeled_pc_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannet/scene0329_01/semantic_point_clouds_no_filter_20.ply'


# fname = 'scene0435_01'
# benchname = 'scannet20'
# label_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/encoders/lseg_encoder/metadata/scannet_label20.txt'
# gs_path = '/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/mcmc_3dgs/scene0435_01/ckpts/point_cloud_30000.ply'
# labeled_pc_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannet/scene0435_01/semantic_point_clouds_no_filter_20.ply'


# fname = 'scene0329_01'
# benchname = 'scannet200'
# label_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/encoders/lseg_encoder/metadata/scannet_label200.txt'
# gs_path = '/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/mcmc_3dgs/scene0329_01/ckpts/point_cloud_30000.ply'
# labeled_pc_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannet/scene0329_01/semantic_point_clouds_no_filter_200.ply'


fname = "scene0435_01"
benchname = "scannet200"
label_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/encoders/lseg_encoder/metadata/scannet_label200.txt"
gs_path = "/srv/beegfs02/scratch/qimaqi_data/data/gaussianworld_subset/scannet_mini_val_set_suite/mcmc_3dgs/scene0435_01/ckpts/point_cloud_30000.ply"
labeled_pc_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/feature-3dgs_Qi/output/scannet/scene0435_01/semantic_point_clouds_no_filter_200.ply"


# ---- 3.1 Load label names & encode text ----
with open(label_path, "r") as f:
    label_names = [line.strip() for line in f if len(line.strip()) > 0]
prompt_list = ["this is a " + name for name in label_names]


with torch.no_grad():
    text = clip.tokenize(label_names)
    text = text.cuda()  # text = text.to(x.device) # TODO: need use correct device
    text_feat = clip_pretrained.encode_text(text)  # torch.Size([150, 512])
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu()  # shape: (150, 512)

text_feat_np = text_feat.numpy()  # shape: (150, 512)
np.save(f"./{benchname}_text_embeddings_lseg_feature_3dgs.npy", text_feat_np)

gauss_xyz, _ = read_ply_file_3dgs(gs_path)

ply_data = trimesh.load(labeled_pc_path)
label_color = ply_data.visual.vertex_colors[:, 0]
label_xyz = ply_data.vertices
valid_label = label_color != 255
label_color = label_color[valid_label]  # label masks
label_xyz = label_xyz[valid_label]
text_feat_np_label = text_feat_np[label_color]  # shape: (M, 512)

# (c) Build KDTree => NN search
kd_tree = KDTree(label_xyz)  # label_xyz gauss_xyz
_, nn_idx = kd_tree.query(gauss_xyz)  # shape (M,) # label_xyz
nn_lang_feat = text_feat_np_label[nn_idx]  # (M, 512)
gs_vote_label = label_color[nn_idx]  # (M, 512)

print("nn_lang_feat", nn_lang_feat.shape)
print("gauss_xyz", gauss_xyz.shape)
print("gs_vote_label", gs_vote_label.shape)
# save nn_lang_feat
np.save(f"./{fname}_{benchname}_feature_3dgs_text_feat.npy", nn_lang_feat)
np.save(f"./{fname}_{benchname}_feature_3dgs_vote_label.npy", gs_vote_label)


# lang_feat_folder = os.path.join(scannetpp_langfeat_root, scene_id)
# langfeat_path = os.path.join(lang_feat_folder, "langfeat.pth")
# if not os.path.isfile(langfeat_path):
#     print(f"[Warning] Language feature .pth not found for scene {scene_id}")
#     continue
# gauss_lang_feat = torch.load(langfeat_path)[0].cpu()  # (G, 512)
# print(f"\nLoaded {gauss_xyz.shape[0]} 3DGS and {gauss_lang_feat.shape[0]} language features for {scene_id}")

# Filter out zero vectors if needed
# norms = gauss_lang_feat.norm(dim=1)
# keep_mask = (norms > 0)
# gauss_xyz = gauss_xyz[keep_mask.numpy()]
# gauss_lang_feat = gauss_lang_feat[keep_mask]
# if gauss_xyz.shape[0] == 0:
#     print(f"[Warning] All 3DGS zero feats in {scene_id}")
#     continue
