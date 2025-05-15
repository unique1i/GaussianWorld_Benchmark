import os
import pdb
from tqdm import tqdm
import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Quantize_kMeans():
    def __init__(self, num_clusters=64, num_leaf_clusters=10, num_iters=10, dim=9, dim_leaf=6):
        self.num_clusters = num_clusters            # k1
        self.leaf_num_clusters = num_leaf_clusters  # k2
        self.num_kmeans_iters = num_iters           # iter
        self.vec_dim = dim                          # coarse-level, dim=9(feat+xyz)
        self.leaf_vec_dim = dim_leaf                # fine-level, dim=6(feat)
        self.centers = torch.empty(0)               # coarse center， [k1, 9]
        self.leaf_centers = torch.empty(0)          # fine center， [k2, 6]
        self.iLeafSubNum = torch.empty(0)           # Number of fine clusters per coarse cluster
        self.cls_ids = torch.empty(0)               # coarse cluster id [num_pts]
        self.leaf_cls_ids = torch.empty(0)          # fine cluster id[num_pts]
        
        self.nn_index = torch.empty(0)              # [num_pts] temporary variable

        # for update_centers
        self.cluster_ids = torch.empty(0)
        self.excl_clusters = []
        self.excl_cluster_ids = []
        self.cluster_len = torch.empty(0)
        self.max_cnt = 0                  
        self.max_cnt_th = 10000
        self.n_excl_cls = 0       

        self.pos_centers = torch.empty(0)           

    def get_dist(self, x, y, mode='sq_euclidean'):
        """Calculate distance between all vectors in x and all vectors in y.

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean_chunk':
            step = 65536
            if x.shape[0] < step:
                step = x.shape[0]
            dist = []
            for i in range(np.ceil(x.shape[0] / step).astype(int)):
                dist.append(torch.cdist(x[(i*step): (i+1)*step, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

    # Update centers in non-cluster assignment iters using cached nn indices.
    def update_centers(self, feat, mode="root", selected_leaf=-1):
        if mode == "root":
            centers = self.centers
            num_clusters = self.num_clusters
            vec_dim = self.vec_dim
        elif mode == "leaf":
            centers = self.leaf_centers
            num_clusters = self.num_clusters * self.leaf_num_clusters + 1
            vec_dim = self.leaf_vec_dim
        feat = feat.detach().reshape(-1, vec_dim)  # [num_pts, dim] [766267, 9]
        # Update all clusters except the excluded ones in a single operation
        # Add a dummy element with zeros at the end
        feat = torch.cat([feat, torch.zeros_like(feat[:1]).cuda()], 0)  # [num_pts+1, dim]
        centers = torch.sum(feat[self.cluster_ids, :].reshape(
            num_clusters, self.max_cnt, -1), dim=1)    # [num_clusters, vec_dim]
        if len(self.excl_cluster_ids) > 0:
            for i, cls in enumerate(self.excl_clusters):
                # Division by num_points in cluster is done during the one-shot averaging of all
                # clusters below. Only the extra elements in the bigger clusters are added here.
                centers[cls] += torch.sum(feat[self.excl_cluster_ids[i], :], dim=0)
        centers /= (self.cluster_len + 1e-6)

    # Update centers during cluster assignment using mask matrix multiplication
    # Mask is obtained from distance matrix
    def update_centers_(self, feat, cluster_mask=None, nn_index=None, avg=False):
        # feat = feat.detach().reshape(-1, self.vec_dim)
        centers = (cluster_mask.T @ feat)   # [1w, num_cluster] * [1w, dim] -> [num_cluster, dim]
        # if avg:
        #     self.centers /= counts.unsqueeze(-1)
        return centers

    def equalize_cluster_size(self, mode="root"):
        """Make the size of all the clusters the same by appending dummy elements.

        """
        # Find the maximum number of elements in a cluster, make size of all clusters
        # equal by appending dummy elements until size is equal to size of max cluster.
        # If max is too large, exclude it and consider the next biggest. Use for loop for
        # the excluded clusters and a single operation for the remaining ones for
        # updating the cluster centers.

        unq, n_unq = torch.unique(self.nn_index, return_counts=True)
        # Find max cluster size and exclude clusters greater than a threshold
        topk = 100
        if len(n_unq) < topk:
            topk = len(n_unq)
        max_cnt_topk, topk_idx = torch.topk(n_unq, topk)
        self.max_cnt = max_cnt_topk[0]
        idx = 0
        self.excl_clusters = []
        self.excl_cluster_ids = []
        while(self.max_cnt > self.max_cnt_th):
            self.excl_clusters.append(unq[topk_idx[idx]])
            idx += 1
            if idx < topk:
                self.max_cnt = max_cnt_topk[idx]
            else:
                break
        self.n_excl_cls = len(self.excl_clusters)
        self.excl_clusters = sorted(self.excl_clusters)
        # Store the indices of elements for each cluster
        all_ids = []
        cls_len = []
        if mode == "root":
            num_clusters = self.num_clusters
        elif mode == "leaf":
            num_clusters = self.num_clusters * self.leaf_num_clusters + 1
        for i in range(num_clusters):
            cur_cluster_ids = torch.where(self.nn_index == i)[0]
            # For excluded clusters, use only the first max_cnt elements
            # for averaging along with other clusters. Separately average the
            # remaining elements just for the excluded clusters.
            cls_len.append(torch.Tensor([len(cur_cluster_ids)]))
            if i in self.excl_clusters:
                self.excl_cluster_ids.append(cur_cluster_ids[self.max_cnt:])
                cur_cluster_ids = cur_cluster_ids[:self.max_cnt]
            # Append dummy elements to have same size for all clusters
            all_ids.append(torch.cat([cur_cluster_ids, -1 * torch.ones((self.max_cnt - len(cur_cluster_ids)),
                                                                       dtype=torch.long).cuda()]))
        all_ids = torch.cat(all_ids).type(torch.long)
        cls_len = torch.cat(cls_len).type(torch.long)
        self.cluster_ids = all_ids
        self.cluster_len = cls_len.unsqueeze(1).cuda()
        if mode == "root":
            self.cls_ids = self.nn_index
        elif mode == "leaf":
            self.leaf_cls_ids = self.nn_index

    def cluster_assign(self, feat, feat_scaled=None, mode="root", selected_leaf=-1):

        # quantize with kmeans
        feat = feat.detach()    # [N, dim]

        if feat_scaled is None:
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8)
        # init. centers and ids
        if len(self.centers) == 0 and mode == "root":
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]
        if len(self.leaf_centers) == 0 and mode == "leaf":
            # [num_clusters, leaf_num_clusters, dim_leaf] eg. [640, 6]
            self.leaf_centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters * self.leaf_num_clusters+1], :]
            self.leaf_cls_ids = torch.ones(feat.shape[0]).to(torch.int64).cuda() * self.num_clusters * self.leaf_num_clusters

        # start kmeans
        chunk = True
        # tmp centers
        if mode == "root":
            tmp_centers = torch.zeros_like(self.centers)
            counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
        elif mode == "leaf":
            tmp_centers = torch.zeros_like(self.leaf_centers)[:self.leaf_num_clusters, :]
            counts = torch.zeros(self.leaf_num_clusters, dtype=torch.float32).cuda() + 1e-6
            start_id = selected_leaf * self.leaf_num_clusters
            end_id = selected_leaf * self.leaf_num_clusters + self.iLeafSubNum[selected_leaf]
        for iteration in range(self.num_kmeans_iters):
            # chunk for memory issues
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                if mode == "root":
                    while True:
                        dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers)
                        curr_nn_index = torch.argmin(dist, dim=-1)  # [1W]
                        # Assign a single cluster when distance to multiple clusters is same
                        dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)  # [1W, 512]
                        curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)   # [512, 45]
                        counts += dist.detach().sum(0) + 1e-6   # [512]
                        tmp_centers += curr_centers
                        if self.nn_index == None:
                            self.nn_index = curr_nn_index
                        else:
                            self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                        i += 1
                        if i*chunk > feat.shape[0]:
                            break
                elif mode == "leaf":
                    for idx_c in range(self.num_clusters):
                        if idx_c != selected_leaf:
                            continue
                        selected_pts = self.cls_ids == idx_c
                        dist = self.get_dist(feat[selected_pts], self.leaf_centers[start_id:end_id])
                        curr_nn_index = torch.argmin(dist, dim=-1)  # [1W]
                        dist = F.one_hot(curr_nn_index, self.leaf_num_clusters).type(torch.float32)  # [1W, 10]
                        curr_centers = self.update_centers_(feat[selected_pts], dist, curr_nn_index, avg=False)   # [512, 45]
                        counts += dist.detach().sum(0) + 1e-6   # [512]
                        tmp_centers += curr_centers
                        self.leaf_cls_ids[selected_pts] = curr_nn_index + start_id
            # avrage centers
            if mode == "root":
                self.centers = tmp_centers / counts.unsqueeze(-1)   
            elif mode == "leaf":
                self.leaf_centers[start_id: start_id+self.leaf_num_clusters] = tmp_centers / counts.unsqueeze(-1)   
            # Reinitialize to 0
            tmp_centers[tmp_centers != 0] = 0.
            counts[counts > 0.1] = 0.

        # Reassign ID according to the new centers
        if chunk:
            self.nn_index = None
            i = 0
            # chunk = 100000
            if mode == "root":
                while True:
                    dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i * chunk > feat.shape[0]:
                        break
            elif mode == "leaf":
                for idx_c in range(self.num_clusters):
                    if idx_c != selected_leaf:
                        continue
                    selected_pts = self.cls_ids == idx_c
                    dist = self.get_dist(feat[selected_pts], self.leaf_centers[start_id:end_id])
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    self.leaf_cls_ids[selected_pts] = curr_nn_index + start_id
                self.nn_index = self.leaf_cls_ids
        self.equalize_cluster_size(mode=mode)

    def rescale(self, feat, scale=None):
        """Scale the feature to be in the range [-1, 1] by dividing by its max value.

        """
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)

    def forward(self, gaussian, iteration, assign=False, mode="root", selected_leaf=-1, pos_weight=1.0):
        if mode == "root":
            # (1) coarse-level: feature + xyz
            scale = pos_weight     # TODO
            xyz_feat = gaussian._xyz.detach() * scale
            feat = torch.cat((gaussian._ins_feat, xyz_feat), dim=1)    # [N, 9]
        elif mode == "leaf":
            # (2) fine-level: feature only
            feat = gaussian._ins_feat

        if assign:
            self.cluster_assign(feat, mode=mode, selected_leaf=selected_leaf)   # gaussian._ins_feat
        else:
            self.update_centers(feat, mode=mode, selected_leaf=selected_leaf)   # gaussian._ins_feat

        if mode == "root":
            centers = self.centers
            vec_dim = self.vec_dim
        elif mode == "leaf":
            centers = self.leaf_centers
            vec_dim = self.leaf_vec_dim
        sampled_centers = torch.gather(centers, 0, self.nn_index.unsqueeze(-1).repeat(1, vec_dim))
        # NOTE: "During backpropagation, the gradients of the quantized features are copied to the instance features", mentioned in the paper.
        gaussian._ins_feat_q = gaussian._ins_feat - gaussian._ins_feat.detach() + sampled_centers[:,:6]

    def replace_with_centers(self, gaussian):
        deg = gaussian._features_rest.shape[1]
        sampled_centers = torch.gather(self.centers, 0, self.nn_index.unsqueeze(-1).repeat(1, self.vec_dim))
        gaussian._features_rest = gaussian._features_rest - gaussian._features_rest.detach() + sampled_centers.reshape(-1, deg, 3)
