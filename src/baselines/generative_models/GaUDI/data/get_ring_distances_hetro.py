
import os, sys
current_file_path = os.path.realpath(__file__)
parant_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parant_path)

from collections import defaultdict

from tqdm import tqdm

from typing import Tuple
from torch import Tensor

import torch
from rdkit import Chem
from rdkit.Chem import Draw

# from diffusion.data import xyz2mol
# from diffusion.data.aromatic_dataloader import (
#     create_data_loaders,
#     RINGS_LIST,
#     ATOMS_LIST,
# )
# from diffusion.data.ring import RINGS_DICT
from aromatic_dataloader import create_data_loaders, RINGS_LIST, get_splits

from utils.args_edm import Args_EDM
# from diffusion.utils.args_edm import Args_EDM
# from diffusion.utils.ring_graph import NO_ORIENTATION_RINGS
from utils.helpers import positions2adj, ring_distances_hetro
import matplotlib.pyplot as plt
import numpy as np


def coord2distances(x):
    # x 输入为【1*N*3】，表示一个分子中所有环的质心，变化为【1*N*1*3】
    x = x.unsqueeze(2)
    # x_t 为 【1*1*N*3】
    x_t = x.transpose(1, 2)
    # dist最终为【1 * N * N】，表示所有环质心之间的间距
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    return dist


def positions2adj(
    x: Tensor, ring_type, adj_full, ring_distances_dict, dataset="cata",
) -> Tuple[Tensor, Tensor]:
    # 当x的矩阵形状为【1 * N * 3】；dist最终为【1 * N * N】，表示所有环质心之间的间距
    dist = coord2distances(x)
    dis = dist.squeeze()
    if len(ring_type.shape) == 3:
        ring_type = ring_type.argmax(2)
    # adj的形状与dist保持一致
    adj = adj_full.int()
    adjj = adj.squeeze()

    loop = adj.nonzero()
    for b, i, j in loop:
        si = RINGS_LIST[dataset][ring_type[b, i]]
        sj = RINGS_LIST[dataset][ring_type[b, j]]
        key_list = sorted([si, sj])
        key = f"{key_list[0]}-{key_list[1]}"
        ring_distances_dict[key].append(dist[b, i, j].item())

    # for b in range(dist.shape[0]):
    #     for i in range(dist.shape[1]):
    #         for j in range(i + 1, dist.shape[1]):
    #             si = RINGS_LIST[dataset][ring_type[b, i]]
    #             sj = RINGS_LIST[dataset][ring_type[b, j]]
    #             key_list = sorted([si, sj])
    #             key = f"{key_list[0]}-{key_list[1]}"
    #             ring_distances_dict[key].append(dist[b, i, j].item())

    return ring_distances_dict


def gor2goa(x, rings_types, adj_full, ring_distances_dict, dataset="cata"):
    if dataset == "cata":
        n = x.shape[0]
    else:
        n = x.shape[0] // 2

    # x[None, :n]会给矩阵x在0维添加维度，从N * 3 -》 1 * N * 3；adj是一【1*N*N】的矩阵，表示环与环之间的邻接情况，考量了环质心间距等因素
    ring_distances_dict = positions2adj(x[None, :n], rings_types[None, :n], adj_full[None, :n, :n], ring_distances_dict, dataset=dataset)

    return ring_distances_dict


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.device = "cpu"
    args.tol = 0.1
    args.sample_rate = 1
    args.return_adj = True
    # args.batch_size = 2
    # args.num_workers = 0
    # args.dataset = "cata"
    # args.orientation = False
    # args.target_features = "GAP_eV"
    train_loader = create_data_loaders(args)

    np.random.seed(12)
    ring_distances_dict = defaultdict(list)
    # for i in np.random.randint(0, len(train_loader.dataset), 1):
    for i in tqdm(range(len(train_loader.dataset))):
        x, node_mask, edge_mask, node_features, adj_full, y = train_loader.dataset[i]
        node_mask = node_mask.bool()
        # atoms_positions: torch格式，【原子数 * 2】的矩阵，表示在xy平面上的坐标
        # atoms_types： torch格式，表示原子种类，【原子数】的矩阵
        # bonds：列表格式，[[1,2],[2,3]...]形式，表示原子之间的键接
        a, b = x[node_mask], node_features[node_mask].argmax(1)
        ring_distances_dict = gor2goa(x[node_mask], node_features[node_mask].argmax(1), adj_full[node_mask][:, node_mask], ring_distances_dict, args.dataset)

    print(ring_distances_dict)

    refreshed_ring_distances_dict = {}
    for key, value in ring_distances_dict.items():
        refreshed_ring_distances_dict[key] = (round(min(value), 2), round(max(value), 2))

    for key, value in refreshed_ring_distances_dict.items():
        print('"' + key + '": ', value, ',')