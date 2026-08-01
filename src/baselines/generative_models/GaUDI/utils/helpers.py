
import os, sys
current_file_path = os.path.realpath(__file__)
parant_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parant_path)

from typing import Tuple
import json

import torch
from torch import Tensor

from cond_prediction.prediction_args import PredictionArgs
from data.aromatic_dataloader import RINGS_LIST
from utils.args_edm import Args_EDM

bn_bn_dist = {"min": 2.399, "mean": 2.445, "max": 2.481, "thr": 0.01}
bn_bn_angels3_dict = {  # 0.001 and 0.999 quantiles
    "120": (105.772, 133.193),
    "180": (177.333, 183.089),
    "240": (227.120, 255.250),
}

angels3_dict_hetro = {
    "Bn": {
        "120": (115.5796, 124.6503),
        "180": (178.4123, 180.0),
    },
    "Pd": {
        "120": (120.496, 125.6031),
        "180": (174.3335, 180.0),
    },
    "Pz": {
        "180": (176.938, 180.0),
    },
    "Tr": {
        "0": (0., 0.),
    },
    "Td": {
        "0": (0., 0.),
    },
    "Sed": {
        "0": (0., 0.),
    },
    "Cyc": {
        "140": (136.5737, 145.7095),
    },
    "Pl": {
        "140": (141.236, 151.4487),
    },
    "Fu": {
        "140": (145.7866, 154.0212),
    },
    "Si": {
        "140": (128.6004, 136.8009),
    },
    "Th": {
        "140": (130.2688, 142.7601),
    },
    "Sel": {
        "140": (132.217, 141.3974),
    },

    # "Bl": {
    #     "140": (127.3096694946289, 145.93600463867188),
    # },
    # "Bn": {
    #     "120": (108.33101654052734, 127.21441650390625),
    #     "180": (170.7755126953125, 180.0),
    # },
    # "Db": {
    #     "180": (156.42091369628906, 180.0),
    # },
    # "Fu": {
    #     "140": (135.90780639648438, 153.3458251953125),
    # },
    # "Pl": {
    #     "140": (134.00990295410156, 151.88079833984375),
    # },
    # "Bz": {
    #     "120": (108.01634216308594, 123.69662475585938),
    #     "180": (169.33651733398438, 179.944580078125),
    # },
    # "Pz": {
    #     "180": (168.29324340820312, 180.0),
    # },
    # "Pd": {
    #     "120": (108.94857788085938, 126.54322052001953),
    #     "180": (168.7400360107422, 179.96141052246094),
    # },
    # "Th": {
    #     "140": (126.71401977539062, 142.5613555908203),
    # },
    # "Cbd": {
    #     "180": (155.19215393066406, 180.0),
    # },
}
angels3_dict = {"cata": {"Bn": bn_bn_angels3_dict}, "hetro": angels3_dict_hetro}

angels4_dict = {
    "cata": {  # 0.01 quantile
        "0": 43.943,
        "180": 135.031,
    },
    "hetro": {  # 0.01 quantile
        "0": 42.01443862915039,
        "180": 139.9242706298828,
    },
}
analyzed_rings = {
    "cata": {
        "n_nodes": {
            11: 20559,
            10: 5164,
            9: 1349,
            8: 363,
            7: 108,
            5: 11,
            6: 32,
            3: 2,
            4: 3,
            1: 1,
            2: 1,
        },
        "ring_types": None,  # {0: 303523},
        "distances": None,
        # [270, 434, 972, 1518, 1990, 2758, 3414, 3458, 4022, 4540, 5246, 5532, 538744, 6196, 6542, 6556, 10008, 12276, 11214, 11858, 20486, 416236, 19804, 29788, 188832, 103086, 12392, 12434, 12766, 11822, 8860, 30886, 329638, 68412, 24956, 18418, 76588, 76760, 10590, 10952, 8780, 12796, 102274, 56874, 117212, 48644, 11988, 12820, 25174, 38428, 10530, 8724, 13620, 70440, 32986, 24980, 50426, 18560, 7818, 5448, 7150, 15068, 9172, 12822, 25654, 28590, 11984, 7548, 16716, 8068, 3424, 2026, 2328, 4546, 9494, 13400, 7634, 7602, 2894, 1644, 3890, 3128, 986, 672, 982, 3612, 5154, 2070, 1908, 1324, 456, 182, 656, 850, 248, 212, 1116, 520, 884, 1298]
    },
    "hetro": {
        "n_nodes": {
            13: 26220,
            12: 52125,
            11: 52397,
            10: 35022,
            9: 17540,
            8: 6999,
            7: 2263,
            6: 595,
            5: 159,
            4: 33,
            3: 10,
            2: 7,
        },
    },
}

ring_distances_hetro = {
    "Bn-Cyc": (2.14, 2.24),
    "Bn-Bn": (2.35, 2.51),
    "Bn-Th": (2.19, 2.29),
    "Cyc-Th": (1.97, 2.04),
    "Cyc-Pl": (1.9, 1.95),
    "Bn-Pl": (2.12, 2.2),
    "Td-Th": (1.97, 1.99),
    "Pl-Pl": (1.89, 1.91),
    "Pl-Th": (1.95, 2.0),
    "Bn-Pd": (2.36, 2.48),
    "Cyc-Si": (1.98, 2.1),
    "Th-Tr": (1.92, 1.94),
    "Bn-Si": (2.22, 2.36),
    "Th-Th": (2.02, 2.09),
    "Bn-Sed": (2.16, 2.21),
    "Cyc-Cyc": (1.94, 1.99),
    "Bn-Tr": (2.09, 2.14),
    "Pz-Th": (2.13, 2.18),
    "Bn-Pz": (2.34, 2.4),
    "Pd-Th": (2.17, 2.27),
    "Cyc-Pd": (2.12, 2.22),
    "Bn-Sel": (2.23, 2.32),
    "Si-Th": (2.04, 2.15),
    "Pd-Pl": (2.09, 2.18),
    "Bn-Td": (2.14, 2.19),
    "Sel-Th": (2.05, 2.12),
    "Cyc-Pz": (2.09, 2.13),
    "Pl-Pz": (2.06, 2.09),
    "Pd-Sel": (2.19, 2.3),
    "Cyc-Sel": (2.0, 2.07),
    "Cyc-Td": (1.92, 1.93),
    "Pd-Td": (2.1, 2.17),
    "Pl-Sel": (1.97, 2.03),
    "Pd-Pd": (2.34, 2.43),
    "Pz-Td": (2.06, 2.09),
    "Pd-Tr": (2.05, 2.11),
    "Bn-Fu": (2.12, 2.19),
    "Cyc-Tr": (1.87, 1.89),
    "Fu-Th": (1.94, 1.99),
    "Cyc-Fu": (1.89, 1.94),
    "Sed-Th": (1.99, 2.0),
    "Pl-Tr": (1.84, 1.85),
    "Pl-Td": (1.88, 1.9),
    "Pd-Pz": (2.31, 2.38),
    "Cyc-Sed": (1.94, 1.95),
    "Pz-Sel": (2.16, 2.22),
    "Fu-Pl": (1.88, 1.91),
    "Pd-Si": (2.2, 2.33),
    "Sel-Td": (2.0, 2.02),
    "Sel-Tr": (1.95, 1.97),
    "Pl-Si": (1.95, 2.07),
    "Sel-Sel": (2.08, 2.14),
    "Fu-Pd": (2.09, 2.16),
    "Pz-Tr": (2.01, 2.03),
    "Sel-Si": (2.08, 2.18),
    "Fu-Tr": (1.83, 1.83),
    "Si-Td": (2.03, 2.05),
    "Pz-Pz": (2.27, 2.31),
    "Pd-Sed": (2.13, 2.18),
    "Fu-Pz": (2.05, 2.07),
    "Fu-Sel": (1.95, 2.01),
    "Fu-Td": (1.87, 1.88),
    "Sed-Sel": (2.02, 2.03),
    "Pl-Sed": (1.9, 1.91),
    "Fu-Fu": (1.87, 1.87),
    "Pz-Si": (2.17, 2.26),
    "Si-Tr": (1.99, 2.0),
    "Pz-Sed": (2.09, 2.11),
    "Si-Si": (2.2, 2.21),
    "Fu-Si": (2.03, 2.04),
    "Sed-Si": (2.06, 2.06),
    "Fu-Sed": (1.89, 1.89),

    # Sed, Td, Tr三种环之间只能形成二环，无法得到大环，此处不考虑他们的六种组合的间距信息
    "Sed-Sed": (1.8, 2.),
    "Td-Td": (1.8, 2.),
    "Tr-Tr": (1.8, 2.),
    "Sed-Td": (1.8, 2.),
    "Sed-Tr": (1.8, 2.),
    "Tr-Td": (1.8, 2.),
}

ring_distances_cata = {
    "Bn-Bn": (2.42, 2.48),
}
ring_distances = {
    "cata": ring_distances_cata,
    "peri": ring_distances_cata,
    "hetro": ring_distances_hetro,
}


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
    x: Tensor, ring_type, tol=0.1, dataset="cata"
) -> Tuple[Tensor, Tensor]:
    # 当x的矩阵形状为【1 * N * 3】；dist最终为【1 * N * N】，表示所有环质心之间的间距
    dist = coord2distances(x)
    if len(ring_type.shape) == 3:
        ring_type = ring_type.argmax(2)
    # adj的形状与dist保持一致
    adj = torch.zeros(dist.shape[0], dist.shape[1], dist.shape[1])
    for b in range(dist.shape[0]):
        for i in range(dist.shape[1]):
            for j in range(i + 1, dist.shape[1]):
                si = RINGS_LIST[dataset][ring_type[b, i]]
                sj = RINGS_LIST[dataset][ring_type[b, j]]
                key = f"{si}-{sj}"
                if key not in ring_distances[dataset]:
                    key = f"{sj}-{si}"

                if key in ring_distances[dataset]:
                    a = ring_distances[dataset][key][0]
                    bb = dist[b, i, j]
                    c = ring_distances[dataset][key][1]

                # 考虑环环间距上下限
                if key in ring_distances[dataset] and ring_distances[dataset][key][0] * (1 - tol) < dist[b, i, j] < ring_distances[dataset][key][1] * (1 + tol):
                    adj[b, i, j] = 1
                    adj[b, j, i] = 1

    return dist, adj

def switch_grad_off(models):
    for m in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

def get_edm_args(exp_dir_path):
    args = Args_EDM().parse_args([])
    with open(exp_dir_path + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.exp_dir = exp_dir_path
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return args

def get_cond_predictor_args(exp_dir_path):
    args = PredictionArgs().parse_args([])
    with open(exp_dir_path + "/args.txt", "r") as f:
        args.__dict__ = json.load(f)
    args.restore = True
    args.exp_dir = exp_dir_path
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    return args
