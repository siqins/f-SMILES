import os, sys
current_file_path = os.path.realpath(__file__)
parant_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parant_path)

import pandas as pd
import torch
from torch import Tensor
import networkx as nx
from data.aromatic_dataloader import create_data_loaders, RINGS_LIST, get_splits
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils.args_edm import Args_EDM
from utils.helpers import (
    positions2adj,
    ring_distances,
    angels3_dict,
    angels4_dict,
)

def angel3(p):
    v1 = p[0] - p[1]
    v2 = p[2] - p[1]
    dot_product = torch.dot(v1, v2)
    norm_product = torch.norm(v1) * torch.norm(v2)
    a = torch.rad2deg(torch.acos(dot_product / norm_product))
    return a if a >= 0 else a + 360


def angel4(p):
    """Praxeolitic formula for dihedral angle"""
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.dot(b0, b1) * b1
    w = b2 - torch.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(b1, v), w)
    return torch.rad2deg(torch.atan2(y, x)).abs()


def check_angels3(angels3: Tensor, tol=0.1, dataset="cata") -> bool:
    a3_dict = angels3_dict[dataset]
    if len(angels3) == 0:
        return True
    symbols = [a[0] for a in angels3]
    for symbol in set(symbols):
        a3_symbol = torch.stack([a[1] for a in angels3 if a[0] == symbol])
        conds = [
            torch.logical_and(
                q_low * (1 - tol) <= a3_symbol, a3_symbol <= q_high * (1 + tol)
            )
            for q_low, q_high in a3_dict[symbol].values()
        ]
        if not torch.stack(conds).any(dim=0).all():
            return False
    return True


def check_angels4(angels4: Tensor, tol=0.1, dataset="cata") -> bool:
    if len(angels4) == 0 or dataset == "hetro":
        return True
    a4_dict = angels4_dict[dataset]
    angels4 = torch.stack([a for s, a in angels4])
    cond = torch.logical_or(
        a4_dict["180"] * (1 - tol) <= angels4, angels4 <= a4_dict["0"] * (1 + tol)
    )
    return cond.all()


def find_triplets_quads(adj: Tensor, x: Tensor, ring_types: Tensor, dataset="cata"):
    rings_list = RINGS_LIST[dataset]
    if len(ring_types.shape) == 2:
        ring_types = ring_types.argmax(1)
    rings = [rings_list[i] for i in ring_types]
    g = nx.from_numpy_array(adj.numpy())
    triplets = []
    for n1, n2 in nx.bfs_edges(g, 0):
        for n3 in g.neighbors(n1):
            if n3 != n2:
                triplets.append((n2, n1, n3))
        for n3 in g.neighbors(n2):
            if n3 != n1:
                triplets.append((n1, n2, n3))
    triplets = [(n1, n2, n3) if n1 < n3 else (n3, n2, n1) for n1, n2, n3 in triplets]
    triplets = list(set(triplets))
    # save all the angels with the center ring type
    # if any([angel3(x[nodes, :])<60 for nodes in triplets]):
    #     print([angel3(x[nodes, :]) for nodes in triplets])
    angels3 = [(rings[nodes[1]], angel3(x[nodes, :])) for nodes in triplets]

    angular_triplets = [t for t in triplets if not 170 < angel3(x[t, :]) < 190]
    # quads
    quads = []
    for n1, n2, n3 in angular_triplets:
        for n4 in g.neighbors(n1):
            if n4 not in [n2, n3]:
                # check the new angle is not linear
                if not 175 < angel3(x[[n4, n1, n2]]) < 185:
                    quads.append((n4, n1, n2, n3))
        for n4 in g.neighbors(n3):
            if n4 not in [n1, n2]:
                # check the new angle is not linear
                if not 175 < angel3(x[[n2, n3, n4]]) < 185:
                    quads.append((n1, n2, n3, n4))
    quads = [
        (n1, n2, n3, n4) if n1 < n4 else (n4, n3, n2, n1) for n1, n2, n3, n4 in quads
    ]
    quads = list(set(quads))
    angels4 = [
        ([rings[nodes[i]] for i in range(4)], angel4(x[nodes, :])) for nodes in quads
    ]

    # if any([80<a[1]<100 for a in angels4]):
    #     print([a[1] for a in angels4])

    return angels3, angels4


def get_angels(xs: Tensor, ring_types, adjs, node_masks=None, dataset="cata"):
    """Extract list of angels from batch of nodes"""
    # _, adjs = positions2adj(xs, ring_types, dataset)
    angels3 = []
    angels4 = []
    for i in range(xs.shape[0]):
        adj = adjs[i]
        x = xs[i]
        ring_type = ring_types[i]
        if node_masks is not None:
            node_mask = node_masks[i].bool()
            adj = adj[node_mask][:, node_mask]
            x = x[node_mask]
            ring_type = ring_type[node_mask]
        a3, a4 = find_triplets_quads(adj, x, ring_type, dataset)
        angels3 += a3
        angels4 += a4

    return angels3, angels4


def check_stability(positions, ring_type, angel_3_dict, angel_4_dict, tol=0.1, dataset="cata") -> dict:
    results = {
        "orientation_nodes": True,
        "dist_stable": False,
        "connected": False,
        "angels3": False,
        "angels4": False,
    }
    if isinstance(positions, np.ndarray):
        positions = Tensor(positions)
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    if len(ring_type.shape) == 2:
        ring_type = ring_type.argmax(1)

    if dataset != "cata":  # orientation nodes
        n_rings = torch.div(positions.shape[0], 2, rounding_mode="trunc")
        positions = positions[:n_rings]
        # check orientation nodes
        orientation_ring_type = len(RINGS_LIST["hetro"]) - 1
        if (
            set(ring_type[n_rings:].numpy()) != set([orientation_ring_type])
            or orientation_ring_type in ring_type[:n_rings]
        ):
            results["orientation_nodes"] = False
            return results
        ring_type = ring_type[:n_rings]
    n_rings = positions.shape[0]
    dist, adj = positions2adj(
        positions[None, :, :], ring_type[None, :], tol, dataset=dataset
    )
    dist = dist[0]
    adj = adj[0]
    min_dist = min([r[0] for r in ring_distances[dataset].values()])
    if ((dist < min_dist * (1 - tol)) * (1 - torch.eye(n_rings))).bool().any():
        return results
    else:
        results["dist_stable"] = True

    g = nx.from_numpy_array(adj.numpy())
    if not nx.is_connected(g):
        return results
    else:
        results["connected"] = True

    angels3, angels4 = get_angels(
        positions[None, :, :], ring_type[None, :], adj[None, :, :], dataset=dataset
    )

    for angle in angels3:
        angel_3_dict[angle[0]].append(angle[1])

    for angle in angels4:
        angel_4_dict[tuple(angle[0])].append(angle[1])

    return angel_3_dict, angel_4_dict


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.device = "cpu"
    args.tol = 0.1
    args.sample_rate = 1
    # args.batch_size = 2
    # args.num_workers = 0
    # args.dataset = "cata"
    # args.orientation = False
    # args.target_features = "GAP_eV"
    train_loader= create_data_loaders(args)

    np.random.seed(0)

    angel_3_dict = defaultdict(list)
    angle_4_dict = defaultdict(list)
    for i in tqdm(range(len(train_loader.dataset))):
        x, node_mask, edge_mask, node_features, y = train_loader.dataset[i]
        node_mask = node_mask.bool()
        # atoms_positions: torch格式，【原子数 * 2】的矩阵，表示在xy平面上的坐标
        # atoms_types： torch格式，表示原子种类，【原子数】的矩阵
        # bonds：列表格式，[[1,2],[2,3]...]形式，表示原子之间的键接

        angel_3_dict, angel_4_dict = check_stability(x[node_mask], node_features[node_mask].argmax(1), angel_3_dict, angle_4_dict,
                                           tol=0.1, dataset=args.dataset)

    os.makedirs('angles', exist_ok=True)
    for angle in angel_3_dict.items():
        df = pd.DataFrame()
        df[angle[0]] = angle[1]
        df.to_csv(f'angles/angle-{angle[0]}.csv', index=False)

    for angle in angel_4_dict.items():
        df = pd.DataFrame()
        df[angle[0]] = angle[1]
        df.to_csv(f'angles/angle-{angle[0]}.csv', index=False)