
import random
import sys
from pathlib import Path
from time import time
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import pandas as pd
from torch import zeros, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm

import os, sys
current_file_path = os.path.realpath(__file__)
parant_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parant_path)

from data.mol import Mol, load_xyz, from_rdkit
from data.ring import RINGS_DICT
from utils.args_edm import Args_EDM
from utils.ring_graph import get_rings, get_rings_adj
from utils.molgraph import get_connectivity_matrix, get_edges

DTYPE = torch.float32
INT_DTYPE = torch.int8
# ATOMS_LIST = __ATOM_LIST__[:8]
ATOMS_LIST = {
    # "hetro": ["H", "C", "B", "N", "O", "S"],
    "hetro": ["H", "C", "N", "O", "Si", "S", "Se"],
}
RINGS_LIST = {
    "cata": ["Bn"],
    "peri": ["Bn"],
    "hetro": list(RINGS_DICT.keys()) + ["."],
}


class RandomRotation(object):
    def __call__(self, x):
        M = torch.randn(3, 3)
        Q, __ = torch.linalg.qr(M)
        return x @ Q


class AromaticDataset(Dataset):
    def __init__(self, args, task: str = "train"):
        """
        Args:
            args: All the arguments.
            task: Select the dataset to load from (train/val/test).
        """
        self.csv_file, self.xyz_root = get_paths(args)

        self.task = task
        self.rings_graph = args.rings_graph
        self.normalize = args.normalize
        self.max_nodes = args.max_nodes
        self.return_adj = args.return_adj
        self.dataset = args.dataset
        self.target_features = getattr(args, "target_features", None)
        self.target_features = (
            self.target_features.split(",") if self.target_features else []
        )
        self.orientation = False if self.dataset == "cata" else True
        self._edge_mask_orientation = None
        self.atoms_list = ATOMS_LIST[self.dataset]
        self.knots_list = RINGS_LIST[self.dataset]

        self.df = getattr(args, f"df_{task}").reset_index()
        self.df = self.df[self.df.n_rings <= args.max_nodes].reset_index()
        if args.normalize:
            train_df = args.df_train
            try:
                target_data = train_df[self.target_features].values
            except:
                self.target_features = [
                    t.replace(" ", "") for t in self.target_features
                ]
                target_data = train_df[self.target_features].values
            self.mean = torch.tensor(target_data.mean(0), dtype=DTYPE)
            self.std = torch.tensor(target_data.std(0), dtype=DTYPE)
        else:
            self.std = torch.ones(1, dtype=DTYPE)
            self.mean = torch.zeros(1, dtype=DTYPE)

        self.examples = np.arange(self.df.shape[0])
        if args.sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * args.sample_rate)
            self.examples = self.examples[:num_files]

        x, node_mask, edge_mask, node_features, y = self.__getitem__(0)[:5]
        self.num_node_features = node_features.shape[1]
        self.num_targets = y.shape[0]

    def get_edge_mask_orientation(self):
        if self._edge_mask_orientation is None:
            self._edge_mask_orientation = torch.zeros(
                2 * self.max_nodes, 2 * self.max_nodes, dtype=torch.bool
            )
            for i in range(self.max_nodes):
                self._edge_mask_orientation[i, self.max_nodes + i] = True
                self._edge_mask_orientation[self.max_nodes + i, i] = True
        return self._edge_mask_orientation.clone()

    def __len__(self):
        return len(self.examples)

    def rescale_loss(self, x):
        # Convert from normalized to the original representation
        if self.normalize:
            x = x * self.std.to(x.device).mean()
        return x

    def get_mol(self, df_row, skip_hydrogen=False) -> Tuple[Mol, list, Tensor, str]:
        name = df_row["molecule"]
        file_path = self.xyz_root + "/" + name
        if os.path.exists(file_path + ".xyz"):
            mol = load_xyz(file_path + ".xyz")
            atom_connectivity = get_connectivity_matrix(
                mol.atoms, skip_hydrogen=skip_hydrogen
            )  # build connectivity matrix
            # edges = bonds
        elif os.path.exists(file_path + ".pkl"):
            mol, atom_connectivity = from_rdkit(file_path + ".pkl")
        else:
            raise NotImplementedError(file_path)
        edges = get_edges(atom_connectivity)
        return mol, edges, atom_connectivity, name

    def get_rings(self, df_row):
        name = df_row["molecule"]
        os.makedirs(self.xyz_root + "_rings_preprocessed", exist_ok=True)
        preprocessed_path = self.xyz_root + "_rings_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features, orientation = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row, skip_hydrogen=True)
            # get_figure(mol, edges, showPlot=True, filename='4.png')
            mol_graph = nx.Graph(edges)
            knots = get_rings(mol.atoms, mol_graph)
            adj = get_rings_adj(knots)
            x = torch.tensor([k.get_coord() for k in knots], dtype=DTYPE)
            knot_type = torch.tensor(
                [self.knots_list.index(k.cycle_type) for k in knots]
            ).unsqueeze(1)
            node_features = (
                one_hot(knot_type, num_classes=len(self.knots_list)).squeeze(1).float()
            )
            orientation = [k.orientation for k in knots]
            torch.save([x, adj, node_features, orientation], preprocessed_path)
        # x:环的质心 N * 3, adj:环的邻接矩阵 N * N，node_features：环种类的onehot特征 N * 11，orientation:环的质心或者杂环取向 列表，其中包含取向原子坐标的列表
        return x, adj, node_features, orientation

    def get_atoms(self, df_row):
        name = df_row["molecule"]
        preprocessed_path = self.xyz_root + "_atoms_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row)
            # get_figure(mol, edges, showPlot=True)
            x = torch.tensor([a.get_coord() for a in mol.atoms], dtype=DTYPE)
            atom_element = torch.tensor(
                [self.atoms_list.index(atom.element) for atom in mol.atoms]
            ).unsqueeze(1)
            node_features = (
                one_hot(atom_element, num_classes=len(self.atoms_list))
                .squeeze(1)
                .float()
            )
            adj = atom_connectivity
            torch.save([x, adj, node_features], preprocessed_path)
        return x, adj, node_features

    def get_all(self, df_row):
        # extract targets
        y = torch.tensor(
            df_row[self.target_features].values.astype(np.float32), dtype=DTYPE
        )
        if self.normalize:
            y = (y - self.mean) / self.std

        # creation of nodes, edges and there features
        # x:环的质心 N * 3, adj:环的邻接矩阵 N * N，node_features：环种类的onehot特征 N * 11，orientation:环的质心或者杂环取向 列表，其中包含取向原子坐标的列表
        x, adj, node_features, orientation = self.get_rings(df_row)

        if self.orientation:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            # orientation为[[[]]]格式，从第二层列表中随机抽取一个1 * 3的列表，最终x_r是一个 N * 3的矩阵
            x_r = torch.tensor([random.sample(o, 1)[0] for o in orientation])
            # x_full为最大节点数两倍的
            x_full = zeros(self.max_nodes * 2, 3)
            x_full[:n_nodes] = x
            x_full[self.max_nodes : self.max_nodes + n_nodes] = x_r

            # node_mask用来遮罩x中不为零的部分
            node_mask = zeros(self.max_nodes * 2)
            node_mask[:n_nodes] = 1
            node_mask[self.max_nodes : self.max_nodes + n_nodes] = 1

            node_features_full = zeros(self.max_nodes * 2, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # mark the orientation nodes as additional ring type
            node_features_full[self.max_nodes : self.max_nodes + n_nodes, -1] = 1

            # 大小为max_node * max_node的矩阵，其中包含环个数*环个数的为1的矩阵，其余值为零
            edge_mask_tmp = node_mask[: self.max_nodes].unsqueeze(0) * node_mask[
                : self.max_nodes
            ].unsqueeze(1)
            # mask diagonal，除了对角线为False,其余值为True的矩阵
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask_tmp *= diag_mask # 在edge_mask_tmp的基础上，将对角线元素全设置为零
            edge_mask = self.get_edge_mask_orientation() # edge_mask大小为2*max_node的方形矩阵，形式为布尔值，只有对角线上线max_node处为Ture
            edge_mask[: self.max_nodes, : self.max_nodes] = edge_mask_tmp

            if self.return_adj:
                adj_full = self.get_edge_mask_orientation()
                adj_full[:n_nodes, :n_nodes] = adj
        else:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_full = zeros(self.max_nodes, 3)

            node_mask = zeros(self.max_nodes)
            x_full[:n_nodes] = x
            node_mask[:n_nodes] = 1

            node_features_full = zeros(self.max_nodes, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # node_features_full = zeros(self.max_nodes, 0)

            # edge_mask = zeros(self.max_nodes, self.max_nodes)
            # edge_mask[:n_nodes, :n_nodes] = adj
            # edge_mask = edge_mask.view(-1, 1)

            edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask *= diag_mask
            # edge_mask = edge_mask.view(-1, 1)

            if self.return_adj:
                adj_full = zeros(self.max_nodes, self.max_nodes)
                adj_full[:n_nodes, :n_nodes] = adj

        if self.return_adj:
            return x_full, node_mask, edge_mask, node_features_full, adj_full, y
        else:
            # x_full: max_node*2 : 3的矩阵，其中记录了分子环的所有质心坐标（:max_node），以及偏向角的杂原子坐标（max_node:2*max_node）
            # node_mask : max_node*2 的无维度矩阵，其中以1表示x_full中哪些位置不为零，含有信息
            # edge_mask : 2*max_node * 2*max_node的矩阵，在max_node*max_node的范围内，只有对角线为False,其余为True；在[max_node:]往上
            #            左下和右上的max_node * max_node矩阵的对角线为True,其余全部为False.
            # node_features_full : 在hetro中，为【2*max_node * 环种类】的矩阵，是独热编码，其中最后一个种类为偏向杂原子类型，不表示任何环，
            #                       用于描述x_full中【max_node：】部分的偏向杂原子
            return x_full, node_mask, edge_mask, node_features_full, y

    def __getitem__(self, idx):
        index = self.examples[idx]
        df_row = self.df.loc[index]
        return self.get_all(df_row)


def get_paths(args):
    if not hasattr(args, "dataset"):
        csv_path = args.csv_file
        xyz_path = args.xyz_root
    elif args.dataset == "hetro":
        # csv_path = r"C:\Users\12233\Downloads\7798697\PASs_csv\db-474K-OPV-filtered.csv"
        # xyz_path = r"C:\Users\12233\Downloads\7798697\PASs_xyz\db-474K-xyz"
        # csv_path = r'C:\Users\12233\OneDrive\项目一\PAS\PAS_generation\data\PAS_data\PAS_csv\dataset.csv'
        # xyz_path = r'C:\Users\12233\OneDrive\项目一\PAS\PAS_generation\data\PAS_data\PAS_xyz'
        # csv_path = r"/home/ssq233/项目/PAS/PAS_generation/data/PAS_data/PAS_csv/dataset-property.csv"
        # xyz_path = r"/home/ssq233/项目/PAS/PAS_generation/data/PAS_data/PAS_xyz"
        csv_path = r"D:\applications\les apps\OneDrive\项目一\PAS\PAS_generation\data\PAS_data\PAS_csv\dataset-property.csv"
        # csv_path = r"E:\data\PAS_data\PAS_csv/dataset.csv"
        xyz_path = r"E:\data\PAS_data\PAS_xyz"
    else:
        raise NotImplementedError
    return csv_path, xyz_path


def get_splits(args, random_seed=42, val_frac=0.1, test_frac=0.1):
    np.random.seed(seed=random_seed)
    csv_path, _ = get_paths(args)
    if hasattr(args, "dataset") and args.dataset == "hetro":
        targets = (
            args.target_features.split(",")
            if getattr(args, "target_features", None) is not None
            else []
        )
        df = pd.read_csv(csv_path, usecols=["name", "nRings", "inchi"] + targets)
        df.rename(columns={"nRings": "n_rings", "name": "molecule"}, inplace=True)
        args.max_nodes = min(args.max_nodes, 13)
    else:
        df = pd.read_csv(csv_path)

    df_all = df.copy()
    df_test = df.sample(frac=test_frac, random_state=random_seed)
    df = df.drop(df_test.index)
    df_val = df.sample(frac=val_frac, random_state=random_seed)
    df_train = df.drop(df_val.index)
    return df_train, df_val, df_test, df_all


def create_data_loaders(args):
    args.df_train, args.df_val, args.df_test, args.df_all = get_splits(args)

    train_dataset = AromaticDataset(
        args=args,
        task="train",
    )
    val_dataset = AromaticDataset(
        args=args,
        task="val",
    )
    test_dataset = AromaticDataset(
        args=args,
        task="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return train_loader, val_loader, test_loader
    # return train_loader
