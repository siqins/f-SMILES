
import os, sys
current_file_path = os.path.realpath(__file__)
parant_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parant_path)

from argparse import Namespace
from tqdm import tqdm

import torch
from rdkit import Chem
from rdkit.Chem import Draw

from data import xyz2mol
from data.aromatic_dataloader import (
    create_data_loaders,
    RINGS_LIST,
    ATOMS_LIST,
)
from data.ring import RINGS_DICT
from data.aromatic_dataloader import create_data_loaders, RINGS_LIST, get_splits
from utils.args_edm import Args_EDM
from utils.ring_graph import NO_ORIENTATION_RINGS, THREE_HETERO_RINGS_1, THREE_HETERO_RINGS_2
from utils.helpers import positions2adj
import matplotlib.pyplot as plt
import numpy as np

hexagon = np.array(
    [
        [6.92302547e-01, -1.19910074e00],
        [-6.92299212e-01, -1.19910016e00],
        [-1.38459997e00, -9.17922477e-07],
        [-6.92301879e-01, 1.19910117e00],
        [6.92298556e-01, 1.19910064e00],
        [1.3846, 0],
    ]
)
pentagon = np.array(
    [[0.3, -1.229], [-0.943, -0.743], [-0.943, 0.742], [0.3, 1.229], [1.286, 0]]
)
square = np.array(
    [
        [5.55111512e-17, 9.47523087e-01],
        [-9.47523087e-01, 5.55111512e-17],
        [-5.55111512e-17, -9.47523087e-01],
        [9.47523087e-01, -5.55111512e-17],
    ]
)
rings = {
    "Bn": hexagon,  # benzene
    "Pd": hexagon,  # pyridine
    "Pz": hexagon,  # pyrazine

    "Tr": pentagon,  # triazole
    "Td": pentagon,  # thiadiazole
    "Sed": pentagon,  # selenadiazole

    "Cyc": pentagon,  # cyclopentadiene
    "Pl": pentagon,  # pyrrole
    "Fu": pentagon,  # furan
    "Si": pentagon,  # silole
    "Th": pentagon,  # thiophene
    "Sel": pentagon,  # selenophene

}


def align_to_xy_plane(x):
    """
    Rotate the molecule into xy-plane.

    """
    I = np.zeros((3, 3))  # set up inertia tensor I
    com = np.zeros(3)  # set up center of mass com

    # calculate moment of inertia tensor I
    for i in range(x.shape[0]):
        atom = x[i]
        # y**2 + z**2, -xy, -xz
        # -xy, x**2 + z**2, -yz
        # -xz, -yz, x**2 + y**2
        I += np.array(
            [
                [(atom[1] ** 2 + atom[2] ** 2), -atom[0] * atom[1], -atom[0] * atom[2]],
                [-atom[0] * atom[1], (atom[0] ** 2 + atom[2] ** 2), -atom[1] * atom[2]],
                [-atom[0] * atom[2], -atom[1] * atom[2], atom[0] ** 2 + atom[1] ** 2],
            ]
        )

        com += atom
    com = com / len(com)

    # extract eigenvalues and eigenvectors for I
    # np.linalg.eigh(I)[0] are eigenValues, [1] are eigenVectors
    eigenVectors = np.linalg.eigh(I)[1]
    eigenVectorsTransposed = np.transpose(eigenVectors)

    a = []
    for i in range(x.shape[0]):
        xyz = x[i]
        a.append(np.dot(eigenVectorsTransposed, xyz - com))
    return np.stack(a)


def plot_goa(x, bonds=[], atom_types=None):
    plt.scatter(x[:, 0], x[:, 1])
    for b in bonds:
        plt.plot(x[b, 0], x[b, 1], c="b")
    if atom_types is not None:
        for i, a in enumerate(atom_types):
            plt.text(x[i, 0], x[i, 1], a)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def rotation_2d(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def lineseg_dists(p, a, b):
    # Handle case where p is a single point, i.e. 1d array.
    # p是经过调整，显示在xy平面的环所包含的原子的坐标
    # a和b是两个环的质心坐标，矩阵形状为【1 ，2】
    #  np.atleast_2d函数确保输出一个至少二维的数组，若输入已经大于等于二维，则不改变输入
    p = np.atleast_2d(p)

    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector，将两质心作差，然后用差值的二阶范数归一化差值b-a
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d[0])
    t = np.dot(p - b, d[0])

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)


def gor2goa(x, rings_types, dataset="cata", tol=0.1):
    if dataset == "cata":
        n = x.shape[0]
    else:
        n = x.shape[0] // 2

    # x[None, :n]会给矩阵x在0维添加维度，从N * 3 -》 1 * N * 3；adj是一【1*N*N】的矩阵，表示环与环之间的邻接情况，考量了环质心间距等因素
    _, adj = positions2adj(x[None, :n], rings_types[None, :n], dataset=dataset, tol=tol)
    adj = adj[0]

    x = align_to_xy_plane(x.numpy())[:, :2]
    # plot_goa(x)
    orientation = x[n:]
    x = x[:n]

    atoms = np.zeros([0, 2])
    atoms_types = []
    bonds = []
    rings_atoms_idxs = {}
    for i in range(x.shape[0]):
        ring_type = RINGS_LIST[dataset][rings_types[i]]
        ring = rings[ring_type].copy()
        if ring_type in NO_ORIENTATION_RINGS:
            # neighbor
            if adj.shape[0] == 1:
                angle = 0
            else:
                j = adj[i].nonzero()[0, 0]
                # 计算相邻环j的质心与该环i的质心的角度
                angle = np.arctan2(x[j, 1] - x[i, 1], x[j, 0] - x[i, 0])
            if ring_type == "Bn":
                angle += np.pi / 6
            elif ring_type == "Cbd":
                angle += np.pi / 4
            else:
                raise ValueError
        elif ring_type in THREE_HETERO_RINGS_1 + THREE_HETERO_RINGS_2:
            hetroatom_coord = orientation[i]
            angle = np.arctan2(hetroatom_coord[1] - x[i, 1], hetroatom_coord[0] - x[i, 0])
            angle -= 2 * np.pi / 5
        else:
            hetroatom_coord = orientation[i]
            angle = np.arctan2(hetroatom_coord[1] - x[i, 1], hetroatom_coord[0] - x[i, 0])
            # angle -= np.pi / 2

        # rotate ring to the correct orientation
        ring = ring @ rotation_2d(-angle)
        ring += x[i]
        rings_atoms_idxs[i] = list(
            range(atoms.shape[0], atoms.shape[0] + ring.shape[0])
        )
        atoms = np.concatenate([atoms, ring], axis=0)

        s_idx = atoms.shape[0] - ring.shape[0]
        for j in range(ring.shape[0] - 1):
            bonds.append([s_idx + j, s_idx + j + 1])
        bonds.append([s_idx + ring.shape[0] - 1, s_idx])

        atoms_types += RINGS_DICT[ring_type]

        # add H's，添加的氢原子的坐标是原点0，0
        if ring_type in ["Pl"]:
            atoms = np.concatenate([atoms, np.zeros([1, 2])], axis=0)
            atoms_types.append("H")
            bonds.append([s_idx + 4, atoms.shape[0] - 1])
        elif ring_type in ['Tr']:
            atoms = np.concatenate([atoms, np.zeros([1, 2])], axis=0)
            atoms_types.append("H")
            bonds.append([s_idx + 3, atoms.shape[0] - 1])
        elif ring_type in ["Cyc", "Si"]:
            atoms = np.concatenate([atoms, np.zeros([2, 2])], axis=0)
            atoms_types += ["H", "H"]
            bonds.append([s_idx + 4, atoms.shape[0] - 2])
            bonds.append([s_idx + 4, atoms.shape[0] - 1])

    # plot_goa(atoms, bonds, atoms_types)

    # remove duplicates，将环连接为一个整体
    # 去除临界矩阵相同的下半部分
    adj_u = np.triu(adj)
    if adj.shape[0] == 1:
        ring_bonds = []
    else:
        ring_bonds = list(zip(*adj_u.nonzero()))
    i_idxs, j_idxs = [], []
    for i, j in ring_bonds:
        # i_atoms 和 j_atoms是两个环i,j所包含的原子下纯编号，与原子序号不直接相关
        # i_coords 和 j_coords是经过调整，显示在xy平面的，i,j两个环所包含的原子的坐标
        i_atoms = rings_atoms_idxs[i]
        j_atoms = rings_atoms_idxs[j]
        i_coords = atoms[i_atoms]
        j_coords = atoms[j_atoms]

        # create line between rings centers and find the closest points
        # p1和p2是两个环的质心坐标，矩阵形状为【1 ，2】
        p1, p2 = x[i][None, :], x[j][None, :]
        di = lineseg_dists(i_coords, p1, p2)
        dj = lineseg_dists(j_coords, p1, p2)
        d_i = np.cross(p2 - p1, p1 - i_coords) / np.linalg.norm(p2 - p1)
        d_j = np.cross(p2 - p1, p1 - j_coords) / np.linalg.norm(p2 - p1)
        di2 = di.copy()
        dj2 = dj.copy()
        di[d_i > 0] = np.inf
        dj[d_j > 0] = np.inf
        di2[d_i < 0] = np.inf
        dj2[d_j < 0] = np.inf

        i_idxs += [i_atoms[di.argmin()], i_atoms[di2.argmin()]]
        j_idxs += [j_atoms[dj.argmin()], j_atoms[dj2.argmin()]]

    new_atoms = []
    new_atoms_type = []
    atoms_map = {}
    for i, j in zip(i_idxs, j_idxs):
        new_atoms.append((atoms[i] + atoms[j]) / 2)
        new_atoms_type.append(atoms_types[i])
        atoms_map[i] = len(new_atoms) + len(atoms) - 1
        atoms_map[j] = len(new_atoms) + len(atoms) - 1
        atoms[i] = 0
        atoms[j] = 0

    if len(new_atoms) > 0:
        atoms = np.concatenate([atoms, np.stack(new_atoms, axis=0)], axis=0)
    atoms_types = atoms_types + new_atoms_type
    atoms_types = [ATOMS_LIST[dataset].index(t) for t in atoms_types]
    bonds = [[atoms_map.get(i, i), atoms_map.get(j, j)] for i, j in bonds]

    idx_delete = i_idxs + j_idxs
    atoms = {i: a for i, a in enumerate(atoms) if i not in idx_delete}
    atoms_types = {i: a for i, a in enumerate(atoms_types) if i not in idx_delete}
    idx = list(atoms.keys())
    bonds = [[idx.index(i), idx.index(j)] for i, j in bonds]
    atoms = np.stack(list(atoms.values()), axis=0)
    atoms_types = list(atoms_types.values())

    # remove duplicate bonds
    bonds = [tuple(sorted(a)) for a in bonds]
    bonds = list(set(bonds))
    # plot_goa(atoms, bonds, [ATOMS_LIST[dataset][t] for t in atoms_types])

    return torch.tensor(atoms), torch.tensor(atoms_types), bonds


def smiles2inchi(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(m)


def draw_mol(mol, title=""):
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.title(title)
    plt.show()


def check_for_single_bonds(bond, bonds, atom_types, dataset):
    # 根据bond的两个原子，查找对应的原子是否含两个氢，如果有两个氢，说明为环戊二烯，连接单键
    bond_i, bond_j = bond
    # bond_i_in_bonds = [a for b in bonds if bond_i in b for a in b if ATOMS_LIST[dataset][atom_types[a].item()] not in ['C', 'Si']]
    # bond_j_in_bonds = [a for b in bonds if bond_j in b for a in b if ATOMS_LIST[dataset][atom_types[a].item()] not in ['C', 'Si']]
    bond_i_in_bonds = [a for b in bonds if bond_i in b for a in b]
    bond_j_in_bonds = [a for b in bonds if bond_j in b for a in b]

    bond_i_in_bonds = [ATOMS_LIST[dataset][atom_types[a]] for a in bond_i_in_bonds]
    bond_j_in_bonds = [ATOMS_LIST[dataset][atom_types[b]] for b in bond_j_in_bonds]
    bond_i_in_bonds_numHs = bond_i_in_bonds.count("H")
    bond_j_in_bonds_numHs = bond_j_in_bonds.count("H")

    if bond_i_in_bonds_numHs == 2 or bond_j_in_bonds_numHs == 2:
        return True
    else:
        return False


def build_molecule_aromatic(atom_types, bonds, dataset):
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(ATOMS_LIST[dataset][atom.item()])
        mol.AddAtom(a)

    for bond in bonds:
        # bond_i = ATOMS_LIST[dataset][atom_types[bond[0]].item()]
        # bond_j = ATOMS_LIST[dataset][atom_types[bond[1]].item()]
        # # bond_i = atom_types[bond[0]]
        # # bond_j = atom_types[bond[1]]
        # if bond_i == "H" or bond_j == "H":
        #     mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.SINGLE)
        # # elif bond_i == "C" and bond_j == "C":
        # #     if check_for_single_bonds(bond, bonds, atom_types, dataset):
        # #         mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.SINGLE)
        # #     else:
        # #         mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.AROMATIC)
        # # elif bond_i == "Si" or bond_j == "Si":
        # #     if check_for_single_bonds(bond, bonds, atom_types, dataset):
        # #         mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.SINGLE)
        # #     else:
        # #         mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.AROMATIC)
        # else:
        #     mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.AROMATIC)

        if atom_types[bond[0]] == "H" or atom_types[bond[1]] == "H":
            mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.SINGLE)
        else:
            mol.AddBond(bond[0], bond[1], Chem.rdchem.BondType.AROMATIC)

    atoms = list(mol.GetAtoms())
    for atom in atoms:
        if atom.GetDegree() == 2 and atom.GetSymbol() in ["C", "Si"]:
            h = Chem.Atom("H")
            h_idx = mol.AddAtom(h)
            mol.AddBond(atom.GetIdx(), h_idx, Chem.rdchem.BondType.SINGLE)

    # Chem.SanitizeMol(mol)

    Draw.MolToFile(mol, 'o.png')

    return mol


def rdkit_valid(atoms_types, bonds, dataset="cata"):
    valid = []
    for a_types, b in zip(atoms_types, bonds):
        mol = build_molecule_aromatic(a_types, b, dataset)
        sm = Chem.MolToSmiles(mol)
        AC = Chem.GetAdjacencyMatrix(mol)
        atoms = [xyz2mol.int_atom(atom.GetSymbol()) for atom in mol.GetAtoms()]
        rwmol = Chem.RWMol()
        for atom in mol.GetAtoms():
            a = Chem.Atom(atom.GetSymbol())
            rwmol.AddAtom(a)

        is_valid = False
        try:
            mol = xyz2mol.AC2mol(rwmol, AC, atoms, 0)
            sm = Chem.MolToSmiles(mol[0])
            Draw.MolToFile(mol[0], 'o1.png')
            if len(mol) == 1:
                mol = mol[0]
                Chem.SanitizeMol(mol)
                if len(Chem.GetMolFrags(mol, asMols=True)) == 1:
                    is_valid = True
        except:
            pass

        if is_valid:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            valid.append(smiles2inchi(smiles))

    return valid, len(valid) / len(atoms_types)


def analyze_rdkit_validity_for_molecules(
    molecule_list,
    tol=0.1,
    dataset="cata",
    calc_novelty=False
):
    n_samples = len(molecule_list)
    molecule_valid_list = []
    molecule_valid_bool = []
    valid_inchi = []

    with tqdm(molecule_list, unit="mol") as tq:
        for i, (x, rings_type) in enumerate(tq):
            try:
                atoms, atoms_types, bonds = gor2goa(
                    x,
                    rings_type,
                    tol=tol,
                    dataset=dataset,
                )
                valid, val_ration = rdkit_valid([atoms_types], [bonds], dataset)
                molecule_valid = len(valid) > 0
            except:
                molecule_valid = False

            molecule_valid_bool.append(molecule_valid)
            if molecule_valid:
                molecule_valid_list.append((x, rings_type))
                valid_inchi += valid

    unique = set(valid_inchi)

    validity_dict = {
        "mol_valid": len(valid_inchi) / float(n_samples),
        "mol_unique": len(unique) / max(len(valid_inchi), 1),
        "molecule_valid_bool": molecule_valid_bool,
        "valid_inchi": valid_inchi,
    }
    if calc_novelty:
        df_train = get_splits(
            Namespace(dataset=dataset, target_features=None, max_nodes=11)
        )[0]
        if "inchi" in df_train.columns:
            train_inchi = df_train["inchi"].tolist()
        else:
            train_smiles = df_train["smiles"].tolist()
            train_inchi = [smiles2inchi(s) for s in train_smiles]

        novel = set(valid_inchi) - set(train_inchi)
        validity_dict["mol_novel"] = len(novel) / max(len(valid_inchi), 1)

    return validity_dict, molecule_valid_list


if __name__ == "__main__":
    args = Args_EDM().parse_args()
    args.device = "cpu"
    args.tol = 0.3
    args.sample_rate = 1
    # args.batch_size = 2
    # args.num_workers = 0
    # args.dataset = "cata"
    # args.orientation = False
    # args.target_features = "GAP_eV"
    train_loader, val_loader, test_loader = create_data_loaders(args)

    np.random.seed(13)
    for i in np.random.randint(0, len(train_loader.dataset), 10):
        x, node_mask, edge_mask, node_features, y = train_loader.dataset[i]
        node_mask = node_mask.bool()
        # atoms_positions: torch格式，【原子数 * 2】的矩阵，表示在xy平面上的坐标
        # atoms_types： torch格式，表示原子种类，【原子数】的矩阵
        # bonds：列表格式，[[1,2],[2,3]...]形式，表示原子之间的键接
        atoms_positions, atoms_types, bonds = gor2goa(
            x[node_mask], node_features[node_mask].argmax(1), args.dataset
        )
        # np.save('GOA-with-Hs.npy', [atoms_positions, atoms_types, bonds])
        valid, val_ration = rdkit_valid([atoms_types], [bonds], args.dataset)
        print(val_ration)
        # print(valid)

    molecule_list = []
    loader = test_loader
    for x, node_mask, edge_mask, h, y in loader:
        node_mask = node_mask.bool()
        molecule_list += [
            (x[i, node_mask[i]], h[i, node_mask[i]].argmax(1))
            for i in range(x.shape[0])
        ]

    stability_dict, molecule_stable_list = analyze_rdkit_validity_for_molecules(molecule_list, args.tol, args.dataset)
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass
