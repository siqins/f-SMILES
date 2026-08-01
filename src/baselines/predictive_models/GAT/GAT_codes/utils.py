import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

np.random.seed(0)

from torch_geometric.data import Data
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, graph_list, label_list=None, device=None, task='regression'):
        self.labels = label_list
        self.graphs = graph_list
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.task = task

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        label = self.labels[index] if self.labels is not None else 0
        if self.task == 'classification':
            label = torch.tensor(label, dtype=torch.long).to(self.device)
        elif self.task == 'regression':
            label = torch.tensor(label, dtype=torch.float).to(self.device)
        graph = self.graphs[index].to(self.device)
        return graph, label


def one_of_k_encoding(x, allowable_set):
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
    possible_atoms = ['P', 'S', 'Zn', '*', 'Al', 'C', 'O', 'Cl', 'Si', 'B', 'F', 'N', 'Se', 'Br', 'Ge', 'I']
    # possible_atoms = ['C', 'N', 'O', 'Si', 'P', 'S', 'Br']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)  # 10
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3])  # 4
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])  # 2
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])  # 7
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])  # 3
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])  # 5
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)

def get_graph_from_smile(molecule_smile):
    """
    Method that constructs a molecular graph with nodes being the atoms
    and bonds being the edges.
    :param molecule_smile: SMILE sequence
    :return: DGL graph object, Node features and Edge features
    """

    molecule = Chem.MolFromSmiles(molecule_smile)
    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        # Use get_atom_features_mnsol if you are building dataset related to solvation free energies
        # atom_i_features = get_atom_features_mnsol(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(bonds, dtype=torch.long).T
    edge_feats = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_feats)
