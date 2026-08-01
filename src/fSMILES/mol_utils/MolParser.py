import os
import sys
import inspect
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from mol_utils.Patterns import (uni_patterns,
                               di_patterns_5rings_starter, di_patterns_6rings_starter,
                               di_patterns_5rings, di_patterns_6rings,
                               tri_patterns_5rings, tri_patterns_6rings, tri_patterns_benzene,
                               tri_patterns_branch_5rings_check, tri_patterns_branch_6rings_check,
                               quad_patterns_5rings, quad_patterns_6rings, quad_patterns_benzene,
                               quad_patterns_branch_5rings_check, quad_patterns_branch_6rings_check,
                               penta_patterns_benzene, penta_patterns_6rings, penta_patterns_5rings,
                               multi_patterns)
from mol_utils.utils import are_tuples_connected, find_outermost_braces, concat, get_products
from mol_utils.Reactions import ReactionManager
sys.setrecursionlimit(5000)


class FSmilesTokenizer:

    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def tokenize(smiles: str):
        n = len(smiles)
        mark_left = 0
        mark_right = 0
        tokens = []
        for i in range(n):
            if smiles[i] == ']' and smiles[i-1].isdigit():
                mark_right = i
                tokens.append(smiles[mark_left: mark_right + 1])
                mark_left = mark_right + 1
            elif smiles[i] == '{' or smiles[i] == '}':
                tokens.append(smiles[i])
                mark_left = i + 1
        return tokens


    def process(self, smiles):
        if isinstance(smiles, list):
            if self.verbose:
                print('Tokenizing------------------------------')
                loop = tqdm(smiles)
            else:
                loop = smiles
            tokens = []
            for smi in loop:
                token = self.tokenize(smi)
                tokens.append(token)
            return tokens
        else:
            return self.tokenize(smiles)

    def __call__(self, smiles):
        return self.process(smiles)


def get_ring_smiles(mol, ring_atom_indices):
    emol = Chem.EditableMol(Chem.Mol())

    atom_map = {}
    atom_symbol = {}
    atom_total_hs = {}
    for idx in ring_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = emol.AddAtom(new_atom)
        atom_map[idx] = new_idx

        atom_symbol[idx] = atom.GetSymbol()
        atom_total_hs[idx] = atom.GetTotalNumHs()

    for i in range(len(ring_atom_indices)):
        idx1 = ring_atom_indices[i]
        for j in range(i, len(ring_atom_indices)):
            idx2 = ring_atom_indices[j]
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond:
                if atom_symbol[idx1] == 'C' and atom_symbol[idx2] == 'C' and atom_total_hs[idx1] == 0 and atom_total_hs[idx2] == 0:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.AROMATIC)
                elif atom_symbol[idx1] == 'C' and atom_symbol[idx2] == 'C' and atom_total_hs[idx1] == 1 and atom_total_hs[idx2] == 0:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.AROMATIC)
                elif atom_symbol[idx1] == 'C' and atom_symbol[idx2] == 'C' and atom_total_hs[idx1] == 0 and atom_total_hs[idx2] == 1:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.AROMATIC)
                elif atom_symbol[idx1] == 'C' and atom_symbol[idx2] == 'C' and atom_total_hs[idx1] == 1 and atom_total_hs[idx2] == 1:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.AROMATIC)
                else:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    ring_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(ring_mol)
    except:
        smiles = Chem.MolToSmiles(ring_mol).upper()
        ring_mol = Chem.MolFromSmiles(smiles)

    ring_smiles = Chem.MolToSmiles(ring_mol)
    return ring_smiles


def get_ring_smiles_extra_markers(mol, ring_atom_indices, marker_atoms_indices):
    emol = Chem.EditableMol(Chem.Mol())

    atom_map = {}
    for idx in ring_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if idx in marker_atoms_indices:
            new_atom = Chem.Atom('O')
        else:
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = emol.AddAtom(new_atom)
        atom_map[idx] = new_idx

    for i in range(len(ring_atom_indices)):
        idx1 = ring_atom_indices[i]
        for j in range(i, len(ring_atom_indices)):
            idx2 = ring_atom_indices[j]
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond:
                if idx1 in marker_atoms_indices or idx2 in marker_atoms_indices:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.SINGLE)
                else:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    ring_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(ring_mol)
    except:
        smiles = Chem.MolToSmiles(ring_mol).upper()
        ring_mol = Chem.MolFromSmiles(smiles)

    return Chem.MolToSmiles(ring_mol)


def get_ring_smiles_extra_markers_for_c(mol, ring_atom_indices, marker_atoms_indices):
    emol = Chem.EditableMol(Chem.Mol())

    atom_map = {}
    for idx in ring_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if idx in marker_atoms_indices and atom.GetSymbol() == 'C' and atom.GetIsAromatic():
            new_atom = Chem.Atom('O')
        else:
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = emol.AddAtom(new_atom)
        atom_map[idx] = new_idx

    for i in range(len(ring_atom_indices)):
        idx1 = ring_atom_indices[i]
        for j in range(i, len(ring_atom_indices)):
            idx2 = ring_atom_indices[j]
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond:
                if idx1 in marker_atoms_indices or idx2 in marker_atoms_indices:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.SINGLE)
                else:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    ring_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(ring_mol)
    except:
        smiles = Chem.MolToSmiles(ring_mol).upper()
        ring_mol = Chem.MolFromSmiles(smiles)

    return Chem.MolToSmiles(ring_mol)


def get_ring_smiles_extra_markers_for_branches(mol, ring_atom_indices, marker_atoms_indices):
    emol = Chem.EditableMol(Chem.Mol())

    atom_map = {}
    for idx in ring_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if idx in marker_atoms_indices:
            if atom.GetSymbol() != 'C' or atom.GetTotalNumHs() ==2:
                new_atom = Chem.Atom(atom.GetSymbol())
            else:
                new_atom = Chem.Atom('O')
        else:
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = emol.AddAtom(new_atom)
        atom_map[idx] = new_idx

    for i in range(len(ring_atom_indices)):
        idx1 = ring_atom_indices[i]
        for j in range(i, len(ring_atom_indices)):
            idx2 = ring_atom_indices[j]
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond:
                if idx1 in marker_atoms_indices or idx2 in marker_atoms_indices:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.SINGLE)
                else:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    ring_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(ring_mol)
    except:
        smiles = Chem.MolToSmiles(ring_mol).upper()
        ring_mol = Chem.MolFromSmiles(smiles)

    return Chem.MolToSmiles(ring_mol)


def get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices):
    emol = Chem.EditableMol(Chem.Mol())

    atom_map = {}
    marked_atoms = [mol.GetAtomWithIdx(idx) for idx in marker_atoms_indices]
    marked_atoms_symbols = [atom.GetSymbol() for atom in marked_atoms]

    marked_atoms_atomicnums = [atom.GetAtomicNum() for atom in marked_atoms]
    marked_atoms_atomicnums_types = list(set(marked_atoms_atomicnums))
    if len(marked_atoms_atomicnums_types) > 2:
        max_atom_atomicnum = max(marked_atoms_atomicnums_types)
    else:
        max_atom_atomicnum = 6

    all_carbon_check = all([atom_symbol == 'C' for atom_symbol in marked_atoms_symbols])

    for idx in ring_atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if idx in marker_atoms_indices:
            if atom.GetSymbol() != 'C':
                if len(marked_atoms_atomicnums_types) > 2:
                    if atom.GetAtomicNum() == max_atom_atomicnum:
                        new_atom = Chem.Atom('Te')
                    else:
                        new_atom = Chem.Atom('C')
                else:
                    new_atom = Chem.Atom('Te')
            elif atom.GetTotalNumHs() == 2:
                if all_carbon_check:
                    new_atom = Chem.Atom('Te')
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() != 'C':
                            new_atom = Chem.Atom(atom.GetSymbol())
                            break
                else:
                    new_atom = Chem.Atom(atom.GetSymbol())
            else:
                new_atom = Chem.Atom(atom.GetSymbol())
        else:
            new_atom = Chem.Atom(atom.GetSymbol())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = emol.AddAtom(new_atom)
        atom_map[idx] = new_idx

    for i in range(len(ring_atom_indices)):
        idx1 = ring_atom_indices[i]
        for j in range(i, len(ring_atom_indices)):
            idx2 = ring_atom_indices[j]
            bond = mol.GetBondBetweenAtoms(idx1, idx2)
            if bond:
                if idx1 in marker_atoms_indices or idx2 in marker_atoms_indices:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], Chem.BondType.SINGLE)
                else:
                    emol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())

    ring_mol = emol.GetMol()
    try:
        Chem.SanitizeMol(ring_mol)
    except:
        smiles = Chem.MolToSmiles(ring_mol).upper()
        ring_mol = Chem.MolFromSmiles(smiles)

    return Chem.MolToSmiles(ring_mol)


def molecule_to_ring_graph(mol):
    ring_info = mol.GetRingInfo()
    rings = [frozenset(ring_atoms) for ring_atoms in ring_info.AtomRings()]
    frags = [get_ring_smiles(mol, ring_atoms) for ring_atoms in ring_info.AtomRings()]

    G = nx.Graph()

    for idx, ring in enumerate(rings):
        G.add_node(tuple(sorted(ring)), atom_symbol=frags[idx])

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared_atoms = rings[i] & rings[j]
            if shared_atoms:
                G.add_edge(tuple(sorted(rings[i])), tuple(sorted(rings[j])),
                           weight=len(shared_atoms))

    return G


def visualize_ring_graph(ring_graph):
    labels = {node: f"Ring_{i}" for i, node in enumerate(ring_graph.nodes())}
    pos = nx.spring_layout(ring_graph)

    nx.draw(ring_graph, pos, labels=labels, with_labels=True, node_size=1000, node_color='skyblue')
    plt.show()


def dfs_ring_order(ring_graph, mol, start_node=None):
    def select_start_node(ring_degrees, mol):
        small_rings = [ring for ring, degree in ring_degrees if len(ring) <= 6]
        if small_rings:
            min_degree = min(degree for ring, degree in ring_degrees if len(ring) <= 6)
            candidate_rings = [ring for ring, degree in ring_degrees if len(ring) <= 6 and degree == min_degree]

            ring_smiles_heteratoms = [count_heteroatoms(mol, ring_atoms) for ring_atoms in candidate_rings]
            ring_smiles_atoms = [len(ring_atoms) for ring_atoms in candidate_rings]

            markers = [str(n_atoms) + '-' + str(n_hetero) for n_atoms, n_hetero in zip(ring_smiles_atoms, ring_smiles_heteratoms)]

            if ring_smiles_heteratoms.count(2) > 0 or ring_smiles_heteratoms.count(3) > 0:
                for special_marker in ['5-1', '5-0', '6-1', '6-0']:
                    if special_marker in markers:
                        idx = markers.index(special_marker)
                        candidate_rings = [candidate_rings[idx]]
                        break

        else:
            min_degree = min(degree for ring, degree in ring_degrees)
            candidate_rings = [ring for ring, degree in ring_degrees if degree == min_degree]

        return min(candidate_rings, key=lambda x: (len(x), min(x)))

    def count_heteroatoms(mol, ring_atoms):
        heteroatoms = 0
        for atom_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != 'C':
                heteroatoms += 1
        return heteroatoms

    def select_neighbor_node(mol, current_node, neighbors):
        scored_neighbors = []
        for neighbor in neighbors:
            ring_size = len(neighbor)
            heteroatoms = count_heteroatoms(mol, neighbor)
            degree = ring_graph.degree(neighbor)

            is_cyclopentadiene = False
            if ring_size == 5:
                ring_smiles = get_ring_smiles(mol, neighbor)
                if ring_smiles == 'C1=CCC=C1':
                    is_cyclopentadiene = True

            if ring_size == 5 and (heteroatoms == 1 or is_cyclopentadiene):
                score = (0, -degree, min(neighbor))
            elif ring_size == 6 and heteroatoms == 1:
                score = (1, -degree, min(neighbor))
            elif ring_size == 6 and heteroatoms == 2:
                score = (2, -degree, min(neighbor))
            elif ring_size == 6 and heteroatoms == 3:
                score = (3, -degree, min(neighbor))
            elif ring_size == 6 and heteroatoms == 4:
                score = (4, -degree, min(neighbor))
            elif ring_size == 6:
                score = (5, -degree, min(neighbor))
            elif ring_size == 5 and heteroatoms > 1:
                score = (6, -degree, min(neighbor))
            else:
                score = (7, -degree, min(neighbor))
            scored_neighbors.append((score, neighbor))
        scored_neighbors.sort(key=lambda x: x[0])
        return [neighbor for score, neighbor in scored_neighbors]

    if start_node is None:
        start_node = select_start_node(ring_graph.degree(), mol)

    visited_order = []

    def _dfs(node):
        visited_order.append(node)

        neighbors = list(ring_graph.neighbors(node))

        unvisited_neighbors = [n for n in neighbors if n not in visited_order]

        if not unvisited_neighbors:
            return

        neighbors = select_neighbor_node(mol, node, unvisited_neighbors)

        for neighbor in neighbors:
            if neighbor not in visited_order:
                _dfs(neighbor)

    _dfs(start_node)
    return visited_order


def check_adjacent_shared_values_and_rings(connections, ring_order):
    keys = sorted(connections.keys())
    problematic_pairs = []

    for i in range(len(keys) - 1):
        k1, k2 = keys[i], keys[i + 1]
        shared = set(connections[k1]) & set(connections[k2])
        if len(shared) > 1:
            problematic_pairs.append((k1, k2, shared))

    if problematic_pairs:
        print('Unsupported for the peri-condensed polycyclic aromatic systems (PASs).')
        return False
    for k,v in connections.items():
        if len(v) > 3:
            print('Unsupported for the peri-condensed polycyclic aromatic systems (PASs).')
            return False

    for idx, (v) in enumerate(ring_order):
        if len(v) > 6:
            print('Unsupported for the huge ring (>6).')
            return False
        if len(v) == 5:
            if len(connections[idx]) > 2:
                print('Unsupported for the five-membered ring connected with too many rings (>2).')
                return False

    def has_3_node_cycle(graph):
        visited = set()
        for node in graph:
            if node not in visited:
                if dfs_detect_3_cycle(graph, node, visited):
                    return True
        return False

    def dfs_detect_3_cycle(graph, start, visited, path=None):
        if path is None:
            path = []

        if start in path:
            cycle_length = len(path) - path.index(start)
            if cycle_length == 3:
                cycle = path[path.index(start):] + [start]
                return True
            return False
        visited.add(start)
        path.append(start)
        for neighbor in graph[start]:
            if dfs_detect_3_cycle(graph, neighbor, visited, path):
                return True
        path.pop()
        return False

    if has_3_node_cycle(connections):
        print('Unsupported for the peri-condensed polycyclic aromatic systems (PASs).')
        return False

    return True


def get_connections(ring_order):
    connections = defaultdict(list)

    for i, ring_atoms in enumerate(ring_order):
        for j in range(i):
            if set(ring_order[j]) & set(ring_atoms):
                connections[j].append(i)
                connections[i].append(j)
    return connections


def rings_to_fsmiles(mol, ring_order, frags_dict):
    """
    the ordered rings to fsmiles

    parameters:
        mol: RDKit mol
        ring_order: a list of ring node with the order of DFS

    return:
        str: fsmiles
    """

    connections = defaultdict(list)
    visited = set()
    output_first_mark = []

    for i, ring_atoms in enumerate(ring_order):
        for j in range(i):
            if set(ring_order[j]) & set(ring_atoms):
                connections[j].append(i)
                connections[i].append(j)

    branch_rings = [i for i in range(len(ring_order)) if len(connections[i]) > 2]

    if not check_adjacent_shared_values_and_rings(connections, ring_order):
        return None

    def dfs(current_ring, parent=None, start_branch=False):
        visited.add(current_ring)
        output = []

        ring_smiles = frags_dict[ring_order[current_ring]]
        is_6_ring = len(ring_order[current_ring]) == 6

        if is_6_ring:
            connected = connections[current_ring]
            if len(connected) >= 2:
                if len(connected) == 2:
                    shared1 = set(ring_order[current_ring]) & set(ring_order[connected[0]])
                    shared1 = tuple(sorted(shared1))
                    shared2 = set(ring_order[current_ring]) & set(ring_order[connected[1]])
                    shared2 = tuple(sorted(shared2))
                    if any([are_tuples_connected(ring_order[current_ring], tuple([i]), tuple([j])) for i in shared1 for j in shared2]):
                        pattern = '1'
                    else:
                        pattern = '2'
                else:
                    pattern = '3'
            else:
                pattern = ''
            output.append(pattern)

            is_benzene = ring_smiles == 'c1ccccc1'
            if is_benzene:
                logger = []
                for j in range(current_ring):
                    for x in range(j):

                        for y in range(x):

                            for z in range(y):

                                if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]) and set(ring_order[x]) & set(ring_order[y]) and set(ring_order[y]) & set(ring_order[z]):
                                    marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]))
                                    ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]) | set(ring_order[current_ring]))
                                    sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                    m_te = Chem.MolFromSmiles(sm_te)
                                    for k, v in penta_patterns_benzene.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger.append(k.split('-')[-1])
                                            break
                                    break

                            if len(logger) == 0:
                                if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]) and set(ring_order[x]) & set(ring_order[y]):
                                    marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]))
                                    ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[current_ring]))
                                    sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                    m_te = Chem.MolFromSmiles(sm_te)
                                    for k, v in quad_patterns_benzene.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger.append(k.split('-')[-1])
                                            break
                                    break

                        if len(logger) == 0:
                            if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]):  # 如果有共享原子
                                sm = get_ring_smiles(mol, tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[current_ring])))
                                m = Chem.MolFromSmiles(sm)
                                for k, v in tri_patterns_benzene.items():
                                    if m.HasSubstructMatch(v):
                                        logger.append(k.split('-')[-1])
                                        break
                                break
                if len(logger) != 0:
                    output.append(logger[0])
            else:
                logger = []
                if len(ring_order) > 1 and parent is None:
                    marker_atoms_indices = (set(ring_order[current_ring + 1]) | set(ring_order[current_ring])) - set(ring_order[current_ring])
                    sm = get_ring_smiles_extra_markers_for_c(mol, tuple(set(ring_order[current_ring + 1]) | set(ring_order[current_ring])),
                                                             marker_atoms_indices=marker_atoms_indices)
                    m = Chem.MolFromSmiles(sm)
                    for k, v in di_patterns_6rings_starter.items():
                        if m.HasSubstructMatch(v):
                            output.append(k.split('-')[-1])
                            break
                for j in range(current_ring):
                    for x in range(j):
                        for y in range(x):
                            for z in range(y):
                                if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]) and set(ring_order[x]) & set(ring_order[y]) and set(ring_order[y]) & set(ring_order[z]):
                                    marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]))
                                    ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]) | set(ring_order[current_ring]))
                                    sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                    m_te = Chem.MolFromSmiles(sm_te)
                                    for k, v in penta_patterns_6rings.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger.append(k.split('-')[-1])
                                            break
                                    break
                            if len(logger) == 0:
                                if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]) and set(ring_order[x]) & set(ring_order[y]):
                                    marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]))
                                    ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[current_ring]))
                                    sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                    m_te = Chem.MolFromSmiles(sm_te)
                                    logger_marker = None
                                    if start_branch:
                                        for k, v in quad_patterns_branch_6rings_check.items():
                                            if m_te.HasSubstructMatch(v):
                                                logger_marker = k.split('-')[-1]
                                                break
                                    if logger_marker is None:
                                        for k, v in quad_patterns_6rings.items():
                                            if m_te.HasSubstructMatch(v):
                                                logger_marker = k.split('-')[-1]
                                                break
                                    if logger_marker is not None:
                                        logger.append(logger_marker)
                                    break
                        if len(logger) == 0:
                            if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]):
                                marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]))
                                ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[current_ring]))
                                sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                m_te = Chem.MolFromSmiles(sm_te)
                                logger_marker = None
                                if start_branch:
                                    for k, v in tri_patterns_branch_6rings_check.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger_marker = k.split('-')[-1]
                                            break
                                if logger_marker is None:
                                    for k, v in tri_patterns_6rings.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger_marker = k.split('-')[-1]
                                            break
                                if logger_marker is not None:
                                    logger.append(logger_marker)
                                break
                    if len(logger) == 0:
                        if set(ring_order[j]) & set(ring_order[current_ring]):
                            marker_atom_indices = (set(ring_order[j]) | set(ring_order[current_ring])) - (set(ring_order[j]))
                            ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[current_ring]))
                            sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                            m_te = Chem.MolFromSmiles(sm_te)
                            for k, v in di_patterns_6rings.items():
                                if m_te.HasSubstructMatch(v):
                                    logger.append(k.split('-')[-1])
                                    break
                            break
                if len(logger) > 0:
                    output.append(logger[0])
            output.append(f"[{ring_smiles}]")
        else:
            logger = []
            if len(ring_order) > 1 and parent is None:
                marker_atoms_indices = (set(ring_order[current_ring + 1]) | set(ring_order[current_ring])) - set(ring_order[current_ring])
                sm = get_ring_smiles_extra_markers_for_c(mol, tuple(set(ring_order[current_ring + 1]) | set(ring_order[current_ring])),
                                                         marker_atoms_indices=marker_atoms_indices)
                m = Chem.MolFromSmiles(sm)
                for k, v in di_patterns_5rings_starter.items():
                    if m.HasSubstructMatch(v):
                        output.append(k.split('-')[-1])
                        break
            for j in range(current_ring):
                for x in range(j):
                    for y in range(x):
                        for z in range(y):
                            if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]) and set(ring_order[x]) & set(ring_order[y]) and set(ring_order[y]) & set(ring_order[z]):
                                marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]))
                                ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[z]) | set(ring_order[current_ring]))
                                sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                m_te = Chem.MolFromSmiles(sm_te)
                                for k, v in penta_patterns_5rings.items():
                                    if m_te.HasSubstructMatch(v):
                                        logger.append(k.split('-')[-1])
                                        break
                                break
                        if len(logger) == 0:
                            if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]) and set(ring_order[x]) & set(ring_order[y]):  # 如果有共享原子

                                marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]))
                                ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x]) | set(ring_order[y]) | set(ring_order[current_ring]))
                                sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                                m_te = Chem.MolFromSmiles(sm_te)
                                logger_marker = None
                                if start_branch:
                                    for k, v in quad_patterns_branch_5rings_check.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger_marker = k.split('-')[-1]
                                            break
                                if logger_marker is None:
                                    for k, v in quad_patterns_5rings.items():
                                        if m_te.HasSubstructMatch(v):
                                            logger_marker = k.split('-')[-1]
                                            break
                                if logger_marker is not None:
                                    logger.append(logger_marker)
                                break
                    if len(logger) == 0:
                        if set(ring_order[j]) & set(ring_order[current_ring]) and set(ring_order[j]) & set(ring_order[x]):
                            marker_atom_indices = (set(ring_order[j]) | set(ring_order[x]) | set(ring_order[current_ring])) - (set(ring_order[j]) | set(ring_order[x]))
                            ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[x])| set(ring_order[current_ring]))
                            sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                            m_te = Chem.MolFromSmiles(sm_te)
                            logger_marker = None
                            if start_branch:
                                for k, v in tri_patterns_branch_5rings_check.items():
                                    if m_te.HasSubstructMatch(v):
                                        logger_marker = k.split('-')[-1]
                                        break
                            else:
                                for k, v in tri_patterns_5rings.items():
                                    if m_te.HasSubstructMatch(v):
                                        logger_marker = k.split('-')[-1]
                                        break
                            if logger_marker is not None:
                                logger.append(logger_marker)
                            break

                if len(logger) == 0:
                    if set(ring_order[j]) & set(ring_order[current_ring]):
                        marker_atom_indices = (set(ring_order[j]) | set(ring_order[current_ring])) - set(ring_order[j])
                        ring_atom_indices = tuple(set(ring_order[j]) | set(ring_order[current_ring]))
                        sm_te = get_ring_smiles_extra_markers_for_quad_rings(mol, ring_atom_indices, marker_atoms_indices=marker_atom_indices)
                        m_te = Chem.MolFromSmiles(sm_te)
                        for k, v in di_patterns_5rings.items():
                            if m_te.HasSubstructMatch(v):
                                logger.append(k.split('-')[-1])
                                break
                        ring_atoms_list = []
                        while len(ring_order[j]) == 6:
                            ring_atoms_list.extend(ring_order[j])
                            j -= 1
                        ring_atoms_list.extend(ring_order[j])
                        sm = get_ring_smiles(mol, tuple(set(ring_atoms_list) | set(ring_order[current_ring])))
                        m = Chem.MolFromSmiles(sm)
                        for k, v in multi_patterns.items():
                            if m.HasSubstructMatch(v):
                                logger.append(k.split('-')[-1])
                                break
                        break

            if len(logger) != 0:
                output.append(logger[0])
            output.append(f"[{ring_smiles}]")
        children = [r for r in connections[current_ring] if r != parent]
        children_sorted = sorted(children, reverse=True)
        children_in_branch_ring = [child in branch_rings for child in children_sorted]
        if any(children_in_branch_ring) and len(children_in_branch_ring) >= 2:
            children_in_branch_ring_index = children_in_branch_ring.index(True)
            children_in_branch_entity = children_sorted[children_in_branch_ring_index]
            children_sorted.remove(children_in_branch_entity)
            children_sorted = children_sorted + [children_in_branch_entity]
        for i, child in enumerate(children_sorted):
            if child not in branch_rings and i < len(children_sorted) - 1:
                output.append("{")
                branch = dfs(child, current_ring, start_branch=True)
                if branch.startswith('/~') or branch.startswith('\\~'):
                    branch = branch[1:]
                elif branch.startswith('\\\\[c1ccccc1]') or branch.startswith('\\/[c1ccccc1]') or branch.startswith('/\\[c1ccccc1]') or branch.startswith('//[c1ccccc1]'):
                    branch = branch[2:]
                output.append(branch)
                output.append("}")
            elif start_branch and '/' not in output and '\\' not in output:
                output.append(dfs(child, current_ring, start_branch=True))
            else:
                output.append(dfs(child, current_ring))
        return "".join(output)

    fsmiles = dfs(0)
    if output_first_mark:
        fsmiles = '~' + fsmiles
    fsmiles = postprocess_fsmiles(fsmiles)
    return fsmiles


def postprocess_fsmiles(fsmiles):

    for token in ['{/\[c1cn[se]n1]}', '{/\[c1cn[nH]n1]}', '{/\[c1cnsn1]}', '{/\[c1cnon1]}']:
        replace_token = token.replace('/\\', '~')
        fsmiles = fsmiles.replace(token, replace_token)

    for token in ['{/[c1cn[se]n1]}', '{/[c1cn[nH]n1]}', '{/[c1cnsn1]}', '{/[c1cnon1]}']:
        replace_token = token.replace('/', '~')
        fsmiles = fsmiles.replace(token, replace_token)

    for token in ['{\\[c1cn[se]n1]}', '{\\[c1cn[nH]n1]}', '{\\[c1cnsn1]}', '{\\[c1cnon1]}']:
        replace_token = token.replace('\\', '~')
        fsmiles = fsmiles.replace(token, replace_token)

    if fsmiles.count('~') >= 2:
        tokens = FSmilesTokenizer.tokenize(fsmiles)
        if len(tokens) > 2:
            mol_tokens_1 = Chem.MolFromSmiles(''.join(''.join(tokens[1].split('[')[-1]).split(']')[:-1]))

            if '~' in tokens[1] and mol_tokens_1.GetNumAtoms() == 5:
                tokens[1] = tokens[1].replace('~', '')
                tokens[0] = '~' + tokens[0]
        fsmiles = ''.join(tokens)

    return fsmiles


def parse_fsmiles_tokens_to_connections(fsmiles_tokens):
    """
    the ordered rings to fsmiles

    parameters:
        mol: RDKit mol
        ring_order: a list of ring node with the order of DFS

    return:
        str: fsmiles
    """

    def dfs(fsmiles_tokens, current_id=0, parent_idx=None):

        unit_types = {}
        connections = defaultdict(list)

        branches, parent_idx_list = find_outermost_braces(fsmiles_tokens)
        branches = sorted(branches, key=lambda x: len(x), reverse=True)

        main_chain = []
        refreshed_parent_idx_list = []
        i = 0
        n = len(fsmiles_tokens)
        while i < n:
            if fsmiles_tokens[i] == '{':
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if fsmiles_tokens[i] == '{':
                        depth += 1
                    elif fsmiles_tokens[i] == '}':
                        depth -= 1
                    i += 1
            else:
                main_chain.append(fsmiles_tokens[i])
                if i in parent_idx_list:
                    refreshed_parent_idx_list.append(len(main_chain) - 1)
                i += 1
        refreshed_parent_idx_list = [idx + current_id for idx in refreshed_parent_idx_list]

        main_chain_ids = []
        if parent_idx is not None:
            connections[current_id] = [parent_idx]
            connections[parent_idx].append(current_id)
        for i, unit in enumerate(main_chain):
            unit_types[current_id] = unit
            if not connections.get(current_id, 0):
                connections[current_id] = []

            if i > 0:
                connections[current_id].append(current_id - 1)
                connections[current_id - 1].append(current_id)

            main_chain_ids.append(current_id)
            current_id += 1

        for i, (child, parent_idx) in enumerate(zip(branches, refreshed_parent_idx_list)):
            unit_types_branched, connections_branched = dfs(child, current_id, parent_idx)
            unit_types = {**unit_types, **unit_types_branched}
            connections = concat(connections, connections_branched)

        return unit_types, connections

    unit_types, connections = dfs(fsmiles_tokens)
    return unit_types, connections


def ring_smiles_from_token(token):
    rings = []
    for fsmiles_tokens in token:
        mark_left = 0
        mark_right = len(fsmiles_tokens) - 1
        for id, char in enumerate(fsmiles_tokens):
            if char == '[' and fsmiles_tokens[id + 1] != 'n' and fsmiles_tokens[id + 1] != 's' and fsmiles_tokens[id + 1] != 'S':
                mark_left = id + 1
            if char == ']' and fsmiles_tokens[id - 1].isdigit():
                mark_right = id
        rings.append(fsmiles_tokens[mark_left:mark_right])
    return rings


def reaction_for_main_chain(main_chain: list, reaction_manager):
    rings = ring_smiles_from_token(main_chain)
    rings = [Chem.MolFromSmiles(ring) for ring in rings]
    markers = [fsmiles_tokens.split('[')[0] for fsmiles_tokens in main_chain]
    for idx, ring in enumerate(rings):
        if idx == 0:
            check_add_marker = False
            for k, v in uni_patterns.items():
                if ring.HasSubstructMatch(v):
                    check_add_marker = True
                    reaction_name = k
                    marker = markers[idx]
                    marker_core = ''
                    if len(marker) > 0:
                        if marker[0] in ['1','2','3']:
                            marker_core = marker_core + marker[0]
                        if marker[-1] in ['>', '<', '~']:
                            marker_core = marker_core + marker[-1]
                    reaction_name_0 = marker_core + reaction_name
                    if reaction_name_0.startswith('3'):
                        reaction_manager.update_ring_markers(add=True)
                        reaction_manager.update_branch_markers()
                    if reaction_manager.reactions_add_markers.get(reaction_name_0, 0) != 0:
                        try:
                            mol = get_products(reaction_manager.reactions_add_markers[reaction_name_0].RunReactants([ring]))
                        except:
                            raise Exception('FSMILES parsing failure.')
                    elif reaction_manager.reactions_add_markers_check.get(reaction_name, 0) != 0:
                        try:
                            mol = get_products(reaction_manager.reactions_add_markers_check[reaction_name].RunReactants([ring]))
                        except:
                            raise Exception('FSMILES parsing failure.')
                    else:
                        raise Exception('FSMILES parsing failure.')

                    reaction_name_0 = marker + reaction_name
                    break
            assert check_add_marker, f'Ring {Chem.MolToSmiles(ring)} is not dealt successfully.'

        else:
            check_add_marker = False
            for k, v in uni_patterns.items():
                if ring.HasSubstructMatch(v):
                    check_add_marker = True
                    reaction_name = k
                    marker = markers[idx]

                    marker_core = ''
                    if len(marker) > 0:
                        if marker[0] in ['1','2','3']:
                            marker_core = marker_core + marker[0]
                        if marker[-1] in ['>', '<', '~']:
                            marker_core = marker_core + marker[-1]
                    reaction_name_1 = marker_core + reaction_name

                    if reaction_name_1.startswith('3'):
                        reaction_manager.update_ring_markers(add=True)
                    if reaction_manager.reactions_add_markers.get(reaction_name_1, 0) != 0:
                        try:
                            ring = get_products(reaction_manager.reactions_add_markers[reaction_name_1].RunReactants([ring]))
                        except:
                            raise Exception('FSMILES parsing failure.')
                    elif reaction_manager.reactions_add_markers_check.get(reaction_name, 0) != 0:
                        try:
                            ring = get_products(reaction_manager.reactions_add_markers_check[reaction_name].RunReactants([ring]))
                        except:
                            raise Exception('FSMILES parsing failure.')
                    else:
                        raise Exception('FSMILES parsing failure.')
                    reaction_name_1 = marker + reaction_name
                    break
            assert check_add_marker, f'Ring {Chem.MolToSmiles(ring)} is not dealt successfully.'

            reaction_name_1_marker = reaction_name_1.split(k)[0]
            if len(reaction_name_1_marker) > 0:
                reaction_direction = reaction_name_1_marker
                if reaction_name_1_marker[0] in ['1','2','3']:
                    reaction_direction = reaction_direction[1:]
                if reaction_name_1_marker[-1] in ['>', '<']:
                    reaction_direction = reaction_direction[:-1]
            else:
                reaction_direction = ''

            if reaction_manager.reactions.get(f'{reaction_name_0}-{reaction_name_1}', 0) != 0:
                try:
                    mol = get_products(reaction_manager.reactions[f'{reaction_name_0}-{reaction_name_1}'].RunReactants([mol, ring]))
                except:
                    raise Exception('FSMILES parsing failure.')

            elif idx == 1 and reaction_manager.reactions_base_start.get(reaction_name_0, 0) != 0:
                try:
                    mol = get_products(reaction_manager.reactions_base_start[reaction_name_0].RunReactants([mol, ring]))
                except:
                    raise Exception('FSMILES parsing failure.')

            elif idx == 1 and reaction_manager.reactions_base_start.get(reaction_name_0+'-'+reaction_direction, 0) != 0:
                try:
                    mol = get_products(reaction_manager.reactions_base_start[reaction_name_0+'-'+reaction_direction].RunReactants([mol, ring]))
                except:
                    raise Exception('FSMILES parsing failure.')

            elif reaction_manager.reactions.get(f'{reaction_name_0}-{reaction_name_1_marker}', 0) != 0:
                try:
                    mol = get_products(reaction_manager.reactions[f'{reaction_name_0}-{reaction_name_1_marker}'].RunReactants([mol, ring]))
                except:
                    raise Exception('FSMILES parsing failure.')
            elif reaction_manager.reactions.get(f'{reaction_name_1}', 0) != 0:
                try:
                    mol = get_products(reaction_manager.reactions[f'{reaction_name_1}'].RunReactants([mol, ring]))
                except:
                    raise Exception('FSMILES parsing failure.')
            elif reaction_manager.reactions.get(f'{reaction_direction}', 0) != 0:
                try:
                    mol = get_products(reaction_manager.reactions[f'{reaction_direction}'].RunReactants([mol, ring]))
                except:
                    raise Exception('FSMILES parsing failure.')
            else:
                raise Exception('FSMILES parsing failure.')

            if reaction_name_1.startswith('3'):
                reaction_manager.update_branch_markers()

            smi = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smi)
            reaction_name_0 = reaction_name_1

    return mol, reaction_manager


def reaction_for_branched_chain(main_chain, branch_chain, child, reaction_manager):
    mark_left = 0
    mark_right = len(child) - 1
    for id, char in enumerate(child):
        if char == '[' and child[id + 1] != 'n' and child[id + 1] != 's':
            mark_left = id + 1
        if char == ']' and child[id - 1].isdigit():
            mark_right = id
    child_rings = child[mark_left:mark_right]

    child_rings = Chem.MolFromSmiles(child_rings)
    marker = child.split('[')[0]

    reaction_name = None
    for k, v in uni_patterns.items():
        if child_rings.HasSubstructMatch(v):
            break

    if len(marker) > 0:
        if marker[-1] in ['>', '<']:
            marker = marker[:-1]
        reaction_name = marker + k
        if marker[0] in ['1', '2', '3']:
            marker = marker[1:]

    if reaction_name in ['1/benzene', '1\\benzene']:
        mol = get_products(reaction_manager.reactions_branch_connection[reaction_name].RunReactants([main_chain, branch_chain]))
    else:
        mol = get_products(reaction_manager.reactions_branch_connection[marker].RunReactants([main_chain, branch_chain]))
    smi = Chem.MolToSmiles(mol)
    smi = smi.replace('~', '')
    mol = Chem.MolFromSmiles(smi)

    return mol, reaction_manager


def parse_fsmiles_tokens(fsmiles_tokens):
    """
    the ordered rings to fsmiles

    parameters:
        mol: RDKit mol
        ring_order: a list of ring node with the order of DFS

    return:
        str: fsmiles
    """

    reaction_manager = ReactionManager()

    def dfs(fsmiles_tokens, current_id=0, parent_idx=None, reaction_manager=None):

        unit_types = {}
        connections = defaultdict(list)

        branches, parent_idx_list = find_outermost_braces(fsmiles_tokens)
        main_chain = []
        refreshed_parent_idx_list = []
        i = 0
        n = len(fsmiles_tokens)
        while i < n:
            if fsmiles_tokens[i] == '{':
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if fsmiles_tokens[i] == '{':
                        depth += 1
                    elif fsmiles_tokens[i] == '}':
                        depth -= 1
                    i += 1
            else:
                main_chain.append(fsmiles_tokens[i])
                if i in parent_idx_list:
                    refreshed_parent_idx_list.append(len(main_chain) - 1)
                i += 1
        refreshed_parent_idx_list = [idx + current_id for idx in refreshed_parent_idx_list]

        main_chain_ids = []
        if parent_idx is not None:
            connections[current_id] = [parent_idx]
            connections[parent_idx].append(current_id)
        for i, unit in enumerate(main_chain):
            unit_types[current_id] = unit
            if not connections.get(current_id, 0):
                connections[current_id] = []
            if i > 0:
                connections[current_id].append(current_id - 1)
                connections[current_id - 1].append(current_id)
            main_chain_ids.append(current_id)
            current_id += 1

        main_chain_mol, reaction_manager = reaction_for_main_chain(main_chain, reaction_manager)

        branch_markers_idx = reaction_manager.branch_markers_idx
        for i, (child, parent_idx) in enumerate(zip(branches, refreshed_parent_idx_list)):
            branched_chain_mol, reaction_manager = dfs(child, current_id, parent_idx, reaction_manager)
            reaction_manager.get_branch_markers(idx=branch_markers_idx-len(branches)+i+1)
            main_chain_mol, reaction_manager = reaction_for_branched_chain(main_chain_mol, branched_chain_mol, child[0], reaction_manager)
        reaction_manager.get_branch_markers(idx=reaction_manager.branch_markers_idx-len(branches))

        return main_chain_mol, reaction_manager

    if len(fsmiles_tokens) > 1:
        mol, reaction_manager = dfs(fsmiles_tokens, reaction_manager=reaction_manager)
        while reaction_manager.reactions_clear['clear_marker'].RunReactants([mol]):
            mol = get_products(reaction_manager.reactions_clear['clear_marker'].RunReactants([mol]))
        reaction_manager.reset()
        smi = Chem.MolToSmiles(mol)
    else:
        smi = ring_smiles_from_token(fsmiles_tokens)[0]

    return smi
