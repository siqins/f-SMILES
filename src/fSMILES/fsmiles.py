import os
import sys
import json
import random
import pandas as pd
import subprocess
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize
from joblib import Parallel, delayed
from mol_utils.MolParser import molecule_to_ring_graph, dfs_ring_order, get_ring_smiles, rings_to_fsmiles
from mol_utils.MolParser import parse_fsmiles_tokens
from frag_units.fragunits import FragUnitCalc, unit_classify, get_fused_rings

def standardize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)

        standardizer = MolStandardize.standardize.Standardizer()
        mol = standardizer.standardize(mol)

        uncharger = MolStandardize.charge.Uncharger()
        mol = uncharger.uncharge(mol)

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canonical_smiles

    except Exception as e:
        print(f"SMILES standardization fails: {smiles}, error: {e}")
        return None

class FSmilesTokenizer:

    def __init__(self, verbose=True):
        self.verbose = verbose

    @ staticmethod
    def tokenize(smiles: str):
        n = len(smiles)
        mark_left = 0
        mark_right = 0
        tokens = []
        for i in range(n):
            if smiles[i] == ']' and smiles[i-1].isdigit() and smiles[i-4:i-2] != 'Si':
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


def canonicalize(smiles: str) -> str:
    cano_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return cano_smiles


def check_fsmiles(fsmiles: str) -> list:

    select_rings = ['\\[c1ccsc1]', '\\[C1=CCC=C1]', '\\[c1cc[nH]c1]', '\\[c1cc[se]c1]', '\\[c1ccoc1]'
                    '/[c1ccsc1]',  '/[C1=CCC=C1]',  '/[c1cc[nH]c1]',  '/[c1cc[se]c1]',  '/[c1ccoc1]']

    if fsmiles.count('[c1ccccc1]') > 2:
        fsmiles_tokens = FSmilesToTokens(fsmiles, unit_level=False)

        f_benzene = 0
        for idx, tok in enumerate(fsmiles_tokens):
            if tok.find('[c1ccccc1]') != -1:
                f_benzene += 1
            else:
                f_benzene = 0
            if f_benzene > 2:
                # fsmiles_tokens[idx] = '\\[c1ccsc1]'
                fsmiles_tokens[idx] = random.choice(select_rings)
                f_benzene = 0
        fsmiles = ''.join(fsmiles_tokens)

    fsmiles_tokens = FSmilesTokenizer.tokenize(fsmiles)

    if fsmiles_tokens[0] == '2[C1=CCC=CC1]':
        fsmiles_tokens[0] = '[C1C=CC=CC1]'

    targets = ['[c1ccccc1]']
    targets_taken = ['2[c1ccccc1]']
    for targets_idx, target in enumerate(targets):
        idx = [i for i, x in enumerate(fsmiles_tokens) if x == target]
        len_toks = len(fsmiles_tokens)
        for id in idx:
            if id != len_toks - 1:
                fsmiles_tokens[id] = targets_taken[targets_idx]

    targets = ['~[c1ccsc1]', '~[C1=CCC=C1]', '~[c1cc[nH]c1]', '~[c1cc[se]c1]', '~[c1cn[nH]n1]', '~[c1cnsn1]', '~[c1cn[se]n1]', '~[C1=CCCC1]']
    targets_taken = ['[c1ccsc1]', '[C1=CCC=C1]', '[c1cc[nH]c1]', '[c1cc[se]c1]', '[c1cc[nH]c1]', '[c1ccsc1]', '[c1cc[se]c1]', '[C1=CCC=C1]']
    for targets_idx, target in enumerate(targets):
        idx = [i for i, x in enumerate(fsmiles_tokens) if x == target]
        len_toks = len(fsmiles_tokens)
        for id in idx:
            if id != 0 and id != len_toks - 1:
                if len(fsmiles_tokens[id+1]) != 1:
                    fsmiles_tokens[id] = targets_taken[targets_idx]

    return fsmiles_tokens

def PureCoreMolFromSmiles(smiles: str) -> Chem.Mol:
    # clear the branches, and obtained the pure core.
    mol = Chem.MolFromSmiles(smiles)
    mol = get_fused_rings(mol)
    return mol


def PureCoreSmilesFromSmiles(smiles: str, kekuleSmiles: bool=False) ->str:
    # clear the branches, and obtained the pure core.
    mol = PureCoreMolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    return smiles


def RingDfFromSmiles(smiles: str) -> pd.DataFrame:
    frag_unit_calc = FragUnitCalc()
    ring_df = frag_unit_calc.compute([smiles])
    ring_df = unit_classify(ring_df)
    ring_df.sort_values(by='ring_type', inplace=True)
    return ring_df


def DoubleCoreMolFromSmiles(smiles: str):
    # Extract the SMILES of double cores from smiles.
    ring_df = RingDfFromSmiles(smiles)
    ring_df = ring_df[ring_df['ring_type'] == 'double']
    if len(ring_df) == 0:
        return []
    else:
        ring_mols = [Chem.MolFromSmiles(ring_smi) for ring_smi in ring_df['smiles']]
        double_ring_mols = []
        for ring_mol in ring_mols:
            fused_ring_mol = get_fused_rings(ring_mol)
            double_ring_mols.append(fused_ring_mol)
        if len(double_ring_mols) == 1:
            return double_ring_mols[0]
        else:
            return double_ring_mols


def DoubleCoreSmilesFromSmiles(smiles: str, kekuleSmiles: bool=False):
    double_ring_mols = DoubleCoreMolFromSmiles(smiles)
    if isinstance(double_ring_mols, list):
        if len(double_ring_mols) == 0:
            return []
        return [Chem.MolToSmiles(double_ring_mol, kekuleSmiles=kekuleSmiles) for double_ring_mol in double_ring_mols]
    else:
        return Chem.MolToSmiles(double_ring_mols, kekuleSmiles=kekuleSmiles)


def FusedCoreMolFromSmiles(smiles: str):
    # Extract the SMILES of fused cores from smiles.
    ring_df = RingDfFromSmiles(smiles)
    ring_df = ring_df[ring_df['ring_type'] == 'fused']
    if len(ring_df) == 0:
        return []
    else:
        ring_mols = [Chem.MolFromSmiles(ring_smi) for ring_smi in ring_df['smiles']]
        fused_ring_mols = []
        for ring_mol in ring_mols:
            fused_ring_mol = get_fused_rings(ring_mol)
            fused_ring_mols.append(fused_ring_mol)
        if len(fused_ring_mols) == 1:
            return fused_ring_mols[0]
        else:
            return fused_ring_mols


def FusedCoreSmilesFromSmiles(smiles: str, kekuleSmiles: bool=False):
    fused_ring_mols = FusedCoreMolFromSmiles(smiles)
    if isinstance(fused_ring_mols, list):
        if len(fused_ring_mols) == 0:
            return []
        return [Chem.MolToSmiles(fused_ring_mol, kekuleSmiles=kekuleSmiles) for fused_ring_mol in fused_ring_mols]
    else:
        return Chem.MolToSmiles(fused_ring_mols, kekuleSmiles=kekuleSmiles)


def MolFromFSmiles(fsmiles: str) -> Chem.Mol:
    smi = FSmilesToSmiles(fsmiles)
    return Chem.MolFromSmiles(smi)


def FSmilesToSmiles(fsmiles: str, kekuleSmiles: bool=False, check_fsmi: bool=False) -> str:
    try:
        fsmiles = fsmiles.strip()
        if check_fsmi:
            fsmiles_tokens = check_fsmiles(fsmiles)
        else:
            fsmiles_tokens = FSmilesTokenizer.tokenize(fsmiles)
        smi = parse_fsmiles_tokens(fsmiles_tokens)
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), kekuleSmiles=kekuleSmiles)
    except Exception as e:
    # except ValueError:
        smi = None
        # print(e)
    return smi


def MolToFSmiles(mol: Chem.Mol) -> str:
    ring_graph = molecule_to_ring_graph(mol)
    ring_order = dfs_ring_order(ring_graph, mol)
    frags_dict = {tuple(sorted(ring_atoms)): get_ring_smiles(mol, ring_atoms) for ring_atoms in ring_graph.nodes()}
    # for idx, ring_atoms in enumerate(ring_graph.nodes()):
    #     a = get_ring_smiles(mol, ring_atoms)
    fsmiles = rings_to_fsmiles(mol, ring_order, frags_dict)
    return fsmiles


def FSmilesFromSmiles(smiles: str) -> str:
    cano_smiles = canonicalize(smiles)
    mol = Chem.MolFromSmiles(cano_smiles)
    return MolToFSmiles(mol)

def SmilesToFSmiles(smiles: str) -> str:
    return FSmilesFromSmiles(smiles)

def SmilesFromFSmiles(fsmiles: str, kekuleSmiles: bool=False, check_fsmi: bool=False) -> str:
    return FSmilesToSmiles(fsmiles, kekuleSmiles=kekuleSmiles, check_fsmi=check_fsmi)

def FSmilesToTokens(fsmiles: str, unit_level:bool=True) -> list[str]:
    '''
    :param fsmiles:
    :param unit_level: whether to tokenize the branches of molecules, if ture, the branches are tokenized separately.
    '''
    fsmiles = fsmiles.strip()
    fsmiles_tokens = FSmilesTokenizer.tokenize(fsmiles)

    if unit_level:
        return fsmiles_tokens
    else:
        new_fsmiles_tokens = []
        right_bracket_idx = 0
        left_bracket_idx = 0
        right_brackets = 0
        left_brackets = 0
        for idx, token in enumerate(fsmiles_tokens):
            if token == '{':
                left_brackets += 1
                if left_brackets == 1:
                    left_bracket_idx = idx
            elif token == '}':
                right_brackets += 1
                right_bracket_idx = idx
                if left_brackets == right_brackets:
                    new_fsmiles_tokens.pop()
                    new_fsmiles_tokens.append(''.join(fsmiles_tokens[left_bracket_idx-1:right_bracket_idx+1]))
                    right_brackets = 0
                    left_brackets = 0
            else:
                if left_brackets==0:
                    new_fsmiles_tokens.append(token)
        return new_fsmiles_tokens

def BatchSmilesFromFSmiles(batch_fsmis: list[str], n_jobs: int=12, check_fsmi:bool=False) -> list[str]:

    def batch_parse(mini_batch_fsmis):
        path = os.path.dirname(os.path.abspath(__file__))
        code = f"""
import os, sys, json
sys.path.append(r"{path}")
import fsmiles as fs

data = sys.stdin.read()
fsmis = json.loads(data)
for fsmi in fsmis:
    smi = fs.SmilesFromFSmiles(fsmi, check_fsmi={check_fsmi})
    print(smi)
"""

        result = subprocess.run([
            'python', '-c', code
        ], input=json.dumps(mini_batch_fsmis), capture_output=True, text=True)
        smis = result.stdout.split('\n')
        if len(smis) == len(mini_batch_fsmis) + 1:
            smis = smis[:-1]
        return smis

    n_split = 200
    n_batch = len(batch_fsmis)//n_split + 1

    results = Parallel(n_jobs=n_jobs)(
        delayed(batch_parse)(batch_fsmis[idx*n_split:(idx+1)*n_split])
        for idx in tqdm(range(n_batch), ncols=100, file=sys.stdout, position=0, leave=True)
    )
    smis = [smi for sublist in results for smi in sublist]

    return smis
