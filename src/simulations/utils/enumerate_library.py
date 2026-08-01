
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit

from itertools import chain
import numpy as np

import random
import time

from typing import List

# disable rdkit warnings
rdkit.RDLogger.DisableLog('rdApp.*')


class PAHsBuilder(object):
    def __init__(self, mean_nRing: int = 8, max_nRing: int = 10,
                 fragments: List[str] = None,
                 reactions: List[str] = None,
                 fragments_code: List[str] = None,
                 fragments_frequency: List[str] = None,
                 random_seed=None):
        """
        Builder to construct polyaromatics hidrocarbons.

        Args:
        :mean_nRing: average number of ring in the molecule
        :max_nRing: maximum number of ring in the molecule
        :fragments: list of SMILES of the fragments
        :reactions: list of SMARTS reactions to constrcut the molecules
        """
        self.mean_nRing = mean_nRing
        self.max_nRing = max_nRing

        self.fragments = fragments
        self.reactions = reactions
        self.fragment_code = fragments_code
        fragments_frequency = np.array(fragments_frequency) / np.array(fragments_frequency).sum()
        self.fragment_frequency = fragments_frequency.tolist()

        self.fragments_dict = {code: Chem.MolFromSmiles(smi) for smi, code in zip(fragments, fragments_code)}
        self.reactions_dict = {code: AllChem.ReactionFromSmarts(sma) for sma, code in zip(reactions, fragments_code)}

        if random_seed is not None:
            self.seed = random_seed
            np.random.seed(random_seed)
        else:
            self.seed = None

    def get_number_of_rings(self) -> int:
        """
        Returns the number of rings based on a quasi-Poisson distribution.
        """
        n_ring = 0
        ep = self.max_nRing - self.mean_nRing
        while n_ring < 2:
            x = np.random.poisson(ep)
            x = np.round(x, decimals=0)
            n_ring = -x + self.max_nRing
        return n_ring

    def get_sequence(self) -> List[str]:
        """
        Generate a sequence of fragments to build the molecule.
        """
        # seq = random.choices(self.fragment_code, k=self.get_number_of_rings())
        # using np.random.choice is faster than random.choices

        seq = np.random.choice(self.fragment_code, size=self.get_number_of_rings(), p=self.fragment_frequency)
        # seq = np.random.choice(self.fragment_code, size=self.get_number_of_rings())
        return seq

    def generate_mol(self) -> Chem.Mol:
        seq = self.get_sequence()
        mol = self.fragments_dict[seq[0]]
        gen_state = True
        for code in seq[1:]:
            rxn = self.reactions_dict[code]
            ps = rxn.RunReactants([mol])
            mols = get_products(ps)
            mols = filter_list(mols, filter_fn=[filter_12diene], useSmiles=True)
            if len(mols) != 0:
                mol = np.random.choice(mols)
            else:
                gen_state = False

        if gen_state:
            string = ""
            for i in seq: string += f"{i}-"
            mol.SetProp("seq", string[:-1])
            return mol
        else:
            return None


def get_products(products_list: List[List[Chem.Mol]]) -> List[Chem.Mol]:
    """
    Extract unique products from a list of reaction products.

    Args:
    :products_list: list of reaction products

    Returns:
    :mols: list of unique products as RDKit molecules
    """
    unique_smi = set(Chem.MolToSmiles(mol) for mol in chain.from_iterable(products_list))
    # we need to make a list from the set to be able to sort it to ensure reproducibility
    unique_smi = list(unique_smi)
    unique_smi.sort()
    mols = [Chem.MolFromSmiles(smi) for smi in unique_smi if Chem.MolFromSmiles(smi)]
    return mols


def filter_list(mols: List[Chem.Mol], filter_fn: List[callable] = [], useInchi: bool = False, useSmiles: bool = False) -> List[Chem.Mol]:
    """
    Filter a list of molecules based on specified filters.

    Args:
    :mols: list of molecules to filter
    :filter_fn: list of filter functions to apply
    :useInchi: flag indicating whether to use InChI filtering
    :useSmiles: flag indicating whether to use SMILES filtering

    Returns:
    :mols: filtered list of molecules
    """
    if filter_fn is not None:
        for fn in filter_fn:
            mols = fn(mols)
    if useInchi:
        unique_inchis = list(set(Chem.MolToInchi(mol) for mol in mols))
        unique_inchis.sort()
        mols = [Chem.inchi.MolFromInchi(inchi) for inchi in unique_inchis if Chem.inchi.MolFromInchi(inchi)]
    if useSmiles:
        unique_smiles = list(set(Chem.MolToSmiles(mol) for mol in mols))
        unique_smiles.sort()
        mols = [Chem.MolFromSmiles(smi) for smi in unique_smiles if Chem.MolFromSmiles(smi)]
    return mols


def filter_12diene(mols: List[Chem.Mol]) -> List[Chem.Mol]:
    """
    Remove molecules containing a 1,2-diene pattern.

    Args:
    :mols: list of molecules

    Returns:
    :mols: list of molecules without the 1,2-diene pattern
    """
    cleanup_12diene = AllChem.ReactionFromSmarts("[#6:5]=[C:2]=[C:3]=[#6:6]>>[#6:5]=[#6:2]-[#6:3]=[#6:6]")
    for mol in mols:
        while cleanup_12diene.RunReactantInPlace(mol): pass
    return mols


def generate_mol_fn(Builder):
    return Builder.generate_mol()

    # These are hardcoded for now


REACTIONS_SMA = [
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:1]:[c:3]:[c:4]:[c:5]:[c:6]:1",  # benzene
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:6]:1:[c:2]:[c:1]:[n:3]:[c:4]:[c:5]:1",  # pyridine
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:1]:[n:3]:[c:4]:[c:5]:[n:6]:1",  # pyrazine

    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[n:5]:[n:4]([H]):[n:3]:[c:1]:1",  # triazole
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[n:5]:[s:4]:[n:3]:[c:1]:1",  # thiadiazole
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[n:5]:[se:4]:[n:3]:[c:1]:1",  # selenadiazole

    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:5]:[c:4]:[c:3]([H])([H]):[c:1]:1",  # cyclopentadiene
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:5]:[c:4]:[n:3]([H]):[c:1]:1",  # pyrrole
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:5]:[c:4]:[o:3]:[c:1]:1",  # furan
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:5]:[c:4]:[si:3]([H])([H]):[c:1]:1",  # silole
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:5]:[c:4]:[s:3]:[c:1]:1",  # thiophene
    "[#6;H;R1:1]~[#6;H1;R1:2]>>[c:2]:1:[c:5]:[c:4]:[se:3]:[c:1]:1",  # selenophene
]

PATTERNS_SMA = [
    "[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~1",  # benzene
    "[#6]~1~[#6]~[#6]~[#7]~[#6]~[#6]~1",  # pyridine
    "[#7]~1~[#6]~[#6]~[#7]~[#6]~[#6]~1",  # pyrazine

    "[#7]~1~[#7]~[#7]~[#6]~[#6]~1",  # triazole
    "[#7]~1~[#16]~[#7]~[#6]~[#6]~1",  # thiadiazole
    "[#7]~1~[#34]~[#7]~[#6]~[#6]~1",  # selenadiazole

    "[#6]~1~[#6]~[#6]~[#6]~[#6]~1",  # cyclopentadiene
    "[#6]~1~[#7]~[#6]~[#6]~[#6]~1",  # pyrrole
    "[#6]~1~[#8]~[#6]~[#6]~[#6]~1",  # furan
    "[#6]~1~[#14]~[#6]~[#6]~[#6]~1",  # silole
    "[#6]~1~[#16]~[#6]~[#6]~[#6]~1",  # thiophene
    "[#6]~1~[#34]~[#6]~[#6]~[#6]~1",  # selenophene
]

PATTERNS_NAME = [
    "benzene",
    "pyridine",
    "pyrazine",

    "triazole",
    "thiadiazole",
    "selenadiazole",

    "cyclopentadiene",
    "pyrrole",
    "furan",
    "silole",
    "thiophene",
    "selenophene",
]

FRAGMENTS_SMI = [
    "c1ccccc1",  # benzene
    "c1ccncc1",  # pyridine
    "c1cnccn1",  # pyrazine

    "c1cn[nH]n1",  # triazole
    "c1cnsn1",  # thiadiazole
    "c1cn[se]n1",  # selenadiazole

    "C1=CCC=C1",  # cyclopentadiene
    "c1cc[nH]c1",  # pyrrole
    "c1ccoc1",  # furan
    "C1=C[SiH2]C=C1",  # silole
    "c1ccsc1",  # thiophene
    "c1cc[se]c1",  # selenophene
]

FRAGMENTS_CODE = [
    "benzene",
    "pyridine",
    "pyrazine",

    "triazole",
    "thiadiazole",
    "selenadiazole",

    "cyclopentadiene",
    "pyrrole",
    "furan",
    "silole",
    "thiophene",
    "selenophene",
]


FRAGMENTS_FREQUENCY = [
    7670,
    765,
    100,

    146,
    545,
    14,

    3190,
    1291,
    20,
    25,
    6694,
    219
]


if __name__ == "__main__":

    from tqdm import tqdm
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PAHsBuilder CLI")
    parser.add_argument("-n", "--nmols", type=int, default=100,
                        help="number of molecules to generate")
    parser.add_argument("-o", "--outname", type=str, default="gen",
                        help="output file name")
    parser.add_argument("-m", "--mean_nRing", type=int, default=8,
                        help="mean number of rings")
    parser.add_argument("-x", "--max_nRing", type=int, default=10,
                        help="maximum number of rings")
    args = parser.parse_args()

    # Set parameters
    mean_nRing = args.mean_nRing
    max_nRing = args.max_nRing
    nmols = args.nmols
    outname = args.outname

    # Init builder
    Builder = PAHsBuilder(mean_nRing=mean_nRing,
                          max_nRing=max_nRing,
                          fragments=FRAGMENTS_SMI,
                          reactions=REACTIONS_SMA,
                          fragments_code=FRAGMENTS_CODE,
                          fragments_frequency=FRAGMENTS_FREQUENCY)

    pbar = tqdm(total=nmols)  # Init pbar

    # store molecules
    mols = []
    for i in range(nmols):
        pbar.update(n=1)
        mol = Builder.generate_mol()
        if mol is not None:
            mols.append(mol)

    # Save molecules
    smis = [Chem.MolToSmiles(mol) for mol in mols]
    seqs = [mol.GetProp("seq") for mol in mols]

    time_stamp = str(time.time()).split('.')[0]

    with open(f"{outname}.smi", "w+") as f:
        for smi in smis: f.write(f"{smi}\n")
    with open(f"{outname}.seq", "w+") as f:
        for seq in seqs: f.write(f"{seq}\n")
    with open(f"{outname}.csv", "w+") as f:
        for smi, seq in zip(smis, seqs):
            f.write(f"{smi},{seq}\n")