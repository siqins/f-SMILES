import pandas as pd
import multiprocessing

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers

from time import time
from datetime import datetime

import argparse
from pathlib import Path

from typing import List, Tuple, Any

# Timestamp
TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")


# FUNCTION DEFINITIONS


def mol_from_smiles(smi: str) -> Chem.Mol:
    """
    Generates an RDKit molecule from a SMILES string.

    Args:
        smi (str): The SMILES string representing the molecule.

    Returns:
        Chem.Mol: The RDKit molecule object.

    """
    return Chem.MolFromSmiles(smi)

def add_hs_to_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Adds hydrogens to an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        Chem.Mol: The RDKit molecule object with added hydrogens.

    """
    return Chem.AddHs(mol)

def gen2D(mol: Chem.Mol) -> Tuple[Chem.Mol, any]:
    """
    Generates 2D coordinates for an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        tuple[Chem.Mol, any]: A tuple containing the RDKit molecule object and the generated 2D coordinates.
        If the generation fails, the second element is set to -1.

    """
    try:
        embeded = AllChem.Compute2DCoords(mol)
        return mol, embeded
    except:
        print(f"ERROR: Failed to Compute2DCoords molecule: {Chem.MolToSmiles(mol)}")   
        return mol, -1

def gen3D(mol: Chem.Mol) -> Tuple[Chem.Mol, any]:
    """
    Generates 3D coordinates for an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        tuple[Chem.Mol, any]: A tuple containing the RDKit molecule object and the generated 3D coordinates.
        If the generation fails, the second element is set to -1.

    """
    try:
        embeded = AllChem.EmbedMolecule(mol, ETversion=2,
                                        maxAttempts =5, #the maximum number of attempts to try embedding
                                        useRandomCoords=True, # Start the embedding from random coordinates instead of using eigenvalues of the distance matrix.
                                        useBasicKnowledge=False,
                                        useSmallRingTorsions=True)
        return mol, embeded
    except:
        print(f"ERROR: Failed to embed molecule: {Chem.MolToSmiles(mol)}")   
        return mol, -1
    

def uff_optimize(mol: Chem.Mol) -> Tuple[Chem.Mol, any]:
    """
    Performs UFF optimization on an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        tuple[Chem.Mol, any]: A tuple containing the RDKit molecule object and the optimized structure.
        If the optimization fails, the second element is set to -1.

    """
    try:
        if not rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            return mol, -1
        embeded = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        return mol, embeded
    except:
        print(f"ERROR: Failed to UFFOptimize molecule: {Chem.MolToSmiles(mol)}")   
        return mol, -1

def uff_specialoptimize(mol: Chem.Mol) -> Tuple[Chem.Mol, any]:
    """
    Performs special UFF optimization on an RDKit molecule by modifying boron atoms.

    Args:
        mol (Chem.Mol): The RDKit molecule object.

    Returns:
        tuple[Chem.Mol, any]: A tuple containing the RDKit molecule object and the optimized structure.
        If the optimization fails, the second element is set to -1.

    """
    # spboron_pattern = Chem.MolFromSmarts("[#5H0]")
    spboron_pattern = Chem.MolFromSmarts("[#34H0]")
    print(mol.GetSubstructMatches(spboron_pattern))
    for match in mol.GetSubstructMatches(spboron_pattern):
        boron = mol.GetAtomWithIdx(match[0])
        boron.SetHybridization(Chem.rdchem.HybridizationType.SP3)
    try:
        embeded = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    except:
        embeded = -1
    return mol, embeded


def parser():
    parser = argparse.ArgumentParser(description='Conformer generation script')
    # input file name positional argument
    parser.add_argument('input', type=str, help='input files name.')
    parser.add_argument('--output', type=str, default=None, help='output files name. Default: input file name')
    parser.add_argument('--n_cpu', type=int, default=6, help='Number of CPUs to use. Default: 1')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    
    print("Conformer generation script")
    args = parser()
    print(args)
    if not args.output:
        output = Path(args.input).stem
    else:
        output = args.output
    
    N_CPU = args.n_cpu
    INPUT_FILE = Path(args.input)
    CSV_FILE = Path(f"{output}-UFFopt.csv")
    SDF_FILE_EMB = Path(f"{output}-ETKDG.sdf")
    SDF_FILE_OPT = Path(f"{output}-UFFopt.sdf")
    
    print("Conformer generation script")
    #Reading the CSV_FILE
    df = pd.read_csv(INPUT_FILE)
    print(df.head())
    
    print(f"{N_CPU} cpus allocated\n")
    with multiprocessing.Pool(N_CPU) as pool:
        t0 = time()
        print("Generating mol from smiles")
        moles = pool.map(mol_from_smiles, list(df.smi))
        t1 = time()

        print("Adding hydrogens to molecules")
        moles = pool.map(add_hs_to_mol, moles)
        t2 = time()

        print("Generating mol from smiles")
        moles = pool.map(gen3D, moles) 
        t3 = time()
    
    embed = [flag for _, flag in moles]
    moles = [mol  for mol, _ in moles]
    df['mol'] = moles
    df['mol-ETKDG'] = moles
    df['embed'] = embed

    with multiprocessing.Pool(N_CPU) as pool:
        t0 = time()
        print("Optimizing structures ...")
        moles = pool.map(uff_specialoptimize, list(df.mol))
        t1 = time()    

    embed = [flag for _, flag in moles]
    moles = [mol  for mol, _ in moles]    

    df['mol'  ] = moles
    df['optim'] = embed

    # SDF file with optimized molecules
    t2 = time()
    PandasTools.WriteSDF(df,str(SDF_FILE_OPT), molColName="mol", properties=df.columns)
    t3 = time()

    # CSV file with optimized molecules
    t4 = time()
    df.drop('mol', axis=1).to_csv(CSV_FILE)
    t5 = time()

    # SDF file with embeded molecules
    t6 = time()
    PandasTools.WriteSDF(df,str(SDF_FILE_EMB), molColName="mol-ETKDG", properties=df.columns) 
    t7 = time()

    # # Time
    print("\n\n")
    print(df.columns)
    
    print(f"uff_optimize: {t1-t0:.0f}s")
    print(f"WriteSDF: {t3-t2:.0f}s")    
    print(f"WriteCSV: {t5-t4:.0f}s")
    print(f"WriteSDF: {t7-t6:.0f}s")
