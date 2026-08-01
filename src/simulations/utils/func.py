from pathlib import Path

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from typing import List, Tuple, Any, Dict

def mol_from_smiles(smi):
    return Chem.MolFromSmiles(smi)

def add_hs_to_mol(mol):
    return Chem.AddHs(mol)

def gen3D(mol):
    try:
        embeded = AllChem.EmbedMolecule(mol, ETversion=2)
    except:
        embeded = -1
    return mol, embeded

def gen2D(mol):
    try:
        embeded = AllChem.Compute2DCoords(mol)
    except:
        embeded = -1
    return mol, embeded

def uff_optimize(mol):
    try:
        embeded = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    except:
        embeded = -1
    return mol, embeded

def mmff_optimize(mol):
    try:
        embeded = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    except:
        embeded = -1
    return mol, embeded

def has_clashes(mol):
    # compute molecule distance matrix Get3DDistanceMatrix and
    # check if the distance matrix has any value less than 0.1 (quite arbitrary)
    clash = np.where(Chem.Get3DDistanceMatrix(mol, useAtomWts= True) < 0.1 )[0].shape[0] > 0
    if clash:
        return True
    else: 
        return False

def has_clashes2(mol, ):
    # compute molecule distance matrix Get3DDistanceMatrix and
    # check if the distance matrix has any value less than 0.1 (quite arbitrary)
    clash = np.where(Chem.rdDistGeom.GetMoleculeBoundsMatrix(mol) < 0.1 )[0].shape[0] > 0
    if clash:
        return True
    else: 
        return False

def check_bond_lenghts(mol, max_bond_length=2.0):
    conformer = mol.GetConformers()[0]
    if conformer is None: return None
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_length = Chem.rdMolTransforms.GetBondLength(conformer, begin_atom_idx, end_atom_idx)
        if bond_length >= max_bond_length: 
            return True # wrong bond length
    return False


# this function should be parallelized
def concat_files(folder, name_pattern="", concat_file_name=None):
    if not concat_file_name: concat_file_name = folder/ "concat.txt"
    folder = Path(folder)
    t = ""
    for i, file in enumerate(folder.iterdir()):
        if "UFFopt" in file.name: 
            with open(file,"r") as f:
                t += f.read()
    with open(concat_file_name, "w") as f:
        f.write(t)

def get_atoms(formula: str)-> Dict[str, int]:
    """
    Get the atoms from a molecular formula.

    Args:
        formula (str): The molecular formula.

    Returns:
        dict[str, int]: A dictionary containing the atoms and their counts.

    Example:
        >>> get_atoms("C2H6O")
        ... {"h":6,
            "c":2,
            "b":0,
            "s":0,
            "o":1,
            "n":0,
            }
    """

    # print(formula)
    p = re.compile("((\w)(\d*))")
    matches = p.findall(formula)
    atoms = {"H":0,
            "C":0,
            "B":0,
            "S":0,
            "O":0,
            "N":0,
            }
    for match in matches:
        # print(match)
        if match[2] == "":
            atoms[match[1]] += 1
        else:
            atoms[match[1]] += int(match[2])
    # print(atoms)
    return atoms

def n_part_split(list, batch_size=None, n_batch=None):
    bathces = []
    if not batch_size and not batch_size:
        raise ValueError("batch_size or n_batch must be specified")
    if not batch_size:
        batch_size = len(list) / n_batch
    if not batch_size:
        n_batch = len(list) // batch_size
        
    for i in range(n_batch):
        bathces.append(list[i::n_batch])
    return bathces

def display_mol(mol):
    molBlock = Chem.MolToMolBlock(mol)
    p = py3Dmol.view(width=400,height=400)
    p.removeAllModels()
    p.addModel(molBlock,'sdf')
    p.setStyle({'stick':{}})
    # p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    p.show()

def read_sdf(sdf_file):
    sdf_file = str(sdf_file)
    df = PandasTools.LoadSDF(sdf_file, molColName="mol", removeHs=False)
    return df

def has_sp2boron(mol):
    sp2boron_pattern = Chem.MolFromSmarts("[#5H0]")
    return mol.HasSubstructMatch(sp2boron_pattern)

def has_biciclobutadiene(mol):
    bi_ciclobutadiene = Chem.MolFromSmarts("*~[#6]~1~[#6](~*)~[#6]~2~[#6](~*)~[#6](~*)~[#6]~1~2")
    return mol.HasSubstructMatch(bi_ciclobutadiene)

# System related
#Get CPU info 
import os, platform, subprocess, re, multiprocessing

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

def show_cpu_info():
    print("CPU_MODEL:",get_processor_name())
    print("N_CPU:",multiprocessing.cpu_count())

def get_num_branching(mol):
    branched_ring_pattern = Chem.MolFromSmarts("[*;R2r6]~1~[*;R2r6]~[*;R2r6]~[*;R2r6]~[*;R2r6]~[*;R2r6]~1")
    return(len(mol.GetSubstructMatches(branched_ring_pattern)))

def read_csv(csv_file):
    return pd.read_csv(csv_file, index_col=0) 

def load_sdf(sdf_file):
    return PandasTools.LoadSDF(sdf_file, molColName="mol", removeHs=False)

def calc_aea_aip(data_frame: pd.DataFrame)->pd.DataFrame:
    """
    Calculate the IP and EA from the energies of the neutral, cation and anion.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Dataframe with the energies of the neutral, cation and anion.
    
    Returns
    -------
    pd.DataFrame
        Dataframe with the calculated IP and EA.
    """
    data_frame = data_frame.sort_values(by="name")
    # get the energies
    enery_neutral = data_frame.query("charge == 0")["energy"]
    enery_cation = data_frame.query("charge == 1")["energy"]
    enery_anion = data_frame.query("charge == -1")["energy"]

    # calculate the IP and EA
    IP = enery_cation.values - enery_neutral.values
    EA = enery_anion.values - enery_neutral.values 

    # assign the values in the dataframe
    data_frame['aip'] = np.nan
    data_frame['aea'] = np.nan
    data_frame.loc[data_frame['charge'] == 0, 'aip'] = IP * 27.2114
    data_frame.loc[data_frame['charge'] == 0, 'aea'] = EA * 27.2114
    return data_frame

def calc_gap(data_frame):
    """Calculate the homo/lumo gap"""
    data_frame = data_frame.sort_values(by="name")
    data_frame['gap'] = data_frame['lumo'] - data_frame['homo']
    return data_frame


# calculate the r2_score, rmse, mae, mse, p, r^2 of the variables

def get_metrics(df: pd.DataFrame, col: str):
    """
    Calculate the metrics for a given column (col) in a dataframe
    and the reference column (_dft)

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data
    col : str
        The column name

    Returns
    -------
    r2 : float
        The r2 score
    rmse : float
        The root mean squared error
    mae : float
        The mean absolute error
    mse : float
        The mean squared error
    p : float
        The pearson correlation coefficient
    var : float
        The variance

    Examples
    --------
    >>> get_metrics(df, 'dft')
    ... (0.9999999999999999, 0.0, 0.0, 0.0, 1.0, 0.0)
    """
    ref = f"{col.split('_')[0]}_dft"
    r2 = r2_score(df[ref], df[col])
    mae = mean_absolute_error(df[ref], df[col])
    mse = mean_squared_error(df[ref], df[col])
    rmse = np.sqrt(mean_squared_error(df[ref], df[col]))
    p = stats.pearsonr(df[ref], df[col])[0]
    var = np.var(df[ref] - df[col])
    return r2, rmse, mae, mse, p, var

def get_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the metrics for all columns in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data

    Returns
    -------
    metrics : pd.DataFrame
        A new dataframe containing the metrics for each column
    """

    metrics = []
    for col in df.columns:
        if col == 'name': continue
        if 'energy' in col: continue
        if 'dft' in col: continue
        r2, rmse, mae, mse, p, var = get_metrics(df, col)        
        metrics.append([col, r2, rmse, mae, mse, p, var])
    return pd.DataFrame(metrics, columns=['method', 'r2', 'rmse', 'mae', 'mse', 'pearson', 'var']).T