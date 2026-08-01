from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Geometry import Point3D

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from datetime import datetime

import argparse

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")


def log(text):
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")
########################################################################
# FUNCTION DEFINITIONS
########################################################################

# DATA EXTRATION FUNCTIONS

def get_xtbout_json(json_file: Path) -> dict:
    """
    Extracts the data from the json file

    Parameters
    ----------
    json_file : str
        path to the json file

    Returns
    -------
    dict
        dictionary with the data from the json file
    """
    with open(json_file, "r", encoding="MacRoman") as f:
        # reaplace \ with \\ to avoid json errors
        text = f.read().replace("\\", "\\\\")
        # replace r'"{2}(.*?)"' to bypass issues in windows
        text = re.sub(r'"{2}(.*?)"', r'"\1', text)
        data = json.loads(text)
        # data = json.load(f)
    call_info = get_callinfo_json(data)
    return {**data,**call_info} 

def get_callinfo_json(json_data: dict) -> dict:
    """
    Extracts the call info from the json file

    Parameters
    ----------
    json_data : dict
        dictionary with the data from the json file

    Returns
    -------
    dict
        dictionary with the call info
    """
    
    try:
        uhf = re.findall("--uhf\s+(-?\d+)\s", json_data["program call"])[0]
    except:
        uhf = 0
    try:
        chrg = re.findall("--chrg\s+(-?\d+)\s", json_data["program call"])[0]
    except:
        chrg = 0
    try:
        acc = re.findall("-o.*\s+([A-z]+)\s", json_data["program call"])[0] # match the keywords after the --opt flag
    except:
        acc = "normal"

    data = {"charge": chrg,
            "acc":acc,
            "uhf":uhf,}
    return data 

def get_xtbout_txt(txt_file: Path) -> dict:
    """
    Extracts the data from the txt file

    Parameters
    ----------
    txt_file : str
        path to the txt file

    Returns
    -------
    dict
        dictionary with the data from the txt file
    """
    with open(txt_file, encoding="MacRoman") as f:
        xtb_out_text = f.read()
    data = parse_properties_xtbout_txt(xtb_out_text)
    return data

def parse_properties_xtbout_txt(xtb_out_text: str) -> dict:
    """
    Extracts the data from the txt file

    Parameters
    ----------
    xtb_out_text : str
        text from the xtb output file

    Returns
    -------
    dict
        dictionary with the data from the txt file
    """
    try:
        SCC_energy = re.findall("SCC energy\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        SCC_energy = np.NaN
    try:
        isotropic_ES = re.findall("\sisotropic ES\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        isotropic_ES = np.nan
    try:
        anisotropic_ES = re.findall("anisotropic ES\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        anisotropic_ES = np.nan
    try:
        anisotropic_XC = re.findall("anisotropic XC\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        anisotropic_XC = np.nan
    try:
        dispersion = re.findall("dispersion energy\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        try:
            dispersion = re.findall("dispersion\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
        except:
            pass
        dispersion = np.nan
    try:
        repulsion_energy = re.findall("repulsion energy\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        repulsion_energy = np.nan
    try:
        add_restraining = re.findall("add\. restraining\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        add_restraining = np.nan
    try:
        total_charge = re.findall("total charge\s+(-?\d+\.\d{4})", xtb_out_text)[1] 
    except:
        total_charge = np.nan
    try:
        imaginary_frecuencies = re.findall("imaginary freq\.\s+(\d+)", xtb_out_text)[0]
    except:
        imaginary_frecuencies = np.nan
    try:
        zpe = re.findall("zero point energy\s+(-?\d+\.\d{4})", xtb_out_text)[-1]
    except:
        zpe = np.nan
    try:
        nfod = re.findall("NFOD :\s+(.*)", xtb_out_text)[-1]
    except:
        nfod = np.nan
    data = {"SCC_energy":SCC_energy,
            "zpe":zpe,
            "nfod":nfod,
            "isotropic_ES":isotropic_ES,
            "anisotropic_ES":anisotropic_ES,
            "anisotropic_XC":anisotropic_XC,
            "dispersion":dispersion,
            "repulsion_energy":repulsion_energy,
            "add_restraining":add_restraining,
            "total_charge":total_charge,
            "imaginary_frecuencies":imaginary_frecuencies,}
    return data




# COORDINATES RELATED FUNCTIONS

def get_opt_coords(xyz_file: Path) -> str:
    """
    Extracts the coordinates from the xyz file

    Parameters
    ----------
    xyz_file : str
        path to the xyz file

    Returns
    -------
    str
        string with the coordinates
    """
    if not Path(xyz_file).exists():
        return None
    with open(xyz_file) as f:
        xyz_block = f.read()
    return xyz_block


def update_coordinate(mol: Chem.rdchem.Mol,
                    xyz_block: str,
                    verbose: bool = False) -> Chem.rdchem.Mol:
    """
    Use the coordinates from the xyz file to update the coordiates of the input mol object.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        mol object to update
    xyz_block : str
        string with the coordinates
    verbose : bool, optional
        print the atom index and the coordinates, by default False

    Returns
    -------
    Chem.rdchem.Mol
        updated mol object
    """
    if not xyz_block:
        return mol
    try:
        xyz = xyz_block.split("\n")
        mol = Chem.rdchem.RWMol(mol)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            a ,x,y,z = xyz[2:][i].split()
            x,y,z = float(x), float(y), float(z)
            conf.SetAtomPosition(i, Point3D(x,y,z))
            if verbose:
                log(a, mol.GetAtomWithIdx(i).GetSymbol())
                log(i, x,y,z)
    except:
        pass
    return mol

########################################################################
# CLI
########################################################################


def pareser():
    my_parser = argparse.ArgumentParser(prog='xTB_postprocess',
                                        usage='%(prog)s [options]',
                                        description='Diggest the result of the xTB batch optimization.')

    my_parser.add_argument('-f', '--folder',
                           metavar='xtb_results_folder',
                           type=str,
                           help='The folder containing all the xTB results',
                           action='store',
                           )

    my_parser.add_argument('-sdf', '--sdf_file',
                           metavar='sdf_file',
                           type=str,
                           help='the input sdf file',
                           )

    my_parser.add_argument('-o', '--namespace',
                           metavar='namespace',
                           type=str,
                           help='The output files namespace',
                           default=f"out-{TIMESTAMP}"
                           )

    args = my_parser.parse_args()

    return args

if __name__ == '__main__':

    args = pareser()
    if not args.folder or not args.sdf_file:
        raise ValueError("Missing arguments")
    
    xTB_results_folder = Path(args.folder)
    input_sdf_file = Path(args.sdf_file)
    namespace = args.namespace

    LOG_FILE = Path(f"{namespace}.log")


    log(f"reading {input_sdf_file}")
    df = PandasTools.LoadSDF(input_sdf_file.__str__(), removeHs=False, molColName="mol")
    data_list = {}

    log("Parsin results")
    for idx, folder in enumerate(xTB_results_folder.iterdir()):

        mol_name = folder.name
        chrgs = [-1,0,1]
        for chrg in chrgs:
            name_space = f"{mol_name}_CHRG_{chrg}"
            print(f"{name_space}")

            json_file = folder / (name_space + ".xtbout.json")
            txt_file = folder / (name_space + ".xtbout.txt")
            mol_file = folder / (name_space + ".xtbtopo.mol")
            xyz_file = folder / (name_space + ".xtbopt.xyz")
            
            if not json_file.exists(): print("Missing json file: " + json_file.name); continue
            if not txt_file.exists(): print("Missing txt file: " + txt_file.name); continue
            #if not xyz_file.exists(): print("Missing xyz file: " + xyz_file.name); continue

            try:
                data_from_txt = get_xtbout_txt(txt_file)
            except:
                print(f"Error parsing {txt_file}")
            data_from_json = get_xtbout_json(json_file)
            
            xyz_block = get_opt_coords(xyz_file)
            try:
                mol_inp = df.query(f"name == '{mol_name}'").mol.iloc[0]
            except  IndexError:
                print(f"name == '{mol_name}'")
                mol_inp = df.query(f"name == '{mol_name}'").mol.iloc[0]
            
            mol_opt = update_coordinate(mol_inp, xyz_block)

            data = {"mol":mol_opt,"name":mol_name, **data_from_json, **data_from_txt}
            
            if not data_list:
                for key, value in data.items():
                    data_list[key] = [value]
            else:
                for key, value in data.items():
                    data_list[key].append(value)

        
    df_out = pd.DataFrame(data_list)
    
    log("Calculating smiles")
    df_out["smiles"] = df_out.apply(lambda x: Chem.MolToSmiles(x.mol),axis=1)
    
    # csv_out = f"{namespace}-{TIMESTAMP}.csv"
    # sdf_out = f"{namespace}-{TIMESTAMP}.sdf"
    csv_out = f"{namespace}.csv"
    sdf_out = f"{namespace}.sdf"

    log("Writing csv file")
    df_out.drop("mol",axis=1).to_csv(csv_out)
    
    log("Writing sdf file")
    PandasTools.WriteSDF(df_out, sdf_out,
                        properties=["mol", "name", "charge", 
                        "total energy",
                        "HOMO-LUMO gap/eV",
                        "electronic energy",],
                        molColName="mol")
    