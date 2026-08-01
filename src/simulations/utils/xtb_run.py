import subprocess
import multiprocessing

import os
import shutil
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import PandasTools
from time import time

from datetime import datetime
import argparse
from typing import Optional

#xtb = "./xtb-6.5.1/bin/xtb"

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")

def log(message: str) -> None:
    """
    Write a message to the log file.
    """
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

def get_xtb_path() -> str:
    """
    Return the path to the xtb executable.

    Returns:
        str: The path to the xtb executable.

    Raises:
        FileNotFoundError: If the xtb executable is not found.
    """
    # xtb_path = shutil.which("xtb")
    # xtb_path = "/cluster/home/edmayo/xtb-6.5.1/bin/xtb"
    # xtb_path = "xtb-6.5.1\\bin\\xtb"
    # xtb_path = "./xtb-6.5.1/bin/xtb"
    # xtb_path = r"C:\Users\12233\OneDrive\项目一\PAS\xtb-6.5.1\bin\xtb"
    # xtb_path = r'../../xtb-6.5.1/bin/xtb'
    xtb_path = r'/home/ssq233/项目/PAS/xtb-6.5.1/bin/xtb'
    if xtb_path is None:
        raise FileNotFoundError("xtb executable not found")
    # solve path with spaces in windows
    if os.name == "nt":
        xtb_path = f'"{xtb_path}"'
    return xtb_path

def get_xtb_command(xyz_path: str, gfn: int = 1, opt: bool = True,
                    hess: bool = True, acc: str = "normal",
                    chrg: Optional[int] = 0, uhf: Optional[int] = 0, clean_up: bool = True) -> str:
    """
    Generate the xtb command to run a calculation on a given xyz file with the specified options.

    Args:
        xyz_path (str): The path to the xyz file.
        gfn (int): The GFN version to use (default: 1).
        opt (bool): If True, run an optimization (default: True).
        hess (bool): If True, run a hessian calculation (default: True).
        acc (str): The accuracy of the calculation (default: "normal").
        chrg (Optional[int]): The charge of the molecule (default: 0).
        uhf (Optional[int]): The number of unpaired electrons (default: 0).
        clean_up (bool): If True, remove unnecessary xtb output files (default: True).

    Returns:
        str: The xtb command to run the calculation.

    """
    xtb = get_xtb_path()
    command = f"{xtb} {xyz_path}"
    if opt:
        if hess: 
            command += f" --ohess {acc}"
        else:
            command += f" --opt {acc}"
    else:
        command += f""
        if hess: 
            command += f" --hess"
        else:
            command += f""

    if chrg: command += f" --chrg {chrg}"
    if uhf: command += f" --uhf {uhf}"
    
    namespace = f"{xyz_path.parent / xyz_path.stem}_CHRG_{chrg}"
    command += f" --parallel 1 --gfn {gfn} --molden --fod --etemp 5000 --json --namespace {namespace} > {namespace}.xtbout.txt"
    if clean_up:
        command += f" && rm {namespace}.charges {namespace}.hessian {namespace}.wbo {namespace}.xtbrestart {namespace}.xtbtopo.mol {namespace}.xtbopt.log {namespace}.vibspectrum {namespace}.g98.out"
    return command

def pareser():
    my_parser = argparse.ArgumentParser(prog='xTB_runner',
                                        usage='%(prog)s [options]',
                                        description="""Run the xTB optimization 
                                         and hessian calculation
                                         of all the molecules in an SDF file
                                         write the results of each calculation into 
                                         a separated folder. To process the results use 
                                         postprocess_xtb.py
                                         """)

    # add arguments required for the xTB_runner
    my_parser.add_argument('-f', '--sdf_file',
                           metavar='sdf_file',
                           type=str,
                           help='The input sdf file name',
                           required=True
                           )

    my_parser.add_argument('-o', '--output_folder',
                           metavar='output_folder',
                           type=str,
                           help='The output folder name',
                           default=None
                           )

    my_parser.add_argument('-p', '--processes',
                           metavar='processes',
                           type=int,
                           help='Number of processors to be used for all the calculations. \
                         Eg: 4 processes will run all the calculations in a Pool of 4 processes.',
                           default=None
                           )

    # add arguments required for get_xtb_command
    # add store_true to make the argument a boolean for opt and hess
    my_parser.add_argument('-gfn', '--gfn',
                           help='The GFN version to use (default: 1)',
                           type=int,
                           default=1
                           )

    my_parser.add_argument('-opt', '--opt',
                           help='Run an optimization',
                           action='store_true',
                           )

    my_parser.add_argument('-hess', '--hess',
                           help='Run a hessian calculation',
                           action='store_true',
                           )

    my_parser.add_argument('-acc', '--acc',
                           metavar='acc',
                           type=str,
                           help='Accuracy of the calculation (crude ,sloppy ,loose ,lax ,normal ,tight ,vtight ,extreme)',
                           default="vtight"
                           )

    my_parser.add_argument('-chrg', '--chrg',
                           metavar='chrg',
                           type=int,
                           help='Charge of the molecule',
                           default=0
                           )

    my_parser.add_argument('-uhf', '--uhf',
                           metavar='uhf',
                           type=int,
                           help='Number of unpaired electrons (default: 1 for odd charge, 0 for even charge)',
                           default=None,
                           )

    my_parser.add_argument('-clean', '--clean',
                           help='Remove all the non necesary xtb output files',
                           action='store_true',
                           )

    args = my_parser.parse_args()

    return args

if __name__ == '__main__':
    #%%
    
    args = pareser()
    
    SDF_FILE = Path(args.sdf_file)

    # parse the arguments
    # output folder
    if not args.output_folder:
        xTB_results = Path(f"{SDF_FILE.stem}_{TIMESTAMP}")
    else: 
        xTB_results = Path(f"{args.output_folder}")
    xTB_results.mkdir(exist_ok=True)
    LOG_FILE = xTB_results / f"run_xtb_{TIMESTAMP}.txt"
	
    # number of processes
    if not args.processes:
        N_CPU = multiprocessing.cpu_count() # this should not be used in a cluster
    else:
        N_CPU = args.processes
    log(f"{multiprocessing.cpu_count()}s CPUs detected")
    log(f"N_CPU: {N_CPU}")

    # xTB options
    gfn = args.gfn
    opt = args.opt
    hess = args.hess
    acc = args.acc
    clean_up = args.clean
    chrg = args.chrg
    
    uhf = args.uhf
    if uhf is None:
        if chrg % 2 == 0:
            uhf = 0
        else:
            uhf = 1

        
    log(f"Reading: {SDF_FILE}")

    t0 = time()
    df = PandasTools.LoadSDF(str(SDF_FILE), removeHs=False, molColName="mol")
    df['ID'] = df.index
    log(f"CPU time: {time() - t0:.0f}")

    

    xyz_paths = []

    t0 = time()
    log(f"Processing {SDF_FILE}")
    log("Creating directories...")
    for _, row in df.iterrows():
        if not 'name' in row.keys():
            mol_name = f"MOL{row['ID']:04d}"
        else:
            mol_name = row['name']
        
        job_name = f"{mol_name}_chrg_{chrg}_uhf_{uhf}"
        mol_folder = xTB_results / mol_name
        mol_folder.mkdir(parents=True, exist_ok=True)
        xyz_path = mol_folder / f"{mol_name}.xyz"
        Chem.MolToXYZFile(row.mol, str(xyz_path))
        xyz_paths.append(xyz_path)
    log(f"CPU time: {time() - t0:.0f}")

    commands = []
    log("Preparing xtb commands..." )
    for xyz_path in xyz_paths:
        namespace = f"{xyz_path.parent / xyz_path.stem}_CHRG_{chrg}"
        # Run only the file if it has not been processed already
        xtb_opt_xyz = f"{xyz_path.parent / xyz_path.stem}_CHRG_{chrg}"
        if Path(xtb_opt_xyz).exists(): log(f"{xtb_opt_xyz} doesn't exists."); continue
        # print(get_xtb_command(xyz_path, opt, hess, acc, chrg, uhf, clean_up))
        commands.append(get_xtb_command(xyz_path, gfn, opt, hess, acc, chrg, uhf, clean_up))
        log(f"{namespace} added to queue.")
    log(f"CPU time: {time() - t0:.0f}")

    log(f"Running xTB calculations for {len(commands)} molecules ...")
    #command = get_xtb_command("molecule.xyz", opt, hess, acc, chrg, uhf, clean_up)
    log(f"xtb command :{commands[0]}")

    t0 = time()
    with multiprocessing.Pool(N_CPU) as p:
        results = p.map(os.system, commands)
    log(f"CPU time: {time() - t0:.0f}s")

    log(f"Done! Results in {xTB_results}")
