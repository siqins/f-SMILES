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

orca = shutil.which('orca')

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
LOG_FILE = f"orca_{TIMESTAMP}.log"

def log(text):
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")

# ORCA STUFF
def write_orca_input(xyz_path:Path, charge=0, multiplicity=1, method="cam-b3lyp", basis="def2-svp", n_cpu=1, maxcore=2048, freq=False)->Path:
    """
    Write ORCA input file
    """
    namespace = f"{xyz_path.parent / xyz_path.stem}_CHRG_{charge}_UHF_{multiplicity}"
    orca_input_file = namespace + ".inp"
    coordinates = '\nO      0.000040    0.404121    0.000000\nH     -0.783057   -0.201983    0.000000\nH      0.783017   -0.202138    0.000000\n'
    with open(orca_input_file, "w") as f:
        # write geometry optimization input
        if freq:
            f.write(f"! {method} {basis} freq\n")
        else:
            f.write(f"! {method} {basis}\n")  
        f.write("# accuracy, approximations, and dispersion corrections\n") 
        f.write("! tightscf rijcosX def2/j d3bj\n") 
        f.write("# type of calculation\n") 
        f.write("! opt\n")
        f.write("# output control\n")      
        f.write("! miniprint\n")
        f.write("# calculation resources\n")
        f.write(f"%pal nprocs {n_cpu} end\n")
        f.write(f"%maxcore {maxcore}\n")
        f.write(f"%base \"{str(namespace)}\"\n")
        f.write("# type of input; charge; multiplicity; input\n")
        f.write(f"*xyzfile {charge} {multiplicity} {xyz_path}\n\n")
    return orca_input_file


def run_orca(orca_input_file:Path, n_cpu=1):
    """
    Run ORCA
    """
    # run ORCA
    print(f"{orca} {orca_input_file} > {orca_input_file}.log")
    log_file = orca_input_file.parent / f"{orca_input_file.stem}.log"
    # launch orca using subprocess and save the output in a log file
    with open(log_file, "w") as f:
        subprocess.run([orca, orca_input_file], stdout=f, stderr=f, check=True)    
    # get the namespace
    # launch orca using os.system and save the output in a log file
    # os.system(f"{orca} {orca_input_file} > {log_file}")
    namespace = orca_input_file.stem.split("_CHRG_")[0]
    return namespace


def prepare_orca_input(xyz_path, charge=0, multiplicity=1, method="cam-b3lyp", basis="def2-svp", maxcore=2048, n_cpu=1, freq=False):
    """
    Prepare ORCA input files
    """
    log(f"Preparing ORCA input files {xyz_path.stem}")
    t0 = time()
    orca_input_file = write_orca_input(xyz_path, charge=charge, multiplicity=multiplicity, method=method, basis=basis, n_cpu=n_cpu, maxcore=maxcore, freq=freq)
    log(f"CPU time {xyz_path.stem}: {time() - t0:.0f}")
    return Path(orca_input_file)

# DEALING WITH TEMPORARY FILES
def get_temp_folder():
    """
    Get the temp folder
    """
    # check if the environment variable TMPDIR is set
    if "TMPDIR" not in os.environ:
        # set the temp folder to the current working directory
        return Path.cwd() / "temp"
    # get the temp folder from the environment variable
    return Path(os.environ["TMPDIR"])

def move_xyz_to_temp_folder(xyz_path, temp_folder=None):
    """
    Move ORCA input xyz file to a temporary folder
    """
    log(f"Moving ORCA input files to a temporary folder {xyz_path.stem}")
    # check if temp folder is given
    if temp_folder is None:
        # get temp folder from environment variable
        temp_folder = get_temp_folder()
    # check if temp folder is a Path object
    if not isinstance(temp_folder, Path):
        # convert to Path object
        temp_folder = Path(temp_folder)

    t0 = time()
    # create a temporary folder
    temp_folder.mkdir(parents=True, exist_ok=True)
    # check if the target file exists
    target = temp_folder / xyz_path.name
    if target.exists():
        # remove the target file
        target.unlink()
    # move the xyz file to the temp folder
    # xyz_path = shutil.move(str(xyz_path), str(temp_folder))
    # copy the xyz file to the temp folder
    xyz_path = Path(shutil.copy(str(xyz_path), str(temp_folder)))
    log(f"CPU time {xyz_path.stem}: {time() - t0:.0f}")
    return xyz_path

def move_orca_output_to_output_dir(name_suffix, output_dir, temp_folder=None):
    """
    Move ORCA output files to the output directory
    """
    log(f"Moving ORCA output files to the output directory {name_suffix}")
    # check if temp folder is given
    if temp_folder is None:
        # get temp folder from environment variable
        temp_folder = get_temp_folder()
    # check if temp folder is a Path object
    if not isinstance(temp_folder, Path):
        # convert to Path object
        temp_folder = Path(temp_folder)
    t0 = time()
    # move the xyz file to the temp folder
    for file in temp_folder.glob(f"*{name_suffix}*"):
        # remove the target file if it exists
        # target = output_dir / file.name
        # if target.exists():
        #     target.unlink()
        # move the file to the output directory
        shutil.copy(str(file), str(output_dir))
    log(f"CPU time {name_suffix}: {time() - t0:.0f}")
    return output_dir

# Wrap the functions:
# 1. move the xyz file to the temp folder
# 2. prepare the orca input file
# 3. run orca
# 4. move the orca output files to the output directory
def run_orca_for_xyz(xyz_path, temp_folder=None, output_dir=None, **kwargs):
    """
    Run ORCA for a xyz file
    """
    log(f"Running ORCA for xyz file {xyz_path.stem}")
    t0 = time()
    src_dir = xyz_path.parent
    # move the xyz file to the temp folder
    xyz_path = move_xyz_to_temp_folder(xyz_path, temp_folder=temp_folder)
    # prepare the orca input file
    orca_input_file = prepare_orca_input(xyz_path, **kwargs)
    # run orca
    orca_input_file = run_orca(orca_input_file)
    orca_input_file = Path(orca_input_file)
    # move the orca output files to the output directory
    name_suffix = orca_input_file.stem
    output_dir = move_orca_output_to_output_dir(name_suffix, src_dir, temp_folder=temp_folder)
    log(f"CPU time {xyz_path.stem}: {time() - t0:.0f}")
    return output_dir

if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser(description="Run ORCA")
    parser.add_argument("xyz_path", type=Path, help="Path to the xyz file")
    parser.add_argument("--charge", type=int, default=0, help="Charge")
    parser.add_argument("--multiplicity", type=int, default=None, help="Multiplicity")
    parser.add_argument("--method", type=str, default="cam-b3lyp", help="Method")
    parser.add_argument("--basis", type=str, default="def2-svp", help="Basis set")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPUs")
    parser.add_argument("--maxcore", type=int, default=2048, help="Maximum RAM per core in MB")
    parser.add_argument("--freq", action="store_true", help="Run frequency calculation")
    args = parser.parse_args()
    # run orca
    if args.multiplicity is None:
        args.multiplicity = 1 if args.charge % 2 == 0 else 2
    orca_job_parameters = { "charge":args.charge, 
                            "multiplicity":args.multiplicity, 
                            "method":args.method, 
                            "basis":args.basis, 
                            "n_cpu":args.n_cpu, 
                            "maxcore":args.maxcore, 
                            "freq":args.freq}
    xyz_path = Path(args.xyz_path)
    run_orca_for_xyz(xyz_path, None, **orca_job_parameters)