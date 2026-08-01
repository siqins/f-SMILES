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

from orcarun_serial import run_orca_for_xyz

orca = shutil.which('orca')

TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
LOG_FILE = f"orca_{TIMESTAMP}.log"

def log(text):
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")

# read an sdf file and create a directory for each molecule and save the xyz file in the directory
def sdf_to_xyz(sdf_file:Path, output_dir:Path,name_suffix="", verbose=False, log=True):
    """
    Read an sdf file and create a directory for each molecule and save the xyz file in the directory

    Parameters
    ----------
    sdf_file : Path
    """
    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # read sdf file do not remove hydrogens 
    df = PandasTools.LoadSDF(str(sdf_file), molColName="mol", removeHs=False)
    # create a directory for each molecule
    xyz_paths = []
    t0 = time()
    log("Creating directories")
    for _, row in df.iterrows():
        mol_name = row['name']
        job_name = f"{mol_name}{name_suffix}"
        mol_folder = output_dir / mol_name
        mol_folder.mkdir(parents=True, exist_ok=True)
        xyz_path = mol_folder / f"{mol_name}.xyz"
        Chem.MolToXYZFile(row["mol"], str(xyz_path))
        xyz_paths.append(xyz_path)
        
    log(f"CPU time: {time() - t0:.0f}")
    return xyz_paths


if __name__ == "__main__":
    import sys    
    # create a parser
    parser = argparse.ArgumentParser(description="preapre xyz files for ORCA")
    # add arguments
    parser.add_argument("sdf_file", help="sdf file")
    parser.add_argument("-o", "--output_dir", help="output directory", default=None)
    parser.add_argument("-n", "--name_suffix", help="name suffix", default="")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--nosbatch", action="store_false", help="Run job in a single node")
    parser.add_argument("--n_tasks", type=int, default=1, help="Number of tasks")
    parser.add_argument("--delay", action="store_true", help="Delay between jobs")
    
    # parse orca arguments
    parser.add_argument("--charge", type=int, default=0, help="Charge")
    parser.add_argument("--multiplicity", type=int, default=None, help="Multiplicity")
    parser.add_argument("--method", type=str, default="cam-b3lyp", help="Method")
    parser.add_argument("--basis", type=str, default="def2-svp", help="Basis set")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPUs")
    parser.add_argument("--maxcore", type=int, default=2048, help="Maximum RAM per core in MB")
    parser.add_argument("--freq", action="store_true", help="Run frequency calculation")
    
    

    
    # parse the arguments
    args = parser.parse_args()
    
    # convert the arguments to Path objects
    sdf_file = Path(args.sdf_file)
    if not args.output_dir:
        output_dir = sdf_file.parent / f"out-{sdf_file.stem}-{TIMESTAMP}"
    else:
        output_dir = Path(f"{args.output_dir}-{TIMESTAMP}")
    if not args.multiplicity:
        args.multiplicity = 1 if args.charge % 2 == 0 else 2        
    name_suffix = args.name_suffix
    verbose = args.verbose
    # check if the input file exists
    if not sdf_file.exists():
        print(f"File {sdf_file} does not exist")
        sys.exit(1)
    # check if the output directory exists
    if not output_dir.exists():
        # create the output directory
        output_dir.mkdir(parents=True, exist_ok=True)
    # create xyz files
    xyz_paths = sdf_to_xyz(sdf_file, output_dir, name_suffix, verbose, log)
    # write orca input files
    log("Writing orca input files")
    t0 = time()
    orca_input_paths = []
    
    orca_job_parameters = { "charge":args.charge, 
                            "multiplicity":args.multiplicity, 
                            "method":args.method, 
                            "basis":args.basis, 
                            "n_cpu":args.n_cpu, 
                            "maxcore":args.maxcore, 
                            "freq":args.freq}
    
    if not args.nosbatch:
        print("Running in parallel")        
        with multiprocessing.Pool(processes=args.n_tasks) as pool:
            for i in zip(xyz_paths, [orca_job_parameters]*len(xyz_paths)):
                print(i)
            pool.starmap(run_orca_for_xyz, zip(xyz_paths, [None, None,  list(orca_job_parameters.values())]*len(xyz_paths)))
    else:
        for i, xyz_path in enumerate(xyz_paths):
            command = f"python orcarun_serial.py {xyz_path}"
            for k, v in orca_job_parameters.items():
                if k == "freq":
                    if v:
                        command += f" --{k}"
                else:
                    command += f" --{k} {v}"
            log(f"{xyz_path.stem} submitted at {datetime.now().strftime('%y%m%d_%H%M%S')}")
            sbatch= f'sbatch -n {args.n_cpu} --time=24:00:00 --mem-per-cpu={int(args.maxcore*1.25)} --tmp={8000} --wrap=\"{command}\"'
            print()
            os.system(sbatch)
            if args.delay:
                time.sleep(0.2)
                if i % 1000 == 0:
                    time.sleep(3600)
                print(f"waiting {i}")
        

    
    
