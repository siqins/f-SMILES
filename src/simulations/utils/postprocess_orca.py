# Import required modules
import re
from datetime import datetime
from pathlib import Path

# Import RDKit modules
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Geometry import Point3D

# Import Pandas and Numpy for data processing
import pandas as pd
import numpy as np

import argparse
namespace = ""

# Set the timestamp for logging
TIMESTAMP = datetime.now().strftime("%y%m%d_%H%M%S")
# Define the log file path
LOG_FILE = "log.txt"

def log(text):
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


# DATA EXTRATION FUNCTIONS
    
def get_orcaout_text(log_out: str) -> dict:
    """
    Parses the ORCA log output file and returns a dictionary with the results.

    Args:
        log_out (str): The path to the ORCA log output file.

    Returns:
        dict: A dictionary with the parsed properties.
    """
    with open(log_out) as f:
        orca_log = f.read()
    data = parse_properties_orcalog_txt(orca_log)
    call_info = get_callinfo_json(orca_log)
    return {**data,**call_info} 


def get_callinfo_json(orca_log: str)-> dict:
    """
    Extracts charge, multiplicity, and other call information from the ORCA log file.

    Args:
        orca_log (str): The path to the ORCA log file.

    Returns:
        dict: A dictionary with call information.
    """
    try:
        # chrg = re.findall("Total Charge\s+Charge\s+\.{4}\s+(-?\d+)", orca_log)[0]
        chrg = re.findall("\*xyzfile\s+(-?\d)\s+-?\d", orca_log)[0]
    except:
        chrg = np.nan

    try:
        # chrg = re.findall("Total Charge\s+Charge\s+\.{4}\s+(-?\d+)", orca_log)[0]
        mult = re.findall("\*xyzfile\s+-?\d\s+(-?\d)", orca_log)[0]
    except:
        mult = np.nan

    acc = np.nan
    data = {"charge": chrg,
            "acc":acc,
            "mult":mult,}
    return data 

def parse_properties_orcalog_txt(orca_log: str) -> dict:
    """Parse orca log file and return a dictionary with the results.
    Args: 
        orca_log (str): The path to the orca log file.
    Returns:
        dict: A dictionary with the results.
        
    Example:
        >>> orca_parser("orca.log")
        {'calculation_completed': True,
            'convergence': True,
            'convergence_energy_change': '1.0E-06',
            ...
        }
        
    """
    # Define a dictionary to store the parsed properties
    data = {}

    try:
        data['calculation_completed'] = bool(re.findall(r"ORCA TERMINATED NORMALLY", orca_log))
    except:
        data['calculation_completed'] = False    
    try:
        data['convergence'] = bool(re.findall(r"THE OPTIMIZATION HAS CONVERGED", orca_log))
    except:
        data['convergence'] = np.nan
    try:
        data['convergence_energy_change'] = re.findall(r"Energy change\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+(\w+)", orca_log)[-1]
    except:
        data['convergence_energy_change'] = np.nan
    try:
        data['convergence_rms_gradient'] = re.findall(r"RMS gradient\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+(\w+)", orca_log)[-1]
    except:
        data['convergence_rms_gradient'] = np.nan
    try:
        data['convergence_max_gradient'] = re.findall(r"MAX gradient\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+(\w+)", orca_log)[-1]
    except:
        data['convergence_max_gradient'] = np.nan
    try:
        data['convergence_rms_step'] = re.findall(r"RMS step\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+(\w+)", orca_log)[-1]
    except:
        data['convergence_rms_step'] = np.nan
    try:
        data['convergence_max_step'] = re.findall(r"MAX step\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\s+(\w+)", orca_log)[-1]
    except:
        data['convergence_max_step'] = np.nan
    try:
        data['energy'] = re.findall(r"FINAL SINGLE POINT ENERGY\s*(\S+)", orca_log)    
    except:
        data['energy'] = np.nan
    try:
        data['energy'] = float(data['energy'][-1])
    except:
        data['energy'] = np.nan
    try:
        data['dispersion_correction'] = re.findall(r"Dispersion correction\s*(-?\d+\.\d+)", orca_log)
    except:
        data['dispersion_correction'] = np.nan
    try:
        data['dispersion_correction'] = float(data['dispersion_correction'][-1])
    except:
        data['dispersion_correction'] = np.nan
    try:
        data['dipole_moment'] = re.findall(r"Total Dipole Moment\s+:\s+(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)", orca_log)
    except:
        data['dipole_moment'] = np.nan
    try:
        data['dipole_moment'] = [float(i) for i in data['dipole_moment'][-1]]
    except:
        data['dipole_moment'] = np.nan
    try:
        data['total_scf_hessian_time'] = re.findall(r"Total SCF Hessian time:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*min\s*(\d+)\s*sec", orca_log)
    except:
        data['total_scf_hessian_time'] = np.nan
    try:
        total_scf_hessian_time = [int(i) for i in total_scf_hessian_time[-1]]
        data['total_scf_hessian_time'] = total_scf_hessian_time[0]*24*60*60 + total_scf_hessian_time[1]*60*60 + total_scf_hessian_time[2]*60 + total_scf_hessian_time[3]
    except:
        data['total_scf_hessian_time'] = np.nan
    try:
        electronic_energy = re.findall(r"Electronic energy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['electronic_energy'] = float(electronic_energy[-1])
    except:
        data['electronic_energy'] = np.nan
    try:
        zero_point_energy = re.findall(r"Zero point energy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['zero_point_energy'] = float(zero_point_energy[-1])
    except:
        data['zero_point_energy'] = np.nan
    try:
        thermal_vibrational_correction = re.findall(r"Thermal vibrational correction\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['thermal_vibrational_correction'] = float(thermal_vibrational_correction[-1])
    except:
        data['thermal_vibrational_correction'] = np.nan
    try:
        thermal_rotational_correction = re.findall(r"Thermal rotational correction\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['thermal_rotational_correction'] = float(thermal_rotational_correction[-1])
    except:
        data['thermal_rotational_correction'] = np.nan
    try:
        thermal_translational_correction = re.findall(r"Thermal translational correction\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['thermal_translational_correction'] = float(thermal_translational_correction[-1])
    except:
        data['thermal_translational_correction'] = np.nan
    try:
        total_thermal_energy = re.findall(r"Total thermal energy\s+(-?\d+\.\d+)", orca_log)
        data['total_thermal_energy'] = float(total_thermal_energy[-1])
    except:
        data['total_thermal_energy'] = np.nan
    try:
        total_thermal_correction = re.findall(r"Total thermal correction\s+(-?\d+\.\d+)", orca_log)
        data['total_thermal_correction'] = float(total_thermal_correction[-1])
    except:
        data['total_thermal_correction'] = np.nan
    try:
        non_thermal_correction = re.findall(r"Non-thermal \(ZPE\) correction\s+(-?\d+\.\d+)", orca_log)
        data['non_thermal_correction'] = float(non_thermal_correction[-1])
    except:
        data['non_thermal_correction'] = np.nan
    try:
        total_enthalpy = re.findall(r"Total Enthalpy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['total_enthalpy'] = float(total_enthalpy[-1])
    except:
        data['total_enthalpy'] = np.nan
    try:
        thermal_enthalpy_correction = re.findall(r"Thermal Enthalpy correction\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['thermal_enthalpy_correction'] = float(thermal_enthalpy_correction[-1])
    except:
        data['thermal_enthalpy_correction'] = np.nan
    try:
        total_free_energy = re.findall(r"Total free energy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['total_free_energy'] = float(total_free_energy[-1])
    except:
        data['total_free_energy'] = np.nan
    try:
        electronic_entropy = re.findall(r"Electronic entropy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['electronic_entropy'] = float(electronic_entropy[-1])
    except:
        data['electronic_entropy'] = np.nan
    try:
        vibrational_entropy = re.findall(r"Vibrational entropy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['vibrational_entropy'] = float(vibrational_entropy[-1])
    except:
        data['vibrational_entropy'] = np.nan
    try:
        rotational_entropy = re.findall(r"Rotational entropy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['rotational_entropy'] = float(rotational_entropy[-1])
    except:
        data['rotational_entropy'] = np.nan
    try:
        translational_entropy = re.findall(r"Translational entropy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['translational_entropy'] = float(translational_entropy[-1])
    except:
        data['translational_entropy'] = np.nan
    try:
        total_entropy_term = re.findall(r"Final entropy term\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['total_entropy_term'] = float(total_entropy_term[-1])
    except:
        data['total_entropy_term'] = np.nan
    try:
        final_gibbs_free_energy = re.findall(r"Final Gibbs free energy\s+\.{3}\s+(-?\d+\.\d+)", orca_log)
        data['final_gibbs_free_energy'] = float(final_gibbs_free_energy[-1])
    except:
        data['final_gibbs_free_energy'] = np.nan
    try:
        total_run_time = re.findall(r"TOTAL RUN TIME:\s+(\d+)\s+days\s+(\d+)\s+hours\s+(\d+)\s+minutes\s+(\d+)\s+seconds", orca_log)
        data['total_run_time'] = int(total_run_time[0][0])*24*60*60 + int(total_run_time[0][1])*60*60 + int(total_run_time[0][2])*60 + int(total_run_time[0][3])
    except:
        data['total_run_time'] = np.nan
    try:
        nuclear_repulsion = re.findall(r"Nuclear Repulsion\s+:\s+(-?\d+\.\d+)", orca_log)
        data['nuclear_repulsion'] = float(nuclear_repulsion[-1])
    except:
        data['nuclear_repulsion'] = np.nan
    try:
        data['electronic_energy'] = re.findall(r"Electronic Energy\s+:\s+(-?\d+\.\d+)", orca_log)
    except:
        data['electronic_energy'] = np.nan
    try:
        data['electronic_energy'] = float(electronic_energy[-1])
    except:
        data['electronic_energy'] = np.nan
    try:
        one_electron_energy = re.findall(r"One Electron Energy:\s+(-?\d+\.\d+)", orca_log)
        data['one_electron_energy'] = float(one_electron_energy[-1])
    except:
        data['one_electron_energy'] = np.nan
    try:
        two_electron_energy = re.findall(r"Two Electron Energy:\s+(-?\d+\.\d+)", orca_log)
        data['two_electron_energy'] = float(two_electron_energy[-1])
    except:
        data['two_electron_energy'] = np.nan
    try:
        max_cosx_asymmetry = re.findall(r"Max COSX asymmetry\s+:\s+(-?\d+\.\d+)", orca_log)
        data['max_cosx_asymmetry'] = float(max_cosx_asymmetry[-1])
    except:
        data['max_cosx_asymmetry'] = np.nan
    try:
        potential_energy = re.findall(r"Potential Energy\s+:\s+(-?\d+\.\d+)", orca_log)
        data['potential_energy'] = float(potential_energy[-1])
    except:
        data['potential_energy'] = np.nan
    try:
        kinetic_energy = re.findall(r"Kinetic Energy\s+:\s+(-?\d+\.\d+)", orca_log)
        data['kinetic_energy'] = float(kinetic_energy[-1])
    except:
        data['kinetic_energy'] = np.nan
    try:
        data['virial_ratio'] = float(re.findall(r"Virial Ratio\s+:\s+(-?\d+\.\d+)", orca_log)[-1])
    except:
        data['virial_ratio'] = np.nan
    try:
        data['spin_contamination'] = float(re.findall(r"Deviation\s+:\s+(-?\d+\.\d+)", orca_log)[-1])
    except:
        data['spin_contamination'] = np.nan
    try:
        data['mulliken_charges'] = re.findall(r"MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS(.*?)Sum of atomic charges", orca_log, re.DOTALL)
    except:
        data['mulliken_charges'] = np.nan
    try:
        data['mulliken_charges'] = [float(re.findall(r"\d+ \w+ :\s*([-]?\d+\.\d+)", i)[0]) for i in mulliken_charges[0].split("\n")[2:-1]]
    except:
        data['mulliken_charges'] = np.nan
    try:
        data['loewdin_charges'] = re.findall(r"LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS(.*?)\n{2}", orca_log, re.DOTALL)
    except:
        data['loewdin_charges'] = np.nan
    try:
        data['loewdin_charges'] = [float(re.findall(r"\d+ \w+ :\s*([-]?\d+\.\d+)", i)[0]) for i in loewdin_charges[0].split("\n")[2:]]
    except:
        data['loewdin_charges'] = np.nan
    try:
        data['mayer_popuation_lines'] = re.findall(r"ATOM\s+NA\s+ZA\s+QA\s+VA\s+BVA\s+FA(.*?)\n{2}", orca_log, re.DOTALL)
    except:
        data['mayer_popuation_lines'] = np.nan
    try:
        data['m'] = re.findall("\s+\d+\s+\w+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",mayer_popuation_lines[0], re.DOTALL)
    except:
        data['m'] = np.nan
    try:
        data['mayer_popuation_analysis'] = np.array(m, dtype=float)
    except:
        data['mayer_popuation_analysis'] = np.nan
    try:
        data['NA'] = data['mayer_popuation_analysis'][:,0]
    except:
        data['NA'] = np.nan
    try:
        data['ZA'] = data['mayer_popuation_analysis'][:,1]
    except:
        data['ZA'] = np.nan
    try:
        data['QA'] = data['mayer_popuation_analysis'][:,2]
    except:
        data['QA'] = np.nan
    try:
        data['VA'] = data['mayer_popuation_analysis'][:,3]
    except:
        data['VA'] = np.nan
    try:
        data['BVA'] = data['mayer_popuation_analysis'][:,4]
    except:
        data['BVA'] = np.nan
    try:
        data['FA'] = data['mayer_popuation_analysis'][:,5]
    except:
        data['FA'] = np.nan
    try:
        spectrum = parse_freq(orca_log)
    except:
        spectrum = {"freq":np.nan, "intensity":np.nan, "imaginary freq":np.nan}

    orbitals_energies = extract_orbitals_energies(orca_log)
    data = {**data, **spectrum, **orbitals_energies}
    return data

def extract_orbitals_energies(orca_log: str) -> dict:
    """
    Extracts orbital energies and related information from an ORCA log file.

    Args:
        orca_log (str): Content of the ORCA log file.

    Returns:
        dict: Dictionary containing the extracted orbital energies and related information.
    """

    orbitals_energies = {}
    # Check if spin-up or spin-down orbitals are present
    try:
        if "SPIN UP" in orca_log:
            orbital_energy_lines = re.findall(r"ORBITAL ENERGIES.*?SPIN UP ORBITALS(.*?)SPIN DOWN ORBITALS(.*?)\n{2}", orca_log, re.DOTALL)
            orbitals_energies = {}
            prefixes = ["_spin_up", "_spin_down"]
            non_prefixed = [""]
            
        else:
            orbital_energy_lines = re.findall(r"ORBITAL ENERGIES\n----------------\n(.*?)\n{2}", orca_log, re.DOTALL)
            orbital_energy_lines = [[block] for block in orbital_energy_lines] # make it a list so it is in the same type of the case above
            prefixes = [""]
            non_prefixed = ["_spin_up", "_spin_down"]
        
        # Extract orbital energies
        for spin_orbital, prefix in zip(orbital_energy_lines[-1], prefixes):
            m = re.findall("\s+\d+\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",spin_orbital, re.DOTALL)
            orbitals_data = np.array(m, dtype=float)
            occ = np.where(orbitals_data[:,0] != 0)[0]
            n_occupied = occ.shape[0]
            homo_idx = occ[-1]
            orbitals_energies[f"orbital_energies{prefix}"] = orbitals_data[:n_occupied+10,2]
            orbitals_energies[f"orbital_population{prefix}"] = orbitals_data[:n_occupied+10,0]

            orbitals_energies[f"homo{prefix}"]   = orbitals_data[:n_occupied+10,2][homo_idx]
            orbitals_energies[f"lumo{prefix}"]   = orbitals_data[:n_occupied+10,2][homo_idx + 1]
            orbitals_energies[f"homo-1{prefix}"] = orbitals_data[:n_occupied+10,2][homo_idx - 1]
            orbitals_energies[f"homo-2{prefix}"] = orbitals_data[:n_occupied+10,2][homo_idx - 2]
            orbitals_energies[f"lumo+1{prefix}"] = orbitals_data[:n_occupied+10,2][homo_idx + 2]
            orbitals_energies[f"lumo+2{prefix}"] = orbitals_data[:n_occupied+10,2][homo_idx + 3]
            # HOMO-LUMO gap/eV
            orbitals_energies[f"HOMO-LUMO gap/eV{prefix}"] = orbitals_energies[f"lumo{prefix}"] - orbitals_energies[f"homo{prefix}"]

        # Set NaN values for non-existent orbitals
        for prefix in non_prefixed:
            orbitals_energies[f"orbital_energies{prefix}"] = np.nan
            orbitals_energies[f"orbital_population{prefix}"] = np.nan

            orbitals_energies[f"homo{prefix}"]   = np.nan
            orbitals_energies[f"lumo{prefix}"]   = np.nan
            orbitals_energies[f"homo-1{prefix}"] = np.nan
            orbitals_energies[f"homo-2{prefix}"] = np.nan
            orbitals_energies[f"lumo+1{prefix}"] = np.nan
            orbitals_energies[f"lumo+2{prefix}"] = np.nan
            orbitals_energies[f"HOMO-LUMO gap/eV{prefix}"] = np.nan

    except:
        non_prefixed = ["_spin_up", "_spin_down", ""]
        for prefix in non_prefixed:
            orbitals_energies[f"orbital_energies{prefix}"] = np.nan
            orbitals_energies[f"orbital_population{prefix}"] = np.nan

            orbitals_energies[f"homo{prefix}"]   = np.nan
            orbitals_energies[f"lumo{prefix}"]   = np.nan
            orbitals_energies[f"homo-1{prefix}"] = np.nan
            orbitals_energies[f"homo-2{prefix}"] = np.nan
            orbitals_energies[f"lumo+1{prefix}"] = np.nan
            orbitals_energies[f"lumo+2{prefix}"] = np.nan
            orbitals_energies[f"HOMO-LUMO gap/eV{prefix}"] = np.nan
    orbitals_energies
    return orbitals_energies

def parse_freq(orca_log: str) -> dict:
    """
    Parse the frequencies from the ORCA log file.

    Args:
        orca_log (str): Content of the ORCA log file.

    Returns:
        dict: Dictionary containing the parsed frequency data.
    """
    # -----------
    # IR SPECTRUM
    # -----------

    #  Mode   freq       eps      Int      T**2         TX        TY        TZ
    #        cm**-1   L/(mol*cm) km/mol    a.u.
    # ----------------------------------------------------------------------------
    #   6:     69.53   0.000017    0.09  0.000077  ( 0.008420 -0.002106  0.001180)
    #   7:     74.09   0.000066    0.33  0.000277  (-0.015893  0.004354 -0.002423)
    #   8:    160.98   0.000097    0.49  0.000189  (-0.013129  0.003664 -0.001668)
    #   9:    189.39   0.000056    0.28  0.000092  (-0.000165  0.003684  0.008857)
    #  10:    215.75   0.000363    1.84  0.000526  (-0.021919  0.005891 -0.003250)
    #  11:    313.44   0.000196    0.99  0.000195  ( 0.013381 -0.003544  0.001852)
    #  12:    328.76   0.000538    2.72  0.000511  (-0.001996 -0.015787 -0.016053)
    #  13:    345.52   0.000052    0.26  0.000047  (-0.006535  0.001689 -0.001095)
    #  14:    373.34   0.000015    0.07  0.000012  (-0.000387 -0.002485 -0.002408)
    # parse the IR spectrum data

    spectrum = {}

    if "IR SPECTRUM" in orca_log:
        # Extract the IR spectrum data
        ir_spectrum = re.findall(r"IR SPECTRUM.*?-\n(.*?)\n{2}", orca_log, re.DOTALL)[0]
        m = re.findall("\s+\d+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",ir_spectrum, re.DOTALL)
        spectrum_data = np.array(m, dtype=float)
        spectrum[f"freq"] = spectrum_data[:,0]
        spectrum[f"intensity"] = spectrum_data[:,2]
        spectrum[f"imaginary freq"] = spectrum_data[spectrum_data[:,0]<0,0]
    else:
        # Set NaN values if IR spectrum data is not found
        spectrum[f"freq"] = np.nan
        spectrum[f"intensity"] = np.nan
        spectrum[f"imaginary freq"] = np.nan
    return spectrum

# COORDINATES RELATED FUNCTIONS

def get_opt_coords(xyz_file: str) -> str:
    """
    Get the optimized coordinates from an XYZ file.

    Args:
        xyz_file (str): Path to the XYZ file.

    Returns:
        str: Contents of the XYZ file.
    """
    with open(xyz_file) as f:
        xyz_block = f.read()
    return xyz_block


def update_coordinate(mol: Chem.rdchem.Mol,
                    xyz_block: str,
                    verbose: bool = False) -> Chem.rdchem.Mol:
    """
    Update the coordinates of a molecule using the provided XYZ block.

    Args:
        mol (Chem.rdchem.Mol): RDKit molecule object.
        xyz_block (str): XYZ block containing the new coordinates.
        verbose (bool, optional): If True, print verbose output. Defaults to False.

    Returns:
        Chem.rdchem.Mol: Updated RDKit molecule object.
    """
    xyz = xyz_block.split("\n")
    mol = Chem.rdchem.RWMol(mol)
    conf = mol.GetConformer()

    for i in range(mol.GetNumAtoms()):
        a, x, y, z = xyz[2:][i].split()
        x, y, z = float(x), float(y), float(z)
        conf.SetAtomPosition(i, Point3D(x,y,z))

        if verbose:
            log(a, mol.GetAtomWithIdx(i).GetSymbol())
            log(i, x,y,z)
    return mol

########################################################################
# CLI
########################################################################

def pareser():
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    my_parser = argparse.ArgumentParser(prog='orca_postprocess',
                                        usage='%(prog)s [options]',
                                        description='Diggest the result of the xTB batch optimization.')

    my_parser.add_argument('-f','--folder',
                            metavar='orca_results_folder',
                            type=str,
                            help='The folder containing all the ORCA results.',
                            action='store',
                            default=None
                            )

    my_parser.add_argument('-sdf', '--sdf_file',
                            metavar='sdf_file',
                            type=str,
                            help='The input SDF file.',
                            )

    my_parser.add_argument('-o', '--namespace',
                           metavar='namespace',
                           type=str,
                           help='The output files namespace.',
                           default=f"out-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                           )
    
    args = my_parser.parse_args()

    return args

if __name__ == '__main__':
    args = pareser()
    
    if not args.folder or not args.sdf_file:
        raise ValueError("Missing arguments")
    
    orca_results_folder = Path(args.folder)
    input_sdf_file = Path(args.sdf_file)
    namespace = args.namespace

    if not orca_results_folder.exists():
        # get the folder from the sdf file and the folders in the current directory
        sdf_folder = input_sdf_file.parent
        folders = [x for x in sdf_folder.iterdir() if x.is_dir()]
        found = False
        for folder in folders:
            if folder.name.startswith(input_sdf_file.stem):
                orca_results_folder = folder
                found = True
                break
        if not found:
            raise ValueError("Could not find the folder with the orca results")

    csv_out = f"{namespace}-{TIMESTAMP}.csv"
    LOG_FILE = f"{namespace}-{TIMESTAMP}.log"

    log(f"reading {input_sdf_file}")
    df = PandasTools.LoadSDF(input_sdf_file.__str__(), removeHs=False, molColName="mol")
    
    data_list = {}

    log("Parsing results")
    for idx, folder in enumerate(orca_results_folder.iterdir()): 
        # log(folder.name)

        mol_name = folder.name
        chrgs = [-1,0,1]
        for chrg in chrgs:
            name_space = f"{mol_name}_CHRG_{chrg}_UHF_{chrg % 2 + 1}"
            log(f"{name_space}")

            
            log_file = folder / (name_space + ".log")
            opt_file = folder / (name_space + ".opt")
            xyz_file = folder / (name_space + ".xyz")
            
            
            if not log_file.exists(): log("Missing txt file: " + str(log_file)); continue
            if not opt_file.exists(): log("Missing opt file: " + opt_file.name); continue
            if not xyz_file.exists(): log("Missing xyz file: " + xyz_file.name); continue

            try:
                data_from_txt = get_orcaout_text(log_file)
            except:
                log(f"Error parsing {log_file}")
            
            xyz_block = get_opt_coords(xyz_file)
            try:
                mol_inp = df.query(f"name == '{mol_name}'").mol.iloc[0]
            except  IndexError:
                log(f"Input coordinates not found: '{mol_name}'")
            
            mol_opt = update_coordinate(mol_inp, xyz_block)

            data = {"mol":mol_opt,
                    "name":mol_name,
                    "charge":chrg,
                    **data_from_txt}
            if not data_list:
                for key, value in data.items():
                    data_list[key] = [value]
            else:
                for key, value in data.items():
                    data_list[key].append(value)

    # Saving the data
    df_out = pd.DataFrame(data_list)
    
    log("Calculating smiles\n\n")
    df_out["smiles"] = df_out.apply(lambda x: Chem.MolToSmiles(x.mol),axis=1)
    
    csv_out = f"{namespace}-{TIMESTAMP}.csv"
    sdf_out = f"{namespace}-{TIMESTAMP}.sdf"
    sdf_out_full = f"{namespace}-FULL-{TIMESTAMP}.sdf"
    
    log("Writing csv file")
    df_out.drop("mol",axis=1).to_csv(csv_out)
    
    log("Writing sdf file")
    PandasTools.WriteSDF(df_out, sdf_out,
                        properties=["mol", "name", "charge", 
                        "energy"],
                        molColName="mol")
    