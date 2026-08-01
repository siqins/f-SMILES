
import subprocess
from collections import Counter, defaultdict

import pandas as pd

from utils.enumerate_library import PATTERNS_NAME
from utils.postprocess_xtb import *
from utils.func import *


def log(text):
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


def count_bblocks(seq_list):

    patterns_name_infos = defaultdict(list)

    for seq in seq_list:
        bblocks = seq.split('-')
        bblocks_count = Counter(bblocks)

        for name in PATTERNS_NAME:
            patterns_name_infos[name].append(bblocks_count.get(name, 0))

    return pd.DataFrame(patterns_name_infos)


if __name__ == '__main__':

    # xTB计算结果整理

    DATA_FOLDER = Path('data')
    ENUM_FOLDER = DATA_FOLDER / 'enum'
    ETKDG_FOLDER = DATA_FOLDER / 'conf-gen'
    XTB_FOLDER = DATA_FOLDER / 'xtb-calc'
    DATASET_FOLDER = DATA_FOLDER / 'dataset'

    DATASET_FOLDER.mkdir(parents=True, exist_ok=True)

    for i in XTB_FOLDER.glob('dataset_*'):
        if not i.is_dir(): continue
        command = ["python", "utils/postprocess_xtb.py",
                   "-f", str(i),
                   "-sdf", f"{ETKDG_FOLDER / 'dataset-UFFopt.sdf'}",
                   "-o", str(i)
                   ]
        print(str(i))
        subprocess.run(command)
        print(" ".join(command))

    # Postprocess of the xTB calculations
    props = []
    for i in XTB_FOLDER.rglob('*CHRG_0.xtbout.txt'):
        with open(i, 'r', encoding="MacRoman") as f:
            text = f.read()
        props.append(parse_properties_xtbout_txt(text))

    # read the output file with coordinate information
    sdf_files = list(XTB_FOLDER.glob("*.sdf"))
    df_sdf = pd.concat([PandasTools.LoadSDF(str(sdf_file), molColName="mol", removeHs=False) for sdf_file in sdf_files], ignore_index=True)

    # read the output file with detailed information
    csv_files = list(XTB_FOLDER.glob("*.csv"))
    df = pd.concat([pd.read_csv(csv_file, index_col=0) for csv_file in csv_files], ignore_index=True)

    # read the enueration csv file
    df_enum = pd.read_csv(ENUM_FOLDER / "dataset.csv", index_col=0)

    # merge the two dataframes
    df = pd.merge(df, df_enum, left_on="name", right_on="name")

    # rename columns
    df = df.rename(columns={'total energy': 'energy'})
    # all columns to lower case
    df.columns = [i.lower() for i in df.columns]

    # convert charge to str becouse PandasTools.LoadSDF convert it to str
    df['charge'] = df['charge'].astype(int)
    df_sdf['charge'] = df_sdf['charge'].astype(int)
    # concat df and df_sdf on name and charge to get the mol column
    df = df.merge(df_sdf[["name", "charge", "mol"]], on=["name", "charge"])

    # Check Imaginary Frequencies
    print(f"number of unique molecules: {df['name'].unique().shape[0]}")
    print(f"number of unique molecules with imaginary frecuencies: {df.query('imaginary_frecuencies > 0')['name'].unique().shape[0]}")

    # filter by imaginary frecuencies
    img_group = df.groupby(['name']).agg({'imaginary_frecuencies': 'sum'})
    # drop molecules with imaginary frecuencies
    df = df.query('name not in @img_group.query("imaginary_frecuencies > 0").index')

    # filter molecules with clashes using the function has_clashes2 in compas.utils
    # filter molecules with clashes
    df["has_clashes"] = df["mol"].apply(lambda x: has_clashes(x))
    print(F'Found {df["has_clashes"].sum()} molecules with clashes')

    # filter molecules with distorted bonds check_bond_lenghts in compas.utils
    # filter molecules with distorted bonds
    df["has_distorted_bonds"] = df["mol"].apply(lambda x: check_bond_lenghts(x))
    print(F'Found {df["has_distorted_bonds"].sum()} molecules with distorted bonds')


    #  Extract orbital information from the xTB calculations
    # conver df['orbital energies/ev'] from string to array
    df['orbital energies/ev'] = df['orbital energies/ev'].apply(lambda x: np.array(x.replace("[", "").replace("]", "").split(','), dtype=float))

    # get the homo index
    df['lumo_id'] = df['number of electrons'].apply(lambda x: np.ceil(x / 2)).astype(int)

    # get the energy of the homo-1, homo, lumo and lumo+1
    df['homo-1'] = df.apply(lambda x: x['orbital energies/ev'][x['lumo_id'] - 2], axis=1)
    df['homo'] = df.apply(lambda x: x['orbital energies/ev'][x['lumo_id'] - 1], axis=1)
    df['lumo'] = df.apply(lambda x: x['orbital energies/ev'][x['lumo_id']], axis=1)
    df['lumo+1'] = df.apply(lambda x: x['orbital energies/ev'][x['lumo_id'] + 1], axis=1)

    # get gap
    df = calc_gap(df)

    # get aea and aip
    df = calc_aea_aip(df)

    # Get structural descriptors
    # get molecular formula
    df['formula'] = df['mol'].apply(lambda mol: Chem.rdMolDescriptors.CalcMolFormula(mol))
    df['smiles'] = df['mol'].apply(lambda mol: Chem.MolToSmiles(mol))
    df['inchi'] = df['mol'].apply(lambda mol: Chem.MolToInchi(mol))

    # save the dataframe to a csv file
    df.drop(columns=['mol']).to_csv(DATASET_FOLDER / "compas-gfn1.csv")
    PandasTools.WriteSDF(df, str(DATASET_FOLDER / "compas-gfn1.sdf"), molColName='mol', properties=list(['name', 'charge', 'energy']))
    # pickling the dataframe
    df.to_pickle(DATA_FOLDER / "dataset" / "gfn1.pkl")

    # 统计一些原子个数
    df['nAtoms'] = [mol.GetNumAtoms() for mol in df['mol']]

    def get_hetero_atoms(mols):
        n, o, si, s, se = [], [], [], [], []
        for mol in mols:
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atom_counts = Counter(atoms)

            n.append(atom_counts.get('N', 0))
            o.append(atom_counts.get('O', 0))
            si.append(atom_counts.get('Si', 0))
            s.append(atom_counts.get('S', 0))
            se.append(atom_counts.get('Se', 0))

        return n, o, si, s, se

    # numN, numO, numSi, numS, numSe = get_hetero_atoms(df['mol'].tolist())
    # df['numN'] = numN
    # df['numO'] = numO
    # df['numSi'] = numSi
    # df['numS'] = numS
    # df['numSe'] = numSe
    # df['nHeteroAtoms'] = df.iloc[:, -5:].sum(axis=1)

    # make final dataset
    mapper = {
        'name': 'name',
        'formula': 'formula',
        'inchi': 'inchi',
        'smiles': 'smiles',
        'smi': 'smi',
        'charge': 'charge',
        'fractional occupation': 'occupancy',
        'orbital energies/ev': 'orbital_energies',
        'homo': 'homo',
        'lumo': 'lumo',
        'lumo+1': 'lumo+1',
        'homo--1': 'homo-1',
        'gap': 'gap',
        'electronic energy': 'electronic_energy',
        'total_energy': 'energy',
        'aip': 'ionization_potential aip',
        'aea': 'electron_affinity aea',
        'zpe': 'zero point energy zpe',
    }

    df = df.rename(mapper, axis=1)

    # 统计片段出现次数
    # seq_list = df['seq'].tolist()
    # seq_infos = count_bblocks(seq_list)

    # seq_infos_copy = seq_infos.copy()
    # seq_infos_columns = seq_infos_copy.columns.tolist()
    # seq_infos_columns.remove('cyclopentadiene')
    # seq_infos_columns.remove('silole')

    # seq_infos['nRings'] = seq_infos.sum(axis=1)
    # seq_infos['nAromaticRings'] = seq_infos_copy.sum(axis=1)

    columns = ['name', 'formula', 'inchi', 'smiles', 'smi', 'charge', 'occupancy', 'orbital_energies', 'homo', 'lumo', 'lumo+1', 'homo-1', 'gap',
               'electronic_energy', 'energy', 'numN', 'numO', 'numSi', 'numS', 'numSe', 'nHeteroAtoms',]
    columns = ['name', 'formula', 'inchi', 'smiles', 'smi', 'charge', 'occupancy', 'orbital_energies', 'homo', 'lumo', 'lumo+1', 'homo-1', 'gap',
               'electronic_energy', 'energy', 'ionization_potential aip', 'electron_affinity aea', 'zero point energy zpe']
    # save the dataframe to a csv file
    # pd.concat([df[columns].reset_index(), seq_infos], axis=1).query('charge == 0').to_csv(DATASET_FOLDER / "dataset.csv", index=False)

    final_data = df[columns].reset_index().query('charge == 0')
    final_data.to_csv(DATASET_FOLDER / "dataset.csv", index=False)
    # save only the 0 charged to run the dft calc
    PandasTools.WriteSDF(df.query('charge == 0'), str(DATASET_FOLDER / "dataset.sdf"), molColName='mol', properties=list(['name', 'charge', 'energy']))