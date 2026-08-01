import os
import random
import numpy as np
import pandas as pd
from collections import Counter
import src.fSMILES as fs
from src.baselines.sequences.transfer import standardize_smiles
from src.utils.metrics import get_all_metrics
import warnings
warnings.filterwarnings('ignore')


def reconstruction(
        unit_level: bool = False,
        random_seed: int = 42,
        n_reconstructions: int = 50,
):

    random.seed(random_seed)

    path = '../datasets/fused_units.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    print('SMILES numbers:', len(smis))

    recon_smis = []
    fsmis = []
    for idx, smi in enumerate(smis):
        fsmi = fs.SmilesToFSmiles(smi)
        toks = fs.FSmilesToTokens(fsmi, unit_level=unit_level)

        for _ in range(n_reconstructions):
            random.shuffle(toks)
            fsmi = ''.join(toks)
            fsmis.append(fsmi)

        if len(fsmis)>500 or idx == len(smis) - 1:
            batch_smis = fs.BatchSmilesFromFSmiles(fsmis, check_fsmi=True)
            for smi in batch_smis:
                stan_smi = standardize_smiles(smi)
                if stan_smi is not None:
                    recon_smis.append(stan_smi)
                    print(f'Reconstructed SMILES {idx}-{len(recon_smis)}: {stan_smi}')
            fsmis = []

    if not os.path.exists('../datasets/fused_units_reconstruction'):
        os.makedirs(f'../datasets/fused_units_reconstruction/{str(random_seed)}')

    df = pd.DataFrame(recon_smis, columns=['smiles'])
    df.to_csv(f'../datasets/fused_units_reconstruction/{str(random_seed)}/fused_units_reconstruction_origin_{str(n_reconstructions)}.csv', index=False)
    df = pd.DataFrame(list(set(recon_smis)), columns=['smiles'])
    df.to_csv(f'../datasets/fused_units_reconstruction/{str(random_seed)}/fused_units_reconstruction_{str(n_reconstructions)}.csv', index=False)

def analyze_reconstruction(
        random_seed: int = 42,
        n_reconstructions: int = 50,
):

    path = '../datasets/fused_units.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    gen_path = f'../datasets/fused_units_reconstruction/{str(random_seed)}/fused_units_reconstruction_origin_{str(n_reconstructions)}.csv'
    gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

    get_all_metrics(
        gen=gen_smis,
        train_set=smis,
        total_num=len(smis) * n_reconstructions,
        save_path=f'../datasets/fused_units_reconstruction/{str(random_seed)}',
        save_name=f'reconstruction_{str(n_reconstructions)}')


def enumeration(
        n_enum: int = 100,
        unit_level: bool = False,
        random_seed: int = 42,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    path = '../datasets/fused_units.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    frag_alphabet = []
    fsmi_toks_len = []
    for smi in smis:
        fsmi = fs.SmilesToFSmiles(smi)
        toks = fs.FSmilesToTokens(fsmi, unit_level=unit_level)
        fsmi_toks_len.append(len(toks))
        frag_alphabet.extend(toks)

    frag_alphabet = Counter(frag_alphabet)
    frag_alphabet = dict(frag_alphabet).items()
    frag_alphabet = pd.DataFrame(frag_alphabet, columns=['smiles', 'count'])
    frag_alphabet.sort_values('count', ascending=False, inplace=True)
    frag_alphabet.to_csv('../datasets/fsmiles_tokens.csv', index=False)

    fsmi_toks_len = Counter(fsmi_toks_len)
    fsmi_toks_len = dict(fsmi_toks_len).items()
    fsmi_toks_len = pd.DataFrame(fsmi_toks_len, columns=['fsmi_toks_len', 'count'])
    fsmi_toks_len.sort_values('fsmi_toks_len', ascending=True, inplace=True)
    fsmi_toks_len.to_csv('../datasets/fsmiles_tokens_len.csv', index=False)

    dist_len_toks = fsmi_toks_len['fsmi_toks_len'].tolist()
    dist_len_toks_p = fsmi_toks_len['count'].values
    dist_len_toks_p = dist_len_toks_p / sum(dist_len_toks_p)
    lens_toks = np.random.choice(dist_len_toks, size=n_enum, replace=True, p=dist_len_toks_p)

    dist_toks = frag_alphabet['smiles'].values
    dist_toks_p = frag_alphabet['count'].values
    dist_toks_p = dist_toks_p / sum(dist_toks_p)

    fsmis = []
    for idx, len_tok in enumerate(lens_toks):
        toks = np.random.choice(dist_toks, size=len_tok, replace=True, p=dist_toks_p).tolist()
        fsmi = ''.join(toks)
        fsmis.append(fsmi)

    enum_smis = fs.BatchSmilesFromFSmiles(fsmis, check_fsmi=True)

    if not os.path.exists('../datasets/fused_units_enumeration'):
        os.makedirs(f'../datasets/fused_units_enumeration/{str(random_seed)}')

    df = pd.DataFrame(enum_smis, columns=['smiles'])
    df.to_csv(f'../datasets/fused_units_enumeration/{str(random_seed)}/fused_units_enumeration_origin_{str(n_enum)}.csv', index=False)
    df = pd.DataFrame(list(set(enum_smis)), columns=['smiles'])
    df.to_csv(f'../datasets/fused_units_enumeration/{str(random_seed)}/fused_units_enumeration_{str(n_enum)}.csv', index=False)


def analyze_enumeration(
        random_seed: int = 42,
        n_enum: int = 50,
):

    path = '../datasets/fused_units.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    gen_path = f'../datasets/fused_units_enumeration/{str(random_seed)}/fused_units_enumeration_origin_{str(n_enum)}.csv'
    gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

    get_all_metrics(gen=gen_smis, train_set=smis,
                    save_path=f'../datasets/fused_units_enumeration/{str(random_seed)}',
                    save_name=f'enumeration_{str(n_enum)}')

def organize(
        organize_name: str='reconstruction',
        num_list: list=[5, 10, 20, 30, 50],
):

    details = []
    details_std = []
    for n_reconstructions in num_list:
        data = []
        for seed in [0, 1, 2]:
            path = f'../datasets/fused_units_{organize_name}/{str(seed)}/{organize_name}_{str(n_reconstructions)}.csv'
            data_seed = pd.read_csv(path)
            data.append(data_seed['Value'].values)
        data = np.array(data).T
        std_data = data.std(axis=-1)
        data = data.mean(axis=-1)
        details.append(data)
        details_std.append(std_data)

    details = np.array(details)
    details_std = np.array(details_std)

    columns = data_seed['Metric'].tolist()
    details = pd.DataFrame(details, columns=columns)
    details.insert(0, f'n_{organize_name}', num_list)
    details_std = pd.DataFrame(details_std, columns=columns)
    details_std.insert(0, f'n_{organize_name}', num_list)
    save_path = rf'../datasets/fused_units_{organize_name}'
    details.to_csv(os.path.join(save_path, 'details.csv'), index=False)
    details_std.to_csv(os.path.join(save_path, 'details_std.csv'), index=False)


def prepare_generated_mol_dataset():
    smis = []

    for idx in [0, 1, 2]:
        path = rf'../datasets/fused_units_enumeration/{str(idx)}/fused_units_enumeration_origin_10000.csv'
        data = pd.read_csv(path)['smiles'].tolist()
        smis.extend(data)

        path = rf'../datasets/fused_units_reconstruction/{str(idx)}/fused_units_reconstruction_origin_50.csv'
        data = pd.read_csv(path)['smiles'].tolist()
        smis.extend(data)

    path = r'../data/fused_units_clear.csv'
    data = pd.read_csv(path)['smiles'].tolist()
    smis.extend(data)

    data = pd.DataFrame(list(set(smis)), columns=['smiles'])
    data.to_csv(f'../datasets/fused_units_generated.csv', index=False)


if __name__ == '__main__':

    # reconstructions
    for n_reconstructions in [5, 10, 20, 30, 50]:
        for seed in [0, 1, 2]:
            reconstruction(n_reconstructions=n_reconstructions, random_seed=seed)

    # analysis for reconstructions
    for n_reconstructions in [5, 10, 20, 30, 50]:
        for seed in [0, 1, 2]:
            print(f'Reconstruction for {n_reconstructions} seed: {seed}')
            analyze_reconstruction(n_reconstructions=n_reconstructions, random_seed=seed)

    # organize the analysis results
    organize(organize_name='reconstruction')

    # enumeration
    for n_enum in [500, 1000, 2000, 3000, 5000, 10000]:
        for seed in [0, 1, 2]:
            enumeration(n_enum=n_enum, random_seed=seed)

    # analysis for enumeration
    for n_enum in [500, 1000, 2000, 3000, 5000, 10000]:
        for seed in [0, 1, 2]:
            print(f'Enumeration for {n_enum} seed: {seed}')
            analyze_enumeration(n_enum=n_enum, random_seed=seed)

    # organize the analysis results
    organize(organize_name='enumeration', num_list=[500, 1000, 2000, 3000, 5000, 10000])

    # prepare generated molecules
    prepare_generated_mol_dataset()
