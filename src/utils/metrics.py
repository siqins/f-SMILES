import os.path
import numpy as np
import pandas as pd
from typing import Union
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine as cos_distance
from fcd_torch import FCD as FCDMetric
from helper import get_mol, canonic_smiles, average_agg_tanimoto, fingerprints, compute_fragments, compute_scaffolds
from helper import calculate_pc_descriptors, continuous_kldiv, discrete_kldiv, calculate_internal_pairwise_similarities

def get_all_metrics(gen, train_set, n_jobs=1, device='cpu', total_num: int=None,
                    batch_size=512, save_path='', save_name='metrics'):
    metrics = {}

    f_valid, valid_gen = fraction_valid(gen, return_valid_mol=True, total_num=total_num)
    f_unique, unique_gen = fraction_unique(valid_gen, return_unique_mol=True)
    f_novel = fraction_novel(unique_gen, train_set)
    f_diversity = internal_diversity(unique_gen)
    kld, kld_scores = KLDivergence(valid_gen, train_set, return_all_scores=True)
    f_cycle_only_ratio = cycle_only_ratio(valid_gen)
    filters = passes_filters(valid_gen)
    metrics['f-valid'] = f_valid
    metrics['f-unique'] = f_unique
    metrics['f-novel'] = f_novel
    metrics['f-diversity'] = f_diversity
    metrics['KL Divergence'] = kld
    metrics['cycle-only-ratio'] = f_cycle_only_ratio
    metrics['filters'] = filters
    metrics = {**metrics, **kld_scores}

    ptest = compute_intermediate_statistics(train_set, n_jobs=n_jobs,
                                            device=device,
                                            batch_size=batch_size,
                                            pool=n_jobs)

    kwargs = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    fcd = FCDMetric(**kwargs)(gen=valid_gen, pref=ptest['FCD'])
    metrics['FCD'] = np.exp(-0.2 * fcd)
    metrics['SNN'] = SNNMetric(**kwargs)(gen=valid_gen, pref=ptest['SNN'])
    metrics['Frag'] = FragMetric(**kwargs)(gen=valid_gen, pref=ptest['Frag'])
    metrics['Scaf'] = ScafMetric(**kwargs)(gen=valid_gen, pref=ptest['Scaf'])

    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df.to_csv(os.path.join(save_path, f'{save_name}.csv'), index=False)
    return df


def get_all_metrics_for_goal_oriented_generation(gen, train_set, gen_prop_values, goal_prop_value=0., total_num: int=None, save_path='', save_name='metrics'):
    metrics = {}

    f_valid, valid_gen = fraction_valid(gen, return_valid_mol=True, total_num=total_num)
    f_unique, unique_gen = fraction_unique(valid_gen, return_unique_mol=True)
    f_novel, novel_gen = fraction_novel(unique_gen, train_set, return_novel_mol=True)
    f_cycle_only_ratio = cycle_only_ratio(valid_gen)
    filters = passes_filters(valid_gen)

    gen_dict = {}
    for smi, prop_value in zip(gen, gen_prop_values):
        gen_dict[canonic_smiles(smi)] = prop_value

    _, cycle_gen = cycle_only_ratio(novel_gen, return_cycle_only_mol=True)
    _, filter_gen = passes_filters(cycle_gen, return_passes_filters_mol=True)
    gen_prop_values_novel = [v for k, v in gen_dict.items() if k in filter_gen]

    topk1 = goal_oriented_top_k_score(gen_prop_values_novel, goal_prop_value, k=1)
    topk10 = goal_oriented_top_k_score(gen_prop_values_novel, goal_prop_value, k=10)
    topk100 = goal_oriented_top_k_score(gen_prop_values_novel, goal_prop_value, k=100)
    topk1000 = goal_oriented_top_k_score(gen_prop_values_novel, goal_prop_value, k=1000)

    metrics['f-valid'] = f_valid
    metrics['f-unique'] = f_unique
    metrics['f-novel'] = f_novel
    metrics['cycle-only-ratio'] = f_cycle_only_ratio
    metrics['filters'] = filters
    metrics['topk1'] = topk1
    metrics['topk10'] = topk10
    metrics['topk100'] = topk100
    metrics['topk1000'] = topk1000

    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df.to_csv(os.path.join(save_path, f'{save_name}.csv'), index=False)
    return df


def fraction_valid(smiles: list, return_valid_mol:bool=False, total_num: int=None) -> Union[float, type[float, list]]:
    '''
    :param smiles: a list of SMILES strings
    :param return_valid_mol: if return the list of valid SMILES strings
    :return:
    :usage
        smiles_list = ['CC', 'c1ccccc1']
        f_valid = fraction_valid(smiles_list)
        f_valid, valid_smiles = fraction_valid(smiles_list, return_valid_mol=True)
    '''
    gen = [canonic_smiles(smi) for smi in smiles]
    if total_num is None:
        f_valid = 1 - gen.count(None) / len(gen)
    else:
        f_valid = (len(gen) - gen.count(None)) / total_num

    if return_valid_mol:
        return f_valid, [smi for smi in gen if smi is not None]
    return f_valid

def fraction_unique(smiles: list, return_unique_mol:bool=False) -> Union[float, type[float, list]]:
    gen = set(smiles)
    f_unique = len(gen) / len(smiles)
    if return_unique_mol:
        return f_unique, list(gen)
    return f_unique

def fraction_novel(smiles: list, train: list, return_novel_mol:bool=False) -> Union[float, type[float, list]]:
    gen = set(smiles)
    train_set = set(train)
    f_novel = len(gen - train_set) / len(gen)
    if return_novel_mol:
        return f_novel, list(smiles)
    return f_novel

def goal_oriented_top_k_score(gen_prop_values, prop_value=0., k: int=1):

    gen_prop_values = [value for value in gen_prop_values if isinstance(value, int) or isinstance(value, float) and not np.isnan(value)]
    gen_prop_values = np.array(gen_prop_values)
    gen_prop_values = np.abs(gen_prop_values - prop_value)
    gen_prop_values.sort()
    top_k_props = gen_prop_values[:k]
    mean_top_k = np.mean(top_k_props)
    score = np.exp(-mean_top_k)
    return score


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()

def KLDivergence(gen, train, return_all_scores:bool=False) -> Union[float, type[float, list]]:
    pc_descriptor_subset = [
        'BertzCT',
        'MolLogP',
        'MolWt',
        'TPSA',
        'NumHAcceptors',
        'NumHDonors',
        'NumRotatableBonds',
        'NumAliphaticRings',
        'NumAromaticRings'
    ]

    d_sampled = calculate_pc_descriptors(gen, pc_descriptor_subset)
    d_chembl = calculate_pc_descriptors(train, pc_descriptor_subset)

    kldivs = {}

    # now we calculate the kl divergence for the float valued descriptors ...
    for i in range(4):
        # print(f'Calculating {pc_descriptor_subset[i]}')
        x_baseline=d_chembl[:, i]
        x_sampled = d_sampled[:, i]
        if x_baseline.var() == 0 or x_sampled.var() == 0:
            continue
        kldiv = continuous_kldiv(X_baseline=x_baseline, X_sampled=x_sampled)
        kldivs['kld-'+pc_descriptor_subset[i]] = kldiv

    # ... and for the int valued ones.
    for i in range(4, 9):
        # print(f'Calculating {pc_descriptor_subset[i]}')
        kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
        kldivs['kld-'+pc_descriptor_subset[i]] = kldiv


    chembl_sim = calculate_internal_pairwise_similarities(train)
    chembl_sim = chembl_sim.max(axis=1)

    sampled_sim = calculate_internal_pairwise_similarities(gen)
    sampled_sim = sampled_sim.max(axis=1)

    kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
    kldivs['internal_similarity'] = kldiv_int_int

    partial_scores = [np.exp(-score) for score in kldivs.values()]
    score = sum(partial_scores) / len(partial_scores)

    if return_all_scores:
        return score, kldivs

    return score


def cycle_only_ratio(smiles: list, return_cycle_only_mol:bool=False) -> Union[float, type[float, list]]:

    def check_cycle_only(smi):
        mol = Chem.MolFromSmiles(smi)
        bonds = mol.GetBonds()
        bonds_in_rings = [True if bond.IsInRing() else False for bond in bonds]

        if all(bonds_in_rings):
            return smi
        else:
            return None

    gen = [check_cycle_only(smi) for smi in smiles]
    f_cycle_only = 1 - gen.count(None) / len(gen)
    if return_cycle_only_mol:
        return f_cycle_only, [smi for smi in gen if smi is not None]
    return f_cycle_only

def passes_filters(smiles: list, return_passes_filters_mol:bool=False) -> Union[float, type[float, list]]:
    def filters(smi):
        mol = Chem.MolFromSmiles(smi)
        atoms = mol.GetAtoms()
        for atom in atoms:
            if atom.GetSymbol() != 'C':
                if len(atom.GetBonds()) != 2:
                    return None
        return smi

    gen = [filters(smi) for smi in smiles]
    f_filters = 1 - gen.count(None) / len(gen)
    if return_passes_filters_mol:
        return f_filters, [smi for smi in gen if smi is not None]
    return f_filters

def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    statistics = {}
    mols = [get_mol(smi) for smi in smiles]
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    return statistics


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])


class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


class WassersteinMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = [self.func(mol) for mol in mols]
        else:
            values = mols
        return {'values': values}

    def metric(self, pref, pgen):
        return wasserstein_distance(
            pref['values'], pgen['values']
        )


if __name__ == '__main__':

    gen = ['CCCCCCC', 'CCCC','CCCCCCCCCCCCC', 'c1ccccc1']
    train_set = pd.read_csv('../data/fused_units_clear.csv')['smiles'].tolist()

    get_all_metrics(gen, train_set)

    path = r'../data/fused_units_clear.csv'
    data = pd.read_csv(path)['smiles'].tolist()
    print(passes_filters(data))