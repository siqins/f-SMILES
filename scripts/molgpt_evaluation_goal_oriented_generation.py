import pathlib
import numpy as np
import pandas as pd
from src.utils.metrics import get_all_metrics_for_goal_oriented_generation


def evaluate(baseline_name, prop_name, gen_path_name='ChemicalVAE', epoch=5):
    baseline_path = pathlib.Path(f'../experiments/molgpt_exps_goal_oriented_generation/{baseline_name}/{props_mapper[prop_name]}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../data/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()
    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    raw_data_path = r'../data/fused_units_generated_calculated.csv'
    prop_data = pd.read_csv(raw_data_path)[prop_name].values
    max_prop_data = np.max(prop_data)
    if prop_name == 'energy':
        max_prop_data = max_prop_data / 10

    eval_results = []
    for idx in range(3):
        gen_path = f'../experiments/molgpt_exps_goal_oriented_generation/{gen_path_name}/{props_mapper[prop_name]}/samples_{epoch}_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()
        gen_prop_values = pd.read_csv(gen_path)[props_mapper[prop_name]].values
        if prop_name == 'energy':
            gen_prop_values = gen_prop_values / 10
        gen_prop_values = gen_prop_values.tolist()

        res = get_all_metrics_for_goal_oriented_generation(gen=gen_smis, train_set=smis,
                                                           gen_prop_values=gen_prop_values,
                                                           goal_prop_value=max_prop_data,
                                                           total_num=10000,
                                                           save_path=baseline_path,
                                                           save_name=f'{baseline_name}_{epoch}_{idx}'
                                                           )
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_{epoch}_eval_results.csv', index=False)

def evaluate_smiles(prop_name, epoch):
    print('Evaluating SMILES')
    evaluate(baseline_name='smiles', prop_name=prop_name, gen_path_name='smiles', epoch=epoch)

def evaluate_smiles_pe(prop_name, epoch):
    print('Evaluating SMILES with PE')
    evaluate(baseline_name='smiles-pair-encoding', prop_name=prop_name, gen_path_name='smiles-pair-encoding', epoch=epoch)

def evaluate_selfies(prop_name, epoch):
    print('Evaluating selfies')
    evaluate(baseline_name='selfies', prop_name=prop_name, gen_path_name='selfies', epoch=epoch)

def evaluate_group_selfies(prop_name, epoch):
    print('Evaluating selfies with groups')
    evaluate(baseline_name='group-selfies', prop_name=prop_name, gen_path_name='group-selfies', epoch=epoch)

def evaluate_deep_smiles(prop_name, epoch):
    print('Evaluating deep SMILES')
    evaluate(baseline_name='deep-smiles', prop_name=prop_name, gen_path_name='deep-smiles', epoch=epoch)

def evaluate_clear_smiles(prop_name, epoch):
    print('Evaluating clear SMILES')
    evaluate(baseline_name='clear-smiles', prop_name=prop_name, gen_path_name='clear-smiles', epoch=epoch)

def evaluate_tsmiles(prop_name, epoch):
    print('Evaluating TSMILES')
    evaluate(baseline_name='tsmiles', prop_name=prop_name, gen_path_name='tsmiles', epoch=epoch)

def evaluate_fsmiles_pair_encoding(prop_name, epoch):
    print('Evaluating FSMILES (Pair-encoding)')
    evaluate(baseline_name='fsmiles-pair-encoding', prop_name=prop_name, gen_path_name='fsmiles-pair-encoding', epoch=epoch)

def evaluate_fsmiles(prop_name, epoch):
    print('Evaluating FSMILES (Unit level)')
    evaluate(baseline_name='fsmiles', prop_name=prop_name, gen_path_name='fsmiles', epoch=epoch)

if __name__ == '__main__':

    prop_names = ['lumo', 'gap', 'energy', 'ionization_potential aip', 'electron_affinity aea']
    props_mapper = {
        'lumo': 'LUMO',
        'gap': 'HLG',
        'energy': 'Erel',
        'ionization_potential aip': 'IP',
        'electron_affinity aea': 'EA',
    }

    epoch = 10
    for prop_name in prop_names:
        evaluate_smiles(prop_name, epoch=epoch)
        evaluate_smiles_pe(prop_name, epoch=epoch)
        evaluate_clear_smiles(prop_name, epoch=epoch)
        evaluate_deep_smiles(prop_name, epoch=epoch)
        evaluate_selfies(prop_name, epoch=epoch)
        evaluate_group_selfies(prop_name, epoch=epoch)
        evaluate_tsmiles(prop_name, epoch=epoch)
        evaluate_fsmiles_pair_encoding(prop_name, epoch=epoch)
        evaluate_fsmiles(prop_name, epoch=epoch)
