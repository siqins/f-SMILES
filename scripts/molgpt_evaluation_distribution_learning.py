import pathlib
import numpy as np
import pandas as pd
from src.utils.metrics import get_all_metrics


def evaluate(baseline_name, gen_path_name='fsmiles', epoch=10):
    baseline_path = pathlib.Path(f'../experiments/molgpt_exps_distribution_learning/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../datasets/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    eval_results = []
    for idx in range(3):
        gen_path = f'../experiments/molgpt_exps_distribution_learning/{gen_path_name}/samples_{epoch}_{idx}.csv'

        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{epoch}_{idx}',
                              total_num=10000)
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_{epoch}_eval_results.csv', index=False)

def evaluate_smiles(epoch=10):
    print('Evaluating SMILES')
    evaluate(baseline_name='smiles', gen_path_name='smiles', epoch=epoch)

def evaluate_smiles_pe(epoch=10):
    print('Evaluating SMILES with PE')
    evaluate(baseline_name='smiles-pair-encoding', gen_path_name='smiles-pair-encoding', epoch=epoch)

def evaluate_selfies(epoch=10):
    print('Evaluating selfies')
    evaluate(baseline_name='selfies', gen_path_name='selfies', epoch=epoch)

def evaluate_group_selfies(epoch=10):
    print('Evaluating selfies with groups')
    evaluate(baseline_name='group-selfies', gen_path_name='group-selfies', epoch=epoch)

def evaluate_deep_smiles(epoch=10):
    print('Evaluating deep SMILES')
    evaluate(baseline_name='deep-smiles', gen_path_name='deep-smiles', epoch=epoch)

def evaluate_clear_smiles(epoch=10):
    print('Evaluating clear SMILES')
    evaluate(baseline_name='clear-smiles', gen_path_name='clear-smiles', epoch=epoch)

def evaluate_tsmiles(epoch=10):
    print('Evaluating TSMILES')
    evaluate(baseline_name='tsmiles', gen_path_name='tsmiles', epoch=epoch)

def evaluate_fsmiles_pair_encoding(epoch=10):
    print('Evaluating FSMILES (Pair-encoding)')
    evaluate(baseline_name='fsmiles-pair-encoding', gen_path_name='fsmiles-pair-encoding', epoch=epoch)

def evaluate_fsmiles(epoch=10):
    print('Evaluating FSMILES (Unit level)')
    evaluate(baseline_name='fsmiles', gen_path_name='fsmiles', epoch=epoch)

if __name__ == '__main__':

    epoch=5
    evaluate_smiles(epoch)
    evaluate_smiles_pe(epoch)
    evaluate_clear_smiles(epoch)
    evaluate_deep_smiles(epoch)
    evaluate_selfies(epoch)
    evaluate_group_selfies(epoch)
    evaluate_tsmiles(epoch)
    evaluate_fsmiles_pair_encoding(epoch)
    evaluate_fsmiles(epoch)
