import pathlib
import numpy as np
import pandas as pd
from src.utils.metrics import get_all_metrics


def evaluate(baseline_name, gen_path_name='ChemicalVAE'):
    baseline_path = pathlib.Path(f'../experiments/baselines_evaluation/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../datasets/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    eval_results = []
    for idx in range(3):
        gen_path = f'../src/baselines/generative_models/{gen_path_name}/experiments/samples_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{idx}.csv',
                              total_num=10000)
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_eval_results.csv', index=False)


def evaluate_chemical_vae():

    baseline_name = 'chemical_vae'
    baseline_path = pathlib.Path(f'../experiments/baselines_evaluation/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../datasets/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    eval_results = []
    for idx in range(3):
        gen_path = f'../src/baselines/generative_models/ChemicalVAE/experiments/samples_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{idx}.csv')
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_eval_results.csv', index=False)

def evaluate_grammar_vae():

    baseline_name = 'grammar_vae'
    baseline_path = pathlib.Path(f'../experiments/baselines_evaluation/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../data/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    np.random.seed(42)
    data_size = len(smis)
    split = int(0.9 * data_size)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    train_indices = indices[:split]
    smis = np.array(smis)[train_indices].tolist()

    eval_results = []
    for idx in range(3):
        gen_path = f'../src/baselines/generative_models/GrammarVAE/experiments/samples_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{idx}.csv')
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_eval_results.csv', index=False)

def evaluate_graph_MCTS():
    evaluate(baseline_name='graph_mcts', gen_path_name='GraphMCTS')

def evaluate_group_vae():
    evaluate(baseline_name='group_vae', gen_path_name='group_selfies_vae')

def evaluate_selfies_vae():
    evaluate(baseline_name='selfies_vae', gen_path_name='selfies-vae')

def evaluate_hg2g():
    evaluate(baseline_name='hg2g', gen_path_name='hG2G')

def evaluate_jtvae():
    evaluate(baseline_name='jtvae', gen_path_name='JTVAE')

def evaluate_aae(baseline_name='aae', gen_path_name='moses'):
    baseline_path = pathlib.Path(f'../experiments/baselines_evaluation/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../datasets/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    eval_results = []
    for idx in range(3):
        gen_path = f'../src/baselines/generative_models/{gen_path_name}/experiments/{baseline_name}/samples_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{idx}.csv')
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_eval_results.csv', index=False)

def evaluate_char_rnn(baseline_name='char_rnn', gen_path_name='moses'):
    baseline_path = pathlib.Path(f'../experiments/baselines_evaluation/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../datasets/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    eval_results = []
    for idx in range(3):
        gen_path = f'../src/baselines/generative_models/{gen_path_name}/experiments/{baseline_name}/samples_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{idx}.csv')
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_eval_results.csv', index=False)

def evaluate_organ(baseline_name='organ', gen_path_name='moses'):
    baseline_path = pathlib.Path(f'../experiments/baselines_evaluation/{baseline_name}')
    baseline_path.mkdir(parents=True, exist_ok=True)

    path = '../datasets/fused_units_generated.csv'
    smis = pd.read_csv(path)['smiles'].tolist()

    split_num = round(len(smis) * 0.9)
    smis = smis[:split_num]

    eval_results = []
    for idx in range(3):
        gen_path = f'../src/baselines/generative_models/{gen_path_name}/experiments/{baseline_name}/samples_{idx}.csv'
        gen_smis = pd.read_csv(gen_path)['smiles'].tolist()

        res = get_all_metrics(gen=gen_smis, train_set=smis,
                              save_path=baseline_path,
                              save_name=f'{baseline_name}_{idx}.csv')
        eval_results.append(res['Value'])

    eval_results = np.array(eval_results)
    eval_results = eval_results.T
    eval_results_std = eval_results.std(axis=-1)
    eval_results_mean = eval_results.mean(axis=-1)

    df = pd.DataFrame(eval_results_mean, columns=['Value'])
    df['std'] = eval_results_std
    df.insert(0, 'Metric', res['Metric'])
    df.to_csv(baseline_path / f'{baseline_name}_eval_results.csv', index=False)


def evaluate_gaudi():
    evaluate(baseline_name='gaudi', gen_path_name='GaUDI')

if __name__ == '__main__':

    evaluate_chemical_vae()
    evaluate_grammar_vae()
    evaluate_graph_MCTS()
    evaluate_group_vae()
    evaluate_selfies_vae()
    evaluate_hg2g()
    evaluate_jtvae()
    evaluate_aae()
    evaluate_char_rnn()
    evaluate_organ()
    evaluate_gaudi()
