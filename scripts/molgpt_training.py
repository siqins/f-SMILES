import os
import time
import pickle
import pandas as pd
from collections import Counter

def count_start_token():

    seqs = ['smiles', 'smiles-pair-encoding', 'clear-smiles', 'deep-smiles', 'selfies', 'group-selfies', 'tsmiles', 'fsmiles-pair-encoding', 'fsmiles']
    raw_save_path = r'../experiments/molgpt_exps_distribution_learning'

    for seq in seqs:
        save_path = os.path.join(raw_save_path, seq)

        train_data_path = os.path.join(save_path, 'train_toks.pkl')
        with open(train_data_path, 'rb') as f:
            train_toks = pickle.load(f)

        start_toks =[toks[0] for toks in train_toks]

        count_lens_df = Counter(start_toks)
        count_lens_df = dict(count_lens_df).items()
        count_lens_df = pd.DataFrame(count_lens_df, columns=['token', 'count'])
        count_lens_df.sort_values('token', ascending=False, inplace=True)
        count_lens_df.to_csv(f'{save_path}/{seq}_start_token_count.csv', index=False)


def deal_with_inconsistent_data(data_path, save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_data_path = os.path.join(data_path, 'train_toks.pkl')
    valid_data_path = os.path.join(data_path, 'valid_toks.pkl')
    with open(train_data_path, 'rb') as f:
        train_toks = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_toks = pickle.load(f)

    toks = train_toks + valid_toks
    raw_data_path = r'../datasets/fused_units_generated.csv'
    raw_data = pd.read_csv(raw_data_path)
    raw_data['toks'] = toks

    calc_data_path = r'../datasets/fused_units_generated_calculated.csv'
    calc_data = pd.read_csv(calc_data_path)

    merge_data = pd.merge(raw_data, calc_data, on='smiles', how='inner')

    n_smis = len(merge_data)
    n_split = int(0.9 * n_smis)
    toks = merge_data['toks'].tolist()
    print('len(toks)', len(toks))
    train_toks = toks[:n_split]
    valid_toks = toks[n_split:]

    save_path = os.path.join(save_path, 'toks_after_calc')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/train_toks.pkl', 'wb') as f:
        pickle.dump(train_toks, f)
    with open(f'{save_path}/valid_toks.pkl', 'wb') as f:
        pickle.dump(valid_toks, f)

def preprocess():

    from src.baselines.generative_models.MolGPT.preprocess import parse

    data_path = r'../datasets/fused_units_generated.csv'
    save_path = r'../experiments/molgpt_exps_distribution_learning'
    sequence_save_path = r'../experiments/sequence_prediction_exps'

    seqs = ['smiles', 'smiles-pair-encoding', 'clear-smiles', 'deep-smiles', 'selfies', 'group-selfies', 'tsmiles', 'fsmiles-pair-encoding', 'fsmiles']
    for seq in seqs:
        parse(data_path=data_path, save_path=os.path.join(save_path, seq), seq=seq, sequence_save_path=os.path.join(sequence_save_path, seq))
        deal_with_inconsistent_data(data_path=os.path.join(save_path, seq), save_path=os.path.join(sequence_save_path, seq))

    count_start_token()

def train_distribution_learning(epochs=10):
    from src.baselines.generative_models.MolGPT.train_molgpt_distribution_learning import parse

    data_path = r'../experiments/molgpt_exps_distribution_learning'
    save_path = r'../experiments/molgpt_exps_distribution_learning'

    max_len_dict = {
        'smiles' : 124,
        'smiles-pair-encoding': 51,
        'clear-smiles': 161,
        'deep-smiles': 143,
        'selfies': 126,
        'group-selfies': 107,
        'tsmiles': 42,
        'fsmiles-pair-encoding': 15,
        'fsmiles': 30
    }

    seqs = ['smiles', 'smiles-pair-encoding', 'clear-smiles', 'deep-smiles', 'selfies', 'group-selfies', 'tsmiles', 'fsmiles-pair-encoding', 'fsmiles']
    for seq in seqs:
        for idx in [0, 1, 2]:
            start_time = time.time()
            parse(data_path=os.path.join(data_path, seq),
                  save_path=os.path.join(save_path, seq),
                  max_len=max_len_dict[seq],
                  seq=seq,
                  model_save_name=f'model_{epochs}_{idx}.pt',
                  epochs=epochs)
            end_time = time.time()
            print(seq, ' Total time:', end_time - start_time)


def train_goal_oriented_generation(epochs=10):
    from src.baselines.generative_models.MolGPT.train_molgpt_goal_oriented_generation import parse

    data_path = r'../experiments/sequence_prediction_exps'
    save_path = r'../experiments/molgpt_exps_goal_oriented_generation'

    max_len_dict = {
        'smiles' : 124,
        'smiles-pair-encoding': 51,
        'clear-smiles': 161,
        'deep-smiles': 143,
        'selfies': 126,
        'group-selfies': 107,
        'tsmiles': 42,
        'fsmiles-pair-encoding': 15,
        'fsmiles': 30
    }

    props = ['lumo', 'gap', 'energy', 'ionization_potential aip', 'electron_affinity aea']
    props_mapper = {
        'lumo': 'LUMO',
        'gap': 'HLG',
        'energy': 'Erel',
        'ionization_potential aip': 'IP',
        'electron_affinity aea': 'EA',
    }

    seqs = ['clear-smiles', 'smiles', 'smiles-pair-encoding', 'deep-smiles', 'selfies', 'group-selfies', 'tsmiles', 'fsmiles-pair-encoding', 'fsmiles']
    for pred_name in props:
        for seq in seqs:
            for idx in [0, 1, 2]:
                start_time = time.time()
                parse(data_path=os.path.join(data_path, seq),
                      save_path=os.path.join(save_path, seq, props_mapper[pred_name]),
                      max_len=max_len_dict[seq],
                      seq=seq,
                      model_save_name=f'model_{epochs}_{idx}.pt',
                      epochs=epochs,
                      pred_name=pred_name)
                end_time = time.time()
                print(seq, ' Total time:', end_time - start_time)

def sample_distribution_learning(epochs=10):
    from src.baselines.generative_models.MolGPT.sample_distribution_learning import parse

    data_path = r'../experiments/molgpt_exps_distribution_learning'
    save_path = r'../experiments/molgpt_exps_distribution_learning'

    max_len_dict = {
        'smiles' : 124,
        'smiles-pair-encoding': 51,
        'clear-smiles': 161,
        'deep-smiles': 143,
        'selfies': 126,
        'group-selfies': 107,
        'tsmiles': 42,
        'fsmiles-pair-encoding': 15,
        'fsmiles': 30
    }

    names = []
    time1, time2 = [], []
    for seq in ['smiles' ]:
        for idx in [0, 1, 2]:
            start_time = time.time()
            time_1 = parse(data_path=os.path.join(data_path, seq),
                  model_path=os.path.join(save_path, seq, f'model_{epochs}_{idx}.pt'),
                  save_path=os.path.join(save_path, seq),
                  max_len=max_len_dict[seq],
                  sample_size=10000,
                  seq=seq,
                  sample_save_name=f'samples_{epochs}_{idx}.csv')
            end_time = time.time()
            time1.append(time_1)
            time2.append(end_time - start_time)
            names.append(seq)
            print(seq, ' Total time:', end_time - start_time)

    import pandas as pd
    df = pd.DataFrame()
    df['names'] = names
    df['time_inference_and_decoding'] = time2
    df['time_decode'] = time1
    df.to_csv(os.path.join(save_path, 'times.csv'), index=False)

def sample_goal_oriented_generation(epochs=10):
    from src.baselines.generative_models.MolGPT.sample_goal_oriented_generation import parse

    data_path = r'../experiments/molgpt_exps_distribution_learning'
    save_path = r'../experiments/molgpt_exps_goal_oriented_generation'

    max_len_dict = {
        'smiles' : 124,
        'smiles-pair-encoding': 51,
        'clear-smiles': 161,
        'deep-smiles': 143,
        'selfies': 126,
        'group-selfies': 107,
        'tsmiles': 42,
        'fsmiles-pair-encoding': 15,
        'fsmiles': 30
    }

    prop_names = ['lumo', 'gap', 'energy', 'ionization_potential aip', 'electron_affinity aea']
    props_mapper = {
        'lumo': 'LUMO',
        'gap': 'HLG',
        'energy': 'Erel',
        'ionization_potential aip': 'IP',
        'electron_affinity aea': 'EA',
    }

    for prop_name in prop_names:
        for seq in ['smiles', 'smiles-pair-encoding', 'clear-smiles', 'deep-smiles', 'selfies', 'group-selfies', 'tsmiles', 'fsmiles-pair-encoding', 'fsmiles']:
            for idx in [0, 1, 2]:
                start_time = time.time()
                parse(data_path=os.path.join(data_path, seq),
                       model_path=os.path.join(save_path, seq, props_mapper[prop_name], f'model_{epochs}_{idx}.pt'),
                       save_path=os.path.join(save_path, seq, props_mapper[prop_name]),
                       max_len=max_len_dict[seq],
                       sample_size=10000,
                       seq=seq,
                       sample_save_name=f'samples_{epochs}_{idx}.csv',
                       prop_name=prop_name)
                end_time = time.time()
                print(seq, ' Total time:', end_time - start_time)

if __name__ == '__main__':

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    preprocess()

    epochs = 10
    train_distribution_learning(epochs=epochs)
    sample_distribution_learning(epochs=epochs)

    train_goal_oriented_generation(epochs=epochs)
    sample_goal_oriented_generation(epochs=epochs)
