import os
import time
import warnings
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger

from moses.script_utils import add_train_args, read_smiles_csv, set_seed, add_sample_args
from moses.models_storage import ModelsStorage
from moses.dataset import get_dataset
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

MODELS = ModelsStorage()
warnings.filterwarnings('ignore')

def get_parser(model_name):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    assert model_name in MODELS.get_model_names()

    parser = add_train_args(
        MODELS.get_model_train_parser(model_name)(
            subparsers.add_parser(model_name)
        )
    )
    return parser


def train(
        model_name: str,
        data_path,
        save_path: str='experiments',

        seed: int=42,
        batch_size: int=64,
        vocab_save: str=None,
):

    parser = get_parser(model_name)
    config = parser.parse_args()
    config.model_save = os.path.join(save_path, 'model.pt')
    config.config_save = save_path
    config.n_batch = batch_size

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(data_path, 'r') as f:
        smis = f.read().splitlines()[1:]
    n_split = int(len(smis) * 0.9)
    train_smis = smis[:n_split]
    test_smis = smis[n_split:]

    set_seed(seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    trainer = MODELS.get_model_trainer(model_name)(config)

    vocab = trainer.get_vocabulary(smis)
    torch.save(vocab, os.path.join(save_path, 'vocab.pt'))
    torch.save(config, os.path.join(save_path, 'config.pt'))

    model = MODELS.get_model_class(model_name)(vocab, config).to(device)
    trainer.fit(model, train_smis, test_smis)

    model = model.to('cpu')
    torch.save(model.state_dict(), config.model_save)


def sample(
        model_name: str,
        model_path: str,
        save_path: str='experiments',
        n_samples: int=1000,

        n_batch: int=32,

        seed: int=42,
        max_len: int=150,
):

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Models sampler script', description='available models')
    parser = add_sample_args(subparsers.add_parser(model_name))
    config = parser.parse_args()
    config.max_len = max_len

    set_seed(seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model_config = torch.load(os.path.join(model_path, 'config.pt'), weights_only=False)
    model_vocab = torch.load(os.path.join(model_path, 'vocab.pt'), weights_only=False)
    model_state = torch.load(os.path.join(model_path, 'model.pt'))

    model = MODELS.get_model_class(model_name)(model_vocab, model_config)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    samples = []
    n = n_samples
    with tqdm(total=n_samples, desc='Generating samples') as T:
        while n > 0:
            current_samples = model.sample(
                min(n, n_batch), config.max_len
            )
            samples.extend(current_samples)

            n -= len(current_samples)
            T.update(len(current_samples))

    samples = pd.DataFrame(samples, columns=['smiles'])
    samples.to_csv(os.path.join(save_path, 'samples_0.csv'), index=False)


if __name__ == '__main__':

    data_path = '../../../../datasets/fused_units_generated.csv'
    save_path = 'experiments/fused_units.npz'

    # model_name : aae char_rnn vae organ latentgan
    model_name = 'char_rnn'
    for model_name in ['aae', 'char_rnn', 'vae', 'organ']:
        for idx in range(3):
            train(model_name=model_name, data_path=data_path, save_path=f'experiments/{str(idx)}/{model_name}')

    model_path = f'experiments/1/{model_name}'
    save_path = f'experiments/{model_name}'
    start_time = time.time()
    sample(model_name=model_name, model_path=model_path, save_path=save_path, n_samples=10000)
    end_time = time.time()
    print('Time elapsed:', end_time - start_time)
