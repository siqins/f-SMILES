import os
import json
import pickle
import random
import torch
import numpy as np
from train.model import GPT, GPTConfig
from train.trainer import Trainer, TrainerConfig
from train.dataset import SmileDataset

def load_tokens(filename='selfies_tokens.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse(data_path, save_path, max_len: int, seq='smiles', model_save_name='model.pt', epochs: int=10):

    scaffold: bool = False
    lstm: bool = False
    lstm_layers: int = 0
    num_props: int = 0
    n_layer: int = 8
    n_embed: int = 256
    n_head: int = 8
    max_epochs: int = epochs
    batch_size: int = 384
    lr: float = 6e-4
    scaffold_max_len = max_len

    train_data_path = os.path.join(data_path, 'train_toks.pkl')
    valid_data_path = os.path.join(data_path, 'valid_toks.pkl')
    with open(train_data_path, 'rb') as f:
        train_toks = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_toks = pickle.load(f)

    train_props = [0] * len(train_toks)
    valid_props = [0] * len(valid_toks)

    alphabet = load_tokens(os.path.join(data_path, f'{seq}_token_alphabet.json'))
    train_dataset = SmileDataset(train_toks, alphabet, max_len, prop=train_props, aug_prob=0,
                                 scaffold=train_toks, scaffold_maxlen= max_len)
    valid_dataset = SmileDataset(valid_toks, alphabet, max_len, prop=valid_props, aug_prob=0,
                                 scaffold=valid_toks, scaffold_maxlen= max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, num_props=num_props,  # args.num_props,
                        n_layer=n_layer, n_head=n_head, n_embd=n_embed, scaffold=scaffold, scaffold_maxlen=scaffold_max_len,
                        lstm=lstm, lstm_layers=lstm_layers)
    model = GPT(mconf)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    set_seed(42)

    tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=lr,
                          lr_decay=True, warmup_tokens=0.1*len(train_toks)*max_len,
                          final_tokens=max_epochs*len(train_toks)*max_len,
                          num_workers=0,
                          ckpt_path=os.path.join(save_path, model_save_name),
                          block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos)

    df = trainer.train()
    df.to_csv(os.path.join(save_path, f'train_losses_{epochs}.csv'), index=False)


def count_start_token():

    import pandas as pd
    from collections import Counter

    seqs = ['clear-smiles', 'smiles', 'smiles-pair-encoding', 'deep-smiles', 'selfies', 'group-selfies',
            'tsmiles', 'fsmiles', 'fsmiles-branched']

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
