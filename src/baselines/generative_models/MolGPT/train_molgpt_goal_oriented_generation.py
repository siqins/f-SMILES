import os
import json
import time
import pickle
import random
import torch
import numpy as np
import pandas as pd
from train.model import GPT, GPTConfig
from train.trainer import Trainer, TrainerConfig
from train.dataset import SmileDataset
from train.utils import get_mol

def load_tokens(filename='selfies_tokens.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse(data_path, save_path, max_len: int, seq='smiles', model_save_name='model.pt', epochs: int=10, pred_name='lumo'):

    scaffold: bool = False
    lstm: bool = False
    lstm_layers: int = 0
    num_props: int = 1
    n_layer: int = 8
    n_embed: int = 256
    n_head: int = 8
    max_epochs: int = epochs
    batch_size: int = 384
    lr: float = 6e-4
    scaffold_max_len = max_len

    if os.path.join(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_data_path = os.path.join(data_path, 'toks_after_calc', 'train_toks.pkl')
    valid_data_path = os.path.join(data_path, 'toks_after_calc', 'valid_toks.pkl')
    with open(train_data_path, 'rb') as f:
        train_toks = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_toks = pickle.load(f)

    data = pd.read_csv(r'../datasets/fused_units_generated_calculated.csv')
    n_smis = len(data)
    n_split = int(0.9 * n_smis)
    props = data[pred_name].values
    train_props = props[:n_split]
    valid_props = props[n_split:]
    mean_, std_ = train_props.mean(), train_props.std()
    train_props = (train_props - mean_) / std_
    valid_props = (valid_props - mean_) / std_


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
                          num_workers=0, ckpt_path=os.path.join(save_path, model_save_name),
                          block_size=train_dataset.max_len, generate=False)
    trainer = Trainer(model, train_dataset, valid_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos)

    df = trainer.train()
    df.to_csv(os.path.join(save_path, f'train_losses_{epochs}.csv'), index=False)
