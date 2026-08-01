import os
import time
import math
import json
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from train.model import GPT, GPTConfig
from generate.utils import sample as samp
from preprocess import inverse_transform


def load_tokens(filename='selfies_tokens.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse(data_path, model_path, save_path, max_len: int, sample_size=100,
          seq='smiles', sample_save_name='samples.csv'):

    scaffold: bool = False
    lstm: bool = False
    lstm_layers: int = 0
    num_props: int = 0
    n_layer: int = 8
    n_embed: int = 256
    n_head: int = 8
    batch_size: int = 48
    scaffold_max_len = max_len

    alphabet = load_tokens(os.path.join(data_path, f'{seq}_token_alphabet.json'))
    vocab_size = len(alphabet)
    itos = {i:ch for i, ch in enumerate(alphabet)}
    stoi = {ch:i for i, ch in enumerate(alphabet)}

    print(itos)
    print(len(itos))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    mconf = GPTConfig(vocab_size, max_len, num_props=num_props,
                      n_layer=n_layer, n_head=n_head, n_embd=n_embed, scaffold=scaffold,
                      scaffold_maxlen=scaffold_max_len,
                      lstm=lstm, lstm_layers=lstm_layers)
    model = GPT(mconf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print('Model loaded')

    start_toks = pd.read_csv(os.path.join(data_path, f'{seq}_start_token_count.csv'))
    start_toks_p = start_toks['count'].values
    start_toks_p = start_toks_p / sum(start_toks_p)
    start_toks = start_toks['token'].tolist()

    gen_iter = math.ceil(sample_size / batch_size)
    last_batch = sample_size % batch_size
    completions = []
    for i in tqdm(range(gen_iter)):
        n_samples= batch_size if i != gen_iter - 1 else last_batch
        toks = np.random.choice(start_toks, size=n_samples, replace=True, p=start_toks_p)
        x = torch.tensor([stoi[s] for s in toks], dtype=torch.long).reshape(-1, 1)
        x = x.to(device)
        p = None
        sca = None
        y = samp(model, x, max_len, temperature=1, sample=True, top_k=None, prop=p, scaffold=sca)  # 0.7 for guacamol
        for gen_mol in y:
            completion = ''.join([itos[int(i)] for i in gen_mol])
            completion = completion.replace(' ', '')
            completions.append(completion)

    start = time.time()
    molecules = inverse_transform(completions, seq=seq)
    end = time.time()
    print(f'{seq}  inverse_transform: {end-start} seconds')

    results = pd.DataFrame(molecules, columns=['smiles'])
    results['completion'] = completions
    results.to_csv(os.path.join(save_path, sample_save_name), index=False)
    return end-start
