import torch
import math
from torch.utils.data import Dataset
from train.utils import SmilesEnumerator
import numpy as np
import re

class SmileDataset(Dataset):

    def __init__(self, data, content, block_size, aug_prob = 0.5, prop = None, scaffold = None, scaffold_maxlen = None, debug=False):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d smiles, %d unique characters.' % (data_size, vocab_size))
    
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        self.sca = scaffold
        self.scaf_max_len = scaffold_maxlen
        self.debug = debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob
    
    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        toks, prop, scaffold = self.data[idx], self.prop[idx], self.sca[idx]    # self.prop.iloc[idx, :].values  --> if multiple properties
        if len(toks) > self.max_len:
            toks = toks[:self.max_len]

        if len(scaffold) > self.scaf_max_len:
            scaffold = scaffold[:self.scaf_max_len]

        dix =  [self.stoi[s] for s in toks]
        sca_dix = [self.stoi[s] for s in scaffold]

        sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)
        prop = torch.tensor([prop], dtype = torch.float)
        return x, y, prop, sca_tensor
