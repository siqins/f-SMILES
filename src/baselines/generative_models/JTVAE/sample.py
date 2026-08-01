import torch
import torch.nn as nn

import math, random, sys
import argparse
from src.utils import load_json_config
from src.jtnn_vae import JTNNVAE
from src.vocab import Vocab, get_vocab
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--config', required=True)

args = parser.parse_args()

config = load_json_config(args.config)
vocab = get_vocab(args.vocab)
vocab = Vocab(vocab)

model = JTNNVAE(vocab, config['hidden_size'], config['latent_size'], config['depthT'], config['depthG'])
model.load_state_dict(torch.load(args.model))
model = model.cuda()
model.eval()

torch.manual_seed(0)

res = []
for i in range(args.nsample):
    smi = model.sample_prior()
    print(i, smi)
    res.append(smi)
with open(args.output, 'w')as f:
    for smi in res:
        f.write(smi + '\n')
