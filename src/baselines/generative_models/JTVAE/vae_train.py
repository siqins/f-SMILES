"""JTVAE training process"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle

from src.jtnn_vae import JTNNVAE
from src.vocab import Vocab, get_vocab
from src.datautils import MolTreeFolder
from src.utils import load_json_config
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--save_dir', required=True)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--use_gpu', type=eval, default=True)

args = parser.parse_args()
print(args)

config = load_json_config(args.config)

vocab = get_vocab(args.vocab)
vocab = Vocab(vocab)

model = JTNNVAE(vocab, config['hidden_size'], config['latent_size'], config['depthT'], config['depthG']).cuda()
print(model)

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = lr_scheduler.ExponentialLR(optimizer, config['anneal_rate'])
scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = config['beta']
meters = np.zeros(4)

for epoch in range(args.epoch):
    loader = MolTreeFolder(args.train, vocab, args.batch_size, num_workers=args.num_workers)
    for batch in loader:
        total_step += 1
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
            optimizer.step()
        except Exception as e:
            print(e)
            continue

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

        if total_step % config['print_iter'] == 0:
            meters /= config['print_iter']
            print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
            sys.stdout.flush()
            meters *= 0

        if total_step % config['save_iter'] == 0:
            torch.save(model.state_dict(), config['save_dir'] + "/model.iter-" + str(total_step))

        if total_step % config['anneal_iter'] == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step % config['kl_anneal_iter'] == 0 and total_step >= config['warmup']:
            beta = min(config['max_beta'], beta + config['step_beta'])
