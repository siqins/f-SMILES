import os.path
import math
import sys
import random
import pickle
import json
import time

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
from src.hgraph import MolGraph, common_atom_vocab, PairVocab, HierVAE, DataFolder
from src.preprocess import tensorize

import rdkit
from joblib import Parallel, delayed

def load_json_config(path):
    """Reutrn a json file into dict"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def get_vocab(file):
    """return the cluster vocaburary"""
    with open(file) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    return vocab

def get_vocab_dict(train_path: str, vocab_path: str) -> None:
    '''
    Preprocess the data, and get the vocab which will be saved in the vocab_path.
    :param train_path: the path of raw_data
    :param vocab_path:  the path of saving the vocab
    :return:
    '''

    if not os.path.exists(vocab_path):
        os.makedirs(os.path.dirname(vocab_path))

    with open(train_path, 'r') as f:
        data = f.read().splitlines()

    vocab = set()
    for item in data[1:]:
        smiles = item.split()[0]
        hmol = MolGraph(smiles)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )

    with open(vocab_path, 'w') as file:
        for c in vocab:
            file.write(c[0] + ' ' + c[1] + '\n')


def preprocess(train_path: str, vocab_path, save_dir: str, batch_size: int, num_workers: int) -> None:

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab = get_vocab(vocab_path)
    vocab = PairVocab(vocab, cuda=False)

    pool = Parallel(n_jobs=num_workers)
    random.seed(42)

    with open(train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f][1:]

    # random.shuffle(data)
    idx_traintest = int(len(data) * 0.9)
    data = data[:idx_traintest]

    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    func = delayed(tensorize)
    all_data = pool(
        func(batch, vocab) for batch in batches
    )

    num_splits = len(all_data) // 1000 + 1

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st: st + le]

        with open(os.path.join(save_dir, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

def train(data_folder: str, vocab_path: str, config_path:str, save_dir: str):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on {}.'.format(device.type))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config = load_json_config(config_path)
    config = SimpleNamespace(**config)
    config.atom_vocab = common_atom_vocab

    vocab = get_vocab(vocab_path)
    vocab = PairVocab(vocab)
    config.vocab = vocab
    config.save_dir = save_dir

    model = HierVAE(config)
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    model = model.to(device)

    for param in model.parameters():
        if param.dim() == 1:
            torch.nn.init.constant_(param, 0)
        else:
            torch.nn.init.xavier_normal_(param)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.anneal_rate)

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    meters = np.zeros(6)
    total_step = beta = 0
    for epoch in range(config.epoch):
        dataset = DataFolder(data_folder, config.batch_size)

        for batch in tqdm(dataset):
            total_step += 1
            model.zero_grad()
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, loss.item(), (wacc * 100).cpu().numpy(),
                                        (iacc * 100).cpu().numpy(), (tacc * 100).cpu().numpy(), (sacc * 100).cpu().numpy()])

            if total_step % config.print_iter == 0:
                meters /= config.print_iter
                print(
                    "[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                        total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5],
                        param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0

            if total_step % config.save_iter == 0:
                ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
                torch.save(ckpt, os.path.join(config.save_dir, f"model.ckpt.{total_step}"))

            if total_step % config.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step >= config.warmup and total_step % config.kl_anneal_iter == 0:
                beta = min(config.max_beta, beta + config.step_beta)


def sample(nsample: int, vocab_path: str, model_path: str, config_path: str, output_path: str) -> None:
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on {}.'.format(device.type))

    config = load_json_config(config_path)
    config = SimpleNamespace(**config)
    config.atom_vocab = common_atom_vocab
    config.batch_size = 1

    vocab = get_vocab(vocab_path)
    vocab = PairVocab(vocab)
    config.vocab = vocab

    model = HierVAE(config)
    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)[0])
    model.eval()

    smis = []
    with torch.no_grad():
        for _ in tqdm(range(nsample // config.batch_size)):
            smiles_list = model.sample(config.batch_size, greedy=True)
            for _, smiles in enumerate(smiles_list):
                # print(_, smiles)
                smis.append(smiles)

    df = pd.DataFrame(smis, columns=['smiles'])
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    train_path = '../../../../datasets/fused_units_generated.csv'

    get_vocab_dict(train_path=train_path, vocab_path='experiments/vocab.txt')

    preprocess(train_path=train_path, vocab_path='experiments/vocab.txt',
               save_dir='experiments/processed', batch_size=32, num_workers=4)

    for idx in range(3):
        train(data_folder='experiments/processed', vocab_path='experiments/vocab.txt',
              config_path='src/configs/config.json', save_dir=f'experiments/vae_models_{idx}')

    for _ in range(3):
        start_time = time.time()
        sample(nsample=10000, vocab_path='experiments/vocab.txt',
               model_path='experiments/vae_models/model_1.ckpt',
               config_path='src/configs/config.json', output_path='experiments/samples_1.csv')
        end_time = time.time()
        t = end_time - start_time
        print('spend time, ', t, 's')