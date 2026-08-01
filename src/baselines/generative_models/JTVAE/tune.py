import os
import sys
import math
import rdkit
import pickle
import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from joblib import Parallel, delayed
from rdkit import Chem
from collections import defaultdict
from src.mol_tree import MolTree, get_vocab
from src.vocab import Vocab
from src.jtnn_vae import JTNNVAE
from src.datautils import MolTreeFolder
from src.utils import load_json_config
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

class Logger:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n_step = 0
        self.metrics = defaultdict(list)

    def log(self, metric: float, metric_name: str) -> None:
        self.metrics[metric_name].append(metric)

    def step(self) -> None:
        self.n_step += 1

    def save(self, path: str) -> None:
        logger = pd.DataFrame(self.metrics)
        logger.insert(0, 'step', list(range(self.n_step)))
        logger.to_csv(path, index=False)


def get_vocab_dict(train_path: str, vocab_path: str) -> None:
    '''
    Preprocess the data, and get the vocab which will be saved in the vocab_path.
    :param train_path: the path of raw_data
    :param vocab_path:  the path of saving the vocab
    :return:
    '''

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    with open(train_path, 'r') as f:
        data = f.read().splitlines()

    cset = set()
    for item in tqdm(data[1:]):
        smiles = item.split()[0]
        mol = MolTree(smiles)
        for c in mol.nodes:
            cset.add(c.smiles)

    with open(vocab_path, 'w') as file:
        for c in cset:
            file.write(c + '\n')


def preprocess(train_path: str, save_dir: str, nsplits: int, num_workers: int) -> None:

    def tensorize(smiles, assm=True):
        """
        transform smiles into tree objects
        """
        try:
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            if assm:
                mol_tree.assemble()
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)

            del mol_tree.mol
            for node in mol_tree.nodes:
                del node.mol
        except Exception as e:
            print(smiles, e)
            return None

        return mol_tree

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # pool = Pool(num_workers)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(train_path, 'r') as f:
        data = f.read().splitlines()[1:]

    idx_traintest = int(len(data) * 0.9)
    data = data[:idx_traintest]

    num_splits = nsplits
    # all_data = pool.map(tensorize, data)

    pool = Parallel(n_jobs=num_workers)
    all_data = pool(
        delayed(tensorize)(smi) for smi in tqdm(data)
    )

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st: st + le]
        if not sub_data:
            break
        with open(os.path.join(save_dir, 'tensors-%d.pkl' % split_id), 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)


def train(data_folder: str, vocab_path: str, config_path: str, save_dir: str, test_data_folder: str = None) -> None:

    num_workers: int = 4
    num_epoch: int = 50
    batch_size: int = 32
    load_epoch: int = 0
    use_gpu: bool = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Running on {}.'.format(device.type))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    config = load_json_config(config_path)

    vocab = get_vocab(vocab_path)
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, config['hidden_size'], config['latent_size'], config['depthT'], config['depthG'])
    model.to(device)

    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.iter-" + str(load_epoch)))

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, config['anneal_rate'])
    # scheduler.step()

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = load_epoch
    beta = config['beta']
    meters = np.zeros(4)

    logger = Logger()
    test_logger = Logger()
    for epoch in range(num_epoch):
        print('Epoch ', epoch)
        loader = MolTreeFolder(data_folder, vocab, batch_size, num_workers=num_workers)
        for batch in tqdm(loader, file=sys.stdout):
            total_step += 1
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'])
                optimizer.step()

                logger.step()
                logger.log(metric=loss.detach().cpu().item(), metric_name='loss')
                logger.log(metric=kl_div, metric_name='kl_div')
                logger.log(metric=wacc, metric_name='wacc')
                logger.log(metric=tacc, metric_name='tacc')
                logger.log(metric=sacc, metric_name='sacc')
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
                torch.save(model.state_dict(), save_dir + "/model.iter-" + str(total_step))

            if total_step % config['anneal_iter'] == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_step % config['kl_anneal_iter'] == 0 and total_step >= config['warmup']:
                beta = min(config['max_beta'], beta + config['step_beta'])

            if total_step % config['test_iter'] == 0 and test_data_folder is not None:
                test_loader = MolTreeFolder(test_data_folder, vocab, batch_size, num_workers=num_workers)
                print('Testing on ', total_step)

                model.eval()
                with torch.no_grad():
                    for batch in tqdm(test_loader, file=sys.stdout):
                        loss, kl_div, wacc, tacc, sacc = model(batch, beta)

                        test_logger.step()
                        test_logger.log(metric=loss.detach().cpu().item(), metric_name='loss')
                        test_logger.log(metric=kl_div, metric_name='kl_div')
                        test_logger.log(metric=wacc, metric_name='wacc')
                        test_logger.log(metric=tacc, metric_name='tacc')
                        test_logger.log(metric=sacc, metric_name='sacc')
                model.train()
                test_logger.save(path=os.path.join(save_dir, 'test_logger.csv'))

        logger.save(path=os.path.join(save_dir, 'logger.csv'))


def sample(nsample: int, vocab_path: str, model_path: str, config_path: str, output_path: str) -> None:
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on {}.'.format(device.type))

    config = load_json_config(config_path)
    vocab = get_vocab(vocab_path)
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, config['hidden_size'], config['latent_size'], config['depthT'], config['depthG'])
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    torch.manual_seed(0)

    res = []
    for i in range(nsample):
        smi = model.sample_prior()
        print(i, smi)
        res.append(smi)

    df = pd.DataFrame(res, columns=['smiles'])
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    train_path = '../../../../datasets/fused_units_generated.csv'

    get_vocab_dict(train_path=train_path, vocab_path='experiments/vocab.txt')

    preprocess(train_path=train_path, save_dir='experiments/processed', nsplits=100, num_workers=4)

    for idx in range(3):
        train(data_folder='experiments/processed', vocab_path='experiments/vocab.txt', config_path='configs/config.json', save_dir=f'experiments/vae_models_{idx}')

    for idx in [1, 2]:
        sample(nsample=10000, vocab_path='experiments/vocab.txt', model_path=f'experiments/vae_models/model.iter-{idx}',
               config_path='configs/config.json', output_path=f'experiments/samples_{idx}.csv')
