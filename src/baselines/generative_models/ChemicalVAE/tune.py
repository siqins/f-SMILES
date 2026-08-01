import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.featurizer import OneHotFeaturizer
from src.models import MolecularVAE
import warnings
warnings.filterwarnings('ignore')

def preprocess(train_path, save_path):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(train_path, 'r') as f:
        smis = f.read().splitlines()[1:]

    ohf = OneHotFeaturizer(padlength=140)
    oh_smiles = ohf.featurize(smis)
    print('SMILES', oh_smiles.shape)
    np.savez_compressed(save_path, arr=oh_smiles)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(data_path, save_path, batch_size=250, epochs=500, save_model_name='model.pth'):

    data = np.load(data_path, allow_pickle=True)['arr'].astype(np.float32)
    vocab_size = data.shape[-1]
    len_seq = data.shape[-2]

    split_num = round(len(data) * 0.9)
    test_data = data[split_num:]
    data = data[:split_num]
    data = torch.utils.data.TensorDataset(torch.from_numpy(data))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(test_data))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = MolecularVAE(vocab_size, len_seq=len_seq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        model.train()
        train_loss = 0
        for batch in tqdm(loader, file=sys.stdout):
            batch = batch[0].transpose(1, 2).to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch.transpose(1, 2), mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('train', train_loss / len(loader))

        model.eval()
        test_loss = 0
        for batch in tqdm(test_loader):
            batch = batch[0].transpose(1, 2).to(device)
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch.transpose(1, 2), mu, logvar)
            test_loss += loss.item()
        print('test', test_loss / len(test_loader))

    torch.save(model.state_dict(), os.path.join(save_path, save_model_name))


def sample(model_path, save_path, batch_size=250, sample_size=100, base_smi: str=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = MolecularVAE(vocab_size=39, len_seq=140)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if base_smi is None:
        start = 'C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]'
    else:
        start = base_smi

    # start = start.ljust(120)
    oh = OneHotFeaturizer(padlength=140)
    oh_smiles = oh.featurize([start])
    start_vec = torch.from_numpy(oh_smiles.astype(np.float32)).to(device)
    start_vec = start_vec.transpose(1, 2)

    samples = []
    for idx in tqdm(range(sample_size)):
        recon_x = model(start_vec)[0].cpu().detach().numpy()
        y = np.argmax(recon_x, axis=2)
        sam = oh.decode_smiles_from_index(y[0])
        samples.append(sam)

    samples = pd.DataFrame(samples, columns=['smiles'])
    samples.to_csv(save_path, index=False)


def easy_sample(model_path, save_path, batch_size=250, sample_size=100, base_smi: str=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = MolecularVAE(vocab_size=39, len_seq=140)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    oh = OneHotFeaturizer(padlength=140)

    samples = []
    for _ in tqdm(range(sample_size)):
        prior = torch.randn(1, 292, device=device)
        recon_x = model.decode(prior).cpu().detach().numpy()
        y = np.argmax(recon_x, axis=2)
        sam = oh.decode_smiles_from_index(y[0])
        samples.append(sam)

    samples = pd.DataFrame(samples, columns=['smiles'])
    samples.to_csv(save_path, index=False)

if __name__ == '__main__':

    train_path = '../../../../datasets/fused_units_generated.csv'
    save_path = 'experiments/fused_units.npz'

    preprocess(train_path, save_path)

    for idx in range(3):
        train(data_path=save_path, save_path='experiments', epochs=400, save_model_name=f'model_{idx}.pth')

    model_path = 'experiments/model_1.pth'
    save_path = 'experiments/samples_4.csv'
    sample(model_path=model_path, save_path=save_path, base_smi='c')

    import time
    start_time = time.time()
    easy_sample(model_path, save_path, sample_size=10000)
    end_time = time.time()
    t = end_time - start_time
    print('spend time, ', t, 's')