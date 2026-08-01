import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import optim
from torch.autograd import Variable

from data.utils import smiles_to_one_hot
import grammar.mol_grammar as G
from Network import GrammarVariationalAutoEncoder, VAELoss
from grammar_models import ZincGrammarModel


class ModelManager():
    def __init__(self, model, device, train_step_init=0, lr=1e-3):
        self.train_step = train_step_init
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = VAELoss()

    def train(self, loader):
        # built-in method for the nn.module, sets a training flag.
        self.model.train()
        _losses = []
        for batch_idx, data in enumerate(loader):
            # have to cast data to FloatTensor. DoubleTensor errors with Conv1D
            data = Variable(data)
            data = data.to(self.device)

            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_fn(data, mu, log_var, recon_batch)
            loss_value = loss.data.detach().cpu().numpy()
            _losses.append(loss_value)
            loss.backward()
            self.optimizer.step()
            self.train_step += 1

            batch_size = len(data)

            if batch_idx == 0:
                print(f'batch size: {batch_size}')
            if batch_idx % 50 == 0:
                print(f'training loss: {loss_value}')

        return _losses

    def test(self, loader):

        self.model.eval()
        test_loss = 0
        for batch_idx, data in enumerate(loader):
            with torch.no_grad():
                data = Variable(data)
                data = data.to(self.device)

                recon_batch, mu, log_var = self.model(data)
                loss = self.loss_fn(data, mu, log_var, recon_batch)
                test_loss += (loss * len(data))

        print(f'testset length is: {len(loader.dataset)}')

        test_loss /= len(loader.dataset)
        print(f'====> Test set loss: {test_loss}')



def preprocess(train_path: str, save_path: str):

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(train_path, 'r') as f:
        smis = f.read().splitlines()[1:]

    oh_smiles = smiles_to_one_hot(smis)
    print('SMILES', oh_smiles.shape)
    np.savez_compressed(save_path, arr=oh_smiles)


def split_data(data):

    # np.random.seed(42)
    # np.random.shuffle(data)

    data_size = data.shape[0]
    split = int(0.9 * data_size)
    indices = np.arange(data_size)

    train_indices = indices[:split]
    val_indices = indices[split:]
    train_data = data[train_indices]
    val_data = data[val_indices]

    return torch.FloatTensor(train_data), torch.FloatTensor(val_data)


def train(
        data_path: str,
        save_path: str,
        model_name: str='mol_grammar',
        epochs: int=100,
        batch_size: int=64,
        latent_dim: int=56,
        model_save_name: str='model.pt',
):
    # model_name : mol_grammar, mol_grammar_selfie

    rules = G.gram.split('\n')

    data = np.load(data_path)['arr'].astype(np.float32)
    data_size = data.shape[0]
    vocab_size = data.shape[-1]
    train_data, val_data = split_data(data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    losses = []
    vae = GrammarVariationalAutoEncoder(
        model_name=model_name,
        rules=rules,
    )
    vae.to(device)

    model_mgr = ModelManager(vae, device=device, lr=2e-3)
    for epoch in range(1, epochs+1):
        losses += model_mgr.train(train_loader)
        print(f'epoch {epoch}, loss: {losses}')
        model_mgr.test(val_loader)
    torch.save(model_mgr.model.state_dict(), os.path.join(save_path, model_save_name))



def sample(model_path, save_path, batch_size=250,
           model_name: str='mol_grammar', sample_size=100, base_smi: str=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    rules = G.gram.split('\n')

    model = GrammarVariationalAutoEncoder(model_name=model_name, rules=rules)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    generator = ZincGrammarModel(model)

    base_smi = base_smi if base_smi is not None else "C"
    smiles = [base_smi] * sample_size

    samples = []
    z1 = generator.encode(smiles, device)
    for mol,real in zip(generator.decode(z1, device),smiles):
        print(f'decode {mol}    origin {real}')
        if mol:
            samples.append(mol)
        else:
            samples.append('')

    samples = pd.DataFrame(samples, columns=['SMILES'])
    samples.to_csv(os.path.join(save_path, 'samples.csv'), index=False)


def prior_sample(model_path, save_path, batch_size=250, model_name: str='mol_grammar', sample_size=100, save_name: str='samples.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    rules = G.gram.split('\n')

    model = GrammarVariationalAutoEncoder(model_name=model_name, rules=rules)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    generator = ZincGrammarModel(model)

    samples = []
    n_batch = sample_size // batch_size + 1

    for batch_idx in tqdm(range(n_batch)):
        if batch_idx == n_batch - 1 and sample_size % batch_size != 0:
            z1 = torch.randn(sample_size % batch_size, 56, device=device)
        else:
            z1 = torch.randn(batch_size, 56, device=device)
        samps = generator.decode(z1, device)
        samples.extend(samps)

    # samples = pd.DataFrame(samples, columns=['smiles'])
    # samples.to_csv(os.path.join(save_path, save_name), index=False)

    pass

if __name__ == '__main__':

    train_path = '../../../../datasets/fused_units_generated.csv'
    save_path = 'experiments/fused_units.npz'

    preprocess(train_path, save_path)

    for idx in range(0, 3):
        train(data_path=save_path, save_path='experiments', epochs=300, model_save_name=f'model_{idx}.pt')

    model_path = 'experiments/model_2.pt'
    save_path = 'experiments'
    sample(model_path=model_path, save_path=save_path)

    import time
    for _ in range(3):
        start_time = time.time()
        prior_sample(model_path, save_path, sample_size=10000, save_name=f'samples_{_}.csv')
        end_time = time.time()
        t = end_time - start_time
        print('spend time, ', t, 's')