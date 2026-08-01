import os, sys
import shutil
import random
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.autograd import Variable
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from MPNN_codes.utils import CustomDataset, get_graph_from_smile
from MPNN_codes.layers.mpnn_layer import MPNN
import warnings
warnings.filterwarnings('ignore')


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    return device

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def set_loss_fn(loss_select):
    if loss_select == 'l1':
        # loss_fn = nn.L1Loss()
        loss_fn = nn.L1Loss(reduction='sum')  # sum，mean,none
    elif loss_select == 'l2':
        loss_fn = nn.MSELoss(reduction='mean')
    elif loss_select == 'sml1':
        loss_fn = nn.SmoothL1Loss(reduction='sum')  # mean,none,sum
    elif loss_select == 'bce':
        loss_fn = nn.BCELoss(reduction='mean')
    else:
        print('No Found the Loss function!')
    return loss_fn


def parse_data(smiles: list, values: list, value_name: str='labels') -> pd.DataFrame:
    df = pd.DataFrame()
    df['smiles'] = smiles
    df[value_name] = values
    return df

def create_data(smiles: list):
    graphs = []
    for smi in smiles:
        graph = get_graph_from_smile(smi)
        graphs.append(graph)
    return graphs


def train_one_epoch(model, loss_fn, train_loader, valid_loader, optimizer, epoch, verbose=True):


    total_train_loss = 0.0
    total_valid_loss = 0.0

    # switch to train mode
    model.train()

    loops = enumerate(tqdm(train_loader, file=sys.stdout, ncols=100)) if verbose else enumerate(train_loader)
    for idx, batch in loops:
        graphs, labels = batch
        outputs = model(graphs)

        optimizer.zero_grad()
        # Compute output
        labels = labels.squeeze()
        outputs = outputs.squeeze()
        if len(labels.shape) == 0:
            labels = labels.unsqueeze(0)
        if len(outputs.shape) == 0:
            outputs = outputs.unsqueeze(0)

        train_loss = loss_fn(outputs, labels)
        total_train_loss += train_loss.item()

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        loops = enumerate(tqdm(valid_loader, file=sys.stdout, ncols=100)) if verbose else enumerate(valid_loader)
        for idx, batch in loops:
            graphs, labels = batch
            outputs = model(graphs)

            labels = labels.squeeze()
            outputs = outputs.squeeze()
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)
            if len(outputs.shape) == 0:
                outputs = outputs.unsqueeze(0)

            loss = loss_fn(outputs, labels)
            total_valid_loss = total_valid_loss + loss.item()

    total_train_loss = total_train_loss / len(train_loader)
    total_valid_loss = total_valid_loss / len(valid_loader)
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_valid_loss:.4f}")
    return total_train_loss, total_valid_loss


def predict_one_epoch(model, loss_fn, test_loader, epoch, verbose=True):
    total_preds = []
    total_labels = []
    model.eval()
    with torch.no_grad():
        loops = enumerate(tqdm(test_loader, file=sys.stdout, ncols=100)) if verbose else enumerate(test_loader)
        for idx, batch in loops:
            graphs, labels = batch
            outputs = model(graphs)

            labels = labels.squeeze()
            outputs = outputs.squeeze()
            if len(labels.shape) == 0:
                labels = labels.unsqueeze(0)
            if len(outputs.shape) == 0:
                outputs = outputs.unsqueeze(0)

            total_preds.append(outputs.cpu().detach())
            total_labels.append(labels.cpu().detach())

    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    loss = loss_fn(total_labels, total_preds)
    print(f"Epoch {epoch}|Test Loss: {loss:.4f}")
    return loss

def train(
        trainset,
        validset,
        testset = None,
        save_path='experiments/processed_data',
        model_save_path: str = 'experiments/MPNN',

        model_select: str = 'MPNN',
        encoder_name: str='MPNN',  # MPNN
        loss_select: str = 'l2',
        task: str = 'regression',

        epochs: int = 501,
        batch_size: int = 128,
        num_workers: int = 0,

        verbose: bool = True,
        patience: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = set_device()
    set_seed(seed=42)

    train_graphs = create_data(trainset.smiles)
    valid_graphs = create_data(validset.smiles)

    train_dateset = CustomDataset(train_graphs, trainset['labels'], device, task)
    valid_dateset = CustomDataset(valid_graphs, validset['labels'], device, task)
    # Data Loader
    train_loader = DataLoader(train_dateset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=False)
    valid_loader = DataLoader(valid_dateset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    if testset is not None:
        test_graphs = create_data(testset.smiles)
        test_dateset = CustomDataset(test_graphs, testset['labels'], device, task)
        test_loader = DataLoader(test_dateset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    # Define model and optimizer
    print('Create model')
    model = MPNN(node_input_dim=51, edge_input_dim=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = model.to(device)

    print('Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = set_loss_fn(loss_select)

    best_epoch, best_loss = 0, 1e9
    train_loss_list, valid_loss_list, test_loss_list = [], [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, valid_loss = train_one_epoch(model, loss_fn, train_loader, valid_loader, optimizer, epoch, verbose)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if testset is not None:
            test_loss = predict_one_epoch(model, loss_fn, test_loader, epoch, verbose)
            test_loss_list.append(test_loss)

        print(f'Epoch {epoch} / {epochs} : Learning rate :', lr)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            print(f'<<<<<< reach best valid_loss : {best_loss} >>>>>>')
            torch.save(model.state_dict(), os.path.join(model_save_path, 'model.pth'))

        if epoch - best_epoch > patience:
            print(f"<<<<<< valid_loss without improvement in {patience} epoch, early stopping >>>>>>")
            torch.save(model.state_dict(), os.path.join(model_save_path, 'model_f.pth'))
            break

    results = pd.DataFrame()
    results['train_loss'] = train_loss_list
    results['valid_loss'] = valid_loss_list
    if testset is not None:
        results['test_loss'] = test_loss_list
    results.to_csv(os.path.join(model_save_path, 'results.csv'))
    return results

def predict(
        smis,
        model_path: str = 'experiments/mpnn',

        model_select: str = 'mpnn',
        encoder_name: str='MPNN',  # CMPNN MPNN

        task: str = 'regression',


        output_size: int=1,

        batch_size: int = 128,
        num_workers: int = 0,

        verbose: bool = True,
        patience: int = 20,
):
    device = set_device()
    set_seed(seed=42)

    graphs = create_data(smis)
    smis = parse_data(smis, values=list(range(len(smis))))

    pred_dataset = CustomDataset(graphs, smis['labels'], device, task)
    loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_workers)

    # Define model and optimizer
    print('Create model')
    model = MPNN(node_input_dim=51, edge_input_dim=10)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        loops = tqdm(loader) if verbose else loader
        for graphs, _ in loops:
            graphs = graphs.to(device)
            preds = model(graphs).squeeze()
            predictions.append(preds.detach().cpu())

    predictions = [tensor if tensor.dim() > 0 else tensor.unsqueeze(0) for tensor in predictions]
    predictions = torch.cat(predictions).cpu().numpy()
    return predictions

if __name__ == "__main__":
    smis = ['*CC*', 'CCOC', 'CCCCCOC', 'CCSOCC']

    data = parse_data(smis, values=list(range(len(smis))))

    infos = train(
        trainset=data,
        validset=data,

        model_select='mpnn',
        epochs=5,
    )
    infos.to_csv('mpnn_results.csv', index=False)

    print('Prediction.')
    model_path = 'experiments/mpnn/model.pth'
    prediction = predict(
        smis=smis,
        model_path=model_path,
        model_select='mpnn',
    )
    print(prediction)


