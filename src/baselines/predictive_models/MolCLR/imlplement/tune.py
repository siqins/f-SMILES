import os, sys
import shutil
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from MolCLR_codes.models.ginet_finetune import GINet
from MolCLR_codes.models.gcn_finetune import GCN

ATOM_LIST = list(range(0,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

class CustomDataset(Dataset):
    def __init__(self, graph_list, label_list=None, device=None, task='regression'):
        self.labels = label_list
        self.graphs = graph_list
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.task = task

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        label = self.labels[index] if self.labels is not None else 0
        if self.task == 'classification':
            label = torch.tensor(label, dtype=torch.long).to(self.device)
        elif self.task == 'regression':
            label = torch.tensor(label, dtype=torch.float).to(self.device)
        graph = self.graphs[index].to(self.device)
        return graph, label

    @staticmethod
    def parse_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

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

def set_model(model_select, task, n_layers, emb_dim, feat_dim, dropout, pool):

    def _load_pre_trained_weights(model, model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    path = os.path.dirname(os.path.dirname(__file__))
    if model_select == 'gin':
        model = GINet(task, n_layers, emb_dim, feat_dim, dropout, pool)
        model_path = os.path.join(path, 'MolCLR_codes', 'ckpt', 'pretrained_gin', 'checkpoints', 'model.pth')
        model = _load_pre_trained_weights(model, model_path)
    elif model_select == 'gcn':
        model = GCN(task, n_layers, emb_dim, feat_dim, dropout, pool)
        model_path = os.path.join(path, 'MolCLR_codes', 'ckpt', 'pretrained_gcn', 'checkpoints', 'model.pth')
        model = _load_pre_trained_weights(model, model_path)
    else:
        print('No Found the model!')
    return model

def parse_data(smiles: list, values: list, value_name: str='labels') -> pd.DataFrame:
    df = pd.DataFrame()
    df['smiles'] = smiles
    df[value_name] = values
    return df

def create_data(df_dataset, save=False, save_path='processed_data.pth'):

    if os.path.exists(save_path):
        graph_list = torch.load(save_path)
    else:
        smiles_list = df_dataset['smiles']
        graph_list = []
        for idx, smiles in enumerate(tqdm(smiles_list, file=sys.stdout, ncols=100)):
            graph = CustomDataset.parse_mol(smiles)
            if graph == False:
                continue
            else:
                graph_list.append(graph)
        if save:
            torch.save(graph_list, save_path)

    return graph_list

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


def train_one_epoch(model, loss_fn, train_loader, valid_loader, optimizer, epoch, verbose=True):

    total_train_loss = 0.0
    total_valid_loss = 0.0

    model.train()
    loops = enumerate(tqdm(train_loader, file=sys.stdout, ncols=100)) if verbose else enumerate(train_loader)
    for idx, batch in loops:
        graphs, labels = batch
        outputs = model(graphs)

        labels = labels.squeeze()
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        total_train_loss = total_train_loss + loss.item()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    loops = enumerate(tqdm(valid_loader, file=sys.stdout, ncols=100)) if verbose else enumerate(valid_loader)
    for idx, batch in loops:
        graphs, labels = batch
        outputs = model(graphs)

        labels = labels.squeeze()
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, labels)
        total_valid_loss = total_valid_loss + loss.item()

    total_train_loss = total_train_loss / len(train_loader)
    total_valid_loss = total_valid_loss / len(valid_loader)
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_valid_loss:.4f}")
    return total_train_loss, total_valid_loss

def predict_one_epoch(model, loss_fn, test_loader, epoch, verbose=True):
    model.eval()

    total_preds = []
    total_labels = []
    loops = enumerate(tqdm(test_loader, file=sys.stdout, ncols=100)) if verbose else enumerate(test_loader)
    for idx, batch in loops:
        graphs, labels = batch
        outputs = model(graphs)

        labels = labels.squeeze()
        outputs = outputs.squeeze()
        total_preds.append(outputs)
        total_labels.append(labels)

    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    loss = loss_fn(total_labels, total_preds)
    print(f"Epoch {epoch}|Test Loss: {loss:.4f}")
    return loss


def train(
        trainset,
        validset,
        testset = None,
        save=False,
        save_path='experiments/processed_data',
        model_save_path: str = 'experiments/gcn',

        model_select: str = 'gin',
        n_layers: int = 5,
        emb_dim: int = 300,
        feat_dim: int = 512,
        dropout: float = 0.1,
        pool: str = 'mean',
        loss_select: str = 'l2',
        task: str = 'regression',

        init_base_lr: float = 0.0001,
        init_lr: float = 0.0005,
        weight_decay: float=1e-6,
        gamma: float = 0.98,
        epochs: int = 501,
        batch_size: int = 128,
        num_workers: int = 0,

        verbose: bool = True,
        patience: int = 20,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = set_device()
    set_seed(seed=42)

    train_graphs = create_data(trainset, save=save, save_path=save_path+'/train.pth' if isinstance(save_path, str) else save_path / 'train.pth')
    valid_graphs = create_data(validset, save=save, save_path=save_path+'/valid.pth' if isinstance(save_path, str) else save_path / 'valid.pth')
    train_dataset = CustomDataset(train_graphs, trainset['labels'], device)
    valid_dataset = CustomDataset(valid_graphs, validset['labels'], device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if testset is not None:
        test_graphs = create_data(testset, save=save, save_path=save_path+'/test.pth' if isinstance(save_path, str) else save_path / 'test.pth')
        test_dataset = CustomDataset(test_graphs, testset['labels'], device)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = set_model(model_select, task, n_layers, emb_dim, feat_dim, dropout, pool)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = model.to(device)

    layer_list = []
    for name, param in model.named_parameters():
        if 'pred_head' in name:
            print(name, param.requires_grad)
            layer_list.append(name)

    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': init_base_lr}, {'params': params}],
        init_lr, weight_decay=weight_decay
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn = set_loss_fn(loss_select)

    best_epoch, best_loss = 0, 1e9
    train_loss_list, valid_loss_list, test_loss_list = [], [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {epochs}")
        train_loss, valid_loss = train_one_epoch(model, loss_fn, train_loader, valid_loader, optimizer, epoch, verbose)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if testset is not None:
            test_loss = predict_one_epoch(model, loss_fn, test_loader, epoch, verbose)
            test_loss_list.append(test_loss)

        scheduler.step()
        print(f'Epoch {epoch} / {epochs} : Learning rate :', scheduler.get_last_lr())

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
        model_path: str = 'gcn',
        model_select: str = "gcn",  # ka_gnn, mlp_sage, kan_sage, kan_sage_two

        task = 'regression',

        n_layers: int = 5,
        emb_dim: int = 300,
        feat_dim: int = 512,
        dropout: float = 0.1,
        pool: str = 'mean',

        batch_size: int = 128,
        verbose: bool = True,
):
    device = set_device()
    set_seed(seed=42)

    smis = pd.DataFrame(smis, columns=['smiles'])
    graphs = create_data(smis, save=False)
    graphs_dataset = CustomDataset(graphs, device=device)
    loader = DataLoader(graphs_dataset, batch_size=batch_size, shuffle=False)

    model = set_model(model_select, task, n_layers, emb_dim, feat_dim, dropout, pool)
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
    predictions = torch.cat(predictions).numpy()
    return predictions
