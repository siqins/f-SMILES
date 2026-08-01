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
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from GraphMVP_codes.src_regression.models_complete_feature import GNN_graphpredComplete, GNNComplete


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
        """ used in MoleculeDataset() class
        Converts rdkit mol objects to graph data object in pytorch geometric
        NB: Uses simplified atom and bond features, and represent as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr """

        # atoms
        # num_atom_features = 2  # atom type, chirality tag

        mol = Chem.MolFromSmiles(smiles)
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = atom_to_feature_vector(atom)
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        if len(mol.GetBonds()) <= 0:  # mol has no bonds
            num_bond_features = 3  # bond type & direction
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

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

def set_model(model_select, task, args):
    if model_select == 'graphmvp':
        molecule_model = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout, gnn_type=args.gnn_type)
        model = GNN_graphpredComplete(args=args, num_tasks=args.num_tasks, molecule_model=molecule_model)
        # model.from_pretrained(input_model_file)
    else:
        print('No Found the Model!')
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

class Args:
    def __init__(self,
                 gnn_type='gin',
                 n_layers=5,
                 emb_dim=300,
                 dropout=0.1,
                 graph_pooling='mean',
                 JK='last',
                 gnn_lr_scale=1,
                 model_3d='schnet',
                 mask_rate='0.15',
                 mask_edge=0.,
                 num_tasks=1
                 ):
        self.gnn_type = gnn_type
        self.num_layer = n_layers
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.graph_pooling = graph_pooling
        self.JK = JK
        self.gnn_lr_scale = gnn_lr_scale
        self.model_3d = model_3d
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.num_tasks = num_tasks


def train(
        trainset,
        validset,
        testset = None,
        save=False,
        save_path='experiments/processed_data',
        model_save_path: str = 'experiments/graphmvp',

        model_select: str = 'graphmvp',

        gnn_type='gin',
        n_layers=5,
        emb_dim=300,
        dropout=0.1,
        graph_pooling='mean',
        JK='last',
        gnn_lr_scale=1,
        model_3d='schnet',
        mask_rate='0.15',
        mask_edge=0.,
        num_tasks=1,

        loss_select: str = 'l2',
        task: str = 'regression',

        lr: float = 0.0005,
        lr_scale = 1.,
        decay: float = 0.0,
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

    args = Args(
        gnn_type=gnn_type,
        n_layers=n_layers,
        emb_dim=emb_dim,
        dropout=dropout,
        graph_pooling=graph_pooling,
        JK=JK,
        gnn_lr_scale=gnn_lr_scale,
        model_3d=model_3d,
        mask_rate=mask_rate,
        mask_edge=mask_edge,
        num_tasks=num_tasks,
    )

    model = set_model(model_select, task, args=args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = model.to(device)

    model_param_group = [
        {'params': model.molecule_model.parameters()},
        {'params': model.graph_pred_linear.parameters(), 'lr': lr * lr_scale}
    ]
    optimizer = torch.optim.Adam(model_param_group, lr=lr, weight_decay=decay)
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
        model_path: str = 'graphmvp',
        model_select: str = 'graphmvp',

        gnn_type='gin',
        n_layers=5,
        emb_dim=300,
        dropout=0.1,
        graph_pooling='mean',
        JK='last',
        gnn_lr_scale=1,
        model_3d='schnet',
        mask_rate='0.15',
        mask_edge=0.,
        num_tasks=1,

        task: str = 'regression',
        batch_size: int = 128,
        verbose: bool = True,
):
    device = set_device()
    set_seed(seed=42)

    smis = pd.DataFrame(smis, columns=['smiles'])
    graphs = create_data(smis, save=False)
    graphs_dataset = CustomDataset(graphs, device=device)
    loader = DataLoader(graphs_dataset, batch_size=batch_size, shuffle=False)

    args = Args(
        gnn_type=gnn_type,
        n_layers=n_layers,
        emb_dim=emb_dim,
        dropout=dropout,
        graph_pooling=graph_pooling,
        JK=JK,
        gnn_lr_scale=gnn_lr_scale,
        model_3d=model_3d,
        mask_rate=mask_rate,
        mask_edge=mask_edge,
        num_tasks=num_tasks,
    )

    model = set_model(model_select, task, args=args)
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
            preds = model(graphs).squeeze().squeeze()
            predictions.append(preds.detach().cpu())

    predictions = [tensor if tensor.dim() > 0 else tensor.unsqueeze(0) for tensor in predictions]
    predictions = torch.cat(predictions).numpy()
    return predictions
