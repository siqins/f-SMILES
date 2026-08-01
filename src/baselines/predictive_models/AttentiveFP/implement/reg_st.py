import time
import pickle
import pathlib
from collections import defaultdict
import numpy as np
import pandas as pd
from torch_geometric.utils import smiles
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from AttentiveFP_codes import (Fingerprint, Fingerprint_viz, save_smiles_dicts,
                               get_smiles_dicts, get_smiles_array, moltosvg_highlight)


def parse_data(smis, values):
    data = pd.DataFrame()
    data['smiles'] = smis
    data['task'] = values
    return data


def train(model, dataset, optimizer, loss_function, feature_dicts, tasks, batch_size, epoch=0, verbose=False, device='cpu'):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0, dataset.shape[0])
    # shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)

    loop = enumerate(tqdm(batch_list)) if verbose else enumerate(batch_list)
    for counter, batch in loop:
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.smiles.values
        y_val = batch_df[tasks[0]].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = (torch.Tensor(x_atom).to(device),
                                                               torch.Tensor(x_bonds).to(device),
                                                               torch.LongTensor(x_atom_index).to(device),
                                                               torch.LongTensor(x_bond_index).to(device),
                                                               torch.Tensor(x_mask).to(device))

        atoms_prediction, mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

        optimizer.zero_grad()
        loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1, 1).to(device))
        loss.backward()
        optimizer.step()


def eval(model, dataset, feature_dicts, tasks, batch_size, return_predictions=False, device='cpu'):
    model.eval()
    test_MAE_list = []
    test_MSE_list = []
    test_prediction_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.smiles.values
        y_val = batch_df[tasks[0]].values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = (torch.Tensor(x_atom).to(device),
                                                               torch.Tensor(x_bonds).to(device),
                                                               torch.LongTensor(x_atom_index).to(device),
                                                               torch.LongTensor(x_bond_index).to(device),
                                                               torch.Tensor(x_mask).to(device))

        atoms_prediction, mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)

        # use reduction='none' to get loss for every item,
        # then they are extended to the list, and get their average finally.
        MAE = F.l1_loss(mol_prediction, torch.Tensor(y_val).view(-1, 1).to(device), reduction='none')
        MSE = F.mse_loss(mol_prediction, torch.Tensor(y_val).view(-1, 1).to(device), reduction='none')
        #         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)

        test_prediction_list.append(mol_prediction.squeeze().cpu().detach().numpy())
        test_MAE_list.extend(MAE.data.squeeze().cpu().numpy())
        test_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
    if return_predictions:
        return np.concatenate(test_prediction_list), np.array(test_MAE_list).mean(), np.array(test_MSE_list).mean(), r2_score(dataset[tasks[0]].values, np.concatenate(test_prediction_list))
    return np.array(test_MAE_list).mean(), np.array(test_MSE_list).mean(), r2_score(dataset[tasks[0]].values, np.concatenate(test_prediction_list))


def predict(model, dataset, feature_dicts, batch_size, device='cpu'):
    model.eval()
    test_prediction_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = (torch.Tensor(x_atom).to(device),
                                                               torch.Tensor(x_bonds).to(device),
                                                               torch.LongTensor(x_atom_index).to(device),
                                                               torch.LongTensor(x_bond_index).to(device),
                                                               torch.Tensor(x_mask).to(device))

        atoms_prediction, mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        test_prediction_list.append(mol_prediction.squeeze().cpu().detach())

    test_prediction_list = [tensor if tensor.dim() > 0 else tensor.unsqueeze(0) for tensor in test_prediction_list]

    return torch.cat(test_prediction_list).numpy()


def train_AFP(
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        prefix_filename: str='property',
        feats_dicts_save_name: str='feats',
        save_path='saved_models',

        lr=0.0001,
        weight_decay=10**-4.3,
        optimizer='adam',
        loss='mse',

        epochs: int = 800,
        batch_size: int = 200,
        verbose: bool = False,
        patience_train: int = 8,
        patience_test: int = 18,

        radius: int = 2,
        T: int = 1,
        p_dropout: float = 0.1,
        fingerprint_dim: int = 200,
) -> None:
    '''
    :param train_dataset: the names of columns contain smiles, and the rest of tasks
    :param test_dataset:
    :param task_type: regression or classification
    :param prefix_filename: prefix filename
    :param feats_dicts_save_name: the file name of feats_dicts
    :param save_path:
    :param prefix_filename:
    :param lr:
    :param weight_decay:
    :param optimizer:
    :param loss:
    :param epochs:
    :param batch_size:
    :param verbose:
    :param patience_train: the training patience
    :param patience_test: the testing patience
    :param radius:
    :param T: parameter for the number of layers in the AFP model.
    :param p_dropout:
    :param fingerprint_dim:
    :return:
    '''

    # set the save_path
    save_path = pathlib.Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # initial time
    start_time = str(time.ctime()).replace(':', '-').replace(' ', '_')

    # initial process for the input molecules
    smis = train_dataset.smiles.tolist() + test_dataset.smiles.tolist()
    if pathlib.Path(f"{save_path}/{feats_dicts_save_name}.pickle").exists():
        feats_dicts = pickle.load(open(f"{save_path}/{feats_dicts_save_name}.pickle", "rb"))
    else:
        feats_dicts = save_smiles_dicts(smis, f"{save_path}/{feats_dicts_save_name}")

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smis[0]], feats_dicts)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]
    per_task_output_units_num = 1
    tasks = train_dataset.columns.tolist()
    tasks.remove('smiles')
    output_units_num = len(tasks) * per_task_output_units_num

    print('num_atom_features', num_atom_features)
    print('num_bond_features', num_bond_features)

    # set model
    model = Fingerprint(
        input_feature_dim=num_atom_features,
        input_bond_dim=num_bond_features,
        output_units_num=output_units_num,
        radius=radius,
        T=T,
        fingerprint_dim=fingerprint_dim,
        p_dropout=p_dropout,
    )
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum(np.prod(p.size()) for p in model_parameters)
    print('Numbers of parameters in the model', params)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    # set optimizer
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be either adam or sgd')

    # set loss
    if loss == 'mse':
        loss = nn.MSELoss()
    elif loss == 'mae':
        loss = nn.L1Loss()
    else:
        raise ValueError('loss must be either mse or mae.')

    # training process
    best_params = {}
    best_params['train_epoch'] = 0
    best_params['test_epoch'] = 0
    best_params['train_MSE'] = 9e8
    best_params['test_MSE'] = 9e8
    metric_logger = defaultdict(list)

    for epoch in range(epochs):
        train_MAE, train_MSE, train_r2 = eval(
            model=model,
            dataset=train_dataset,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
        )

        test_MAE, test_MSE, test_r2 = eval(
            model=model,
            dataset=test_dataset,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
        )

        metric_logger['epoch'].append(epoch)
        metric_logger['train_MAE'].append(train_MAE)
        metric_logger['train_MSE'].append(train_MSE)
        metric_logger['train_r2'].append(train_r2)
        metric_logger['test_MAE'].append(test_MAE)
        metric_logger['test_MSE'].append(test_MSE)
        metric_logger['test_r2'].append(test_r2)
        logger_save_name = 'model_' + prefix_filename + '_' + start_time + '.csv'
        metric_logger_pd = pd.DataFrame(metric_logger)
        metric_logger_pd.to_csv(save_path / logger_save_name, index=False)

        if train_MSE < best_params['train_MSE']:
            best_params['train_epoch'] = epoch
            best_params['train_MSE'] = train_MSE
        if test_MSE < best_params['test_MSE']:
            best_params['test_epoch'] = epoch
            best_params['test_MSE'] = test_MSE
            model_save_name = 'model_' + prefix_filename + '_' + start_time + '_' + str(epoch) + '.pt'
            torch.save(model, save_path / model_save_name)
        if (epoch - best_params["train_epoch"] > patience_train) and (epoch - best_params["test_epoch"] > patience_test):
            break
        print('Epoch', epoch, 'Train MSE', train_MSE, 'Train r2', train_r2, 'Test MSE', test_MSE, 'Test r2', test_r2)

        train(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            loss_function=loss,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            epoch=epoch,
            verbose=verbose,
            device=device,
        )


def eval_AFP(
        eval_dataset: pd.DataFrame,
        feats_path: str='feats.pickle',
        model_path: str='saved_models',
        return_predictions: bool=False,
        batch_size: int = 200,
):
    tasks = train_dataset.columns.tolist()
    tasks.remove('smiles')
    feats_dict = pickle.load(open(feats_path, 'rb'))

    model = torch.load(model_path, weights_only=False)
    res = eval(
        model=model,
        dataset=eval_dataset,
        feature_dicts=feats_dict,
        tasks=tasks,
        batch_size=batch_size,
        return_predictions=return_predictions
    )
    if return_predictions:
        predictions, eval_MAE, eval_MSE, eval_r2 = res
        return predictions, eval_MAE, eval_MSE, eval_r2
    else:
        eval_MAE, eval_MSE, eval_r2 = res
        return eval_MAE, eval_MSE, eval_r2


def predict_AFP(
        tasks: list,
        smiles_list: list,
        model_path: str='saved_models',
        batch_size: int = 200,
):

    feats_dict = save_smiles_dicts(smiles_list)
    model = torch.load(model_path, weights_only=False, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    pred_dataset = pd.DataFrame()
    pred_dataset['smiles'] = smiles_list

    predictions = predict(
        model=model,
        dataset=pred_dataset,
        feature_dicts=feats_dict,
        batch_size=batch_size,
        device=device,
    )

    predictions = pd.DataFrame(predictions)
    predictions.columns = tasks
    predictions = pd.concat([pred_dataset, predictions], axis=1)
    return predictions



if __name__ == '__main__':

    from rdkit import Chem

    tasks = ['task']

    smis = ['CC', 'CCOC', 'CCCCCOC', 'CCSOCC',
            'O=C1C2=C(C=CC=C2)C(/C1=C/C(C3)=CC4=C3C=CC5=C4C=CC6=C5C=CC=C6)=C(C#N)/C#N']
    smis = ['CCOC']
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]

    train_dataset = pd.DataFrame()
    train_dataset['smiles'] = smis
    train_dataset['task'] = range(len(smis))

    test_dataset = pd.DataFrame()
    test_dataset['smiles'] = smis
    test_dataset['task'] = range(len(smis))

    train_AFP(train_dataset=train_dataset, test_dataset=test_dataset, epochs=3, save_path='saved_models')

    print()

    # print(eval_AFP(eval_dataset=test_dataset,
    #                model_path='saved_models/model_reg_st.pt',
    #                feats_path='saved_models/feats.pickle',
    #                return_predictions=True))

    print()
    print(predict_AFP(tasks=tasks,
                      smiles_list=smis,
                      model_path='saved_models/model_reg_st.pt',
                      ))