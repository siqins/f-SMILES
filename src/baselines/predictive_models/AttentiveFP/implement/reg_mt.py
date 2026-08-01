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

def count_mt_ratio(tasks: list, dataset: pd.DataFrame) -> tuple[list[float], list[float]]:
    '''
    Count the ratio for the loss of various tasks,
    the provided values must be the original values to calculate the ratios,
    the values for model training could then be normalized after this.
    :return:
    '''
    ratios = []
    std_list = []
    for task in tasks:
        std = dataset[task].std()
        mad = dataset[task].mad()
        ratio = std / mad
        ratios.append(ratio)
        std_list.append(std)
    return ratios, std_list


def train(model, dataset, optimizer, loss_function, feature_dicts, tasks, batch_size,
          epoch=0, verbose=False, device='cpu', ratios=None):
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

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask = (torch.Tensor(x_atom).to(device),
                                                               torch.Tensor(x_bonds).to(device),
                                                               torch.LongTensor(x_atom_index).to(device),
                                                               torch.LongTensor(x_bond_index).to(device),
                                                               torch.Tensor(x_mask).to(device))

        atoms_prediction, mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
        optimizer.zero_grad()

        loss = 0.0
        for idx, task in enumerate(tasks):
            y_pred = mol_prediction[:, idx]
            y_val = batch_df[task].values
            partial_loss = loss_function(y_pred, torch.Tensor(y_val).squeeze().to(device))
            if ratios is not None:
                partial_loss = partial_loss * ratios[idx] ** 2
            loss += partial_loss
        loss.backward()
        optimizer.step()


def eval(model, dataset, feature_dicts, tasks, batch_size,
         return_predictions=False, device='cpu', std_list=None):
    model.eval()
    eval_MAE_list = {}
    eval_MSE_list = {}
    y_val_list = {}
    y_pred_list = {}
    for idx, task in enumerate(tasks):
        eval_MAE_list[task] = []
        eval_MSE_list[task] = []
        y_pred_list[task] = np.array([])
        y_val_list[task] = np.array([])

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

        for idx, task in enumerate(tasks):
            y_pred = mol_prediction[:, idx]
            y_val = batch_df[task].values

            # use reduction='none' to get loss for every item,
            # then they are extended to the list, and get their average finally.
            MAE = F.l1_loss(y_pred, torch.Tensor(y_val).squeeze().to(device), reduction='none')
            MSE = F.mse_loss(y_pred, torch.Tensor(y_val).squeeze().to(device), reduction='none')

            y_pred_list[task] = np.concatenate([y_pred_list[task], y_pred.cpu().detach().numpy()])
            y_val_list[task] = np.concatenate([y_val_list[task], y_val])
            eval_MAE_list[task] = np.concatenate([eval_MAE_list[task], MAE.data.squeeze().cpu().numpy()])
            eval_MSE_list[task] = np.concatenate([eval_MSE_list[task], MSE.data.squeeze().cpu().numpy()])

        r2_score_list = np.array([r2_score(y_val_list[task], y_pred_list[task]) for task in tasks])
        eval_MAE = np.array([eval_MAE_list[task].mean() for task in tasks])
        eval_MSE = np.array([eval_MSE_list[task].mean() for task in tasks])
        if std_list is not None:
            eval_MAE = np.multiply(eval_MAE, np.array(std_list))
            eval_MSE = np.multiply(eval_MSE, np.array(std_list))

    if return_predictions:
        return y_pred_list, r2_score_list, eval_MAE, eval_MSE
    else:
        return r2_score_list, eval_MAE, eval_MSE


def predict(model, dataset, feature_dicts, tasks, batch_size, device='cpu'):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    for idx, task in enumerate(tasks):
        y_pred_list[task] = np.array([])
        y_val_list[task] = np.array([])

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

        for idx, task in enumerate(tasks):
            y_pred = mol_prediction[:, idx]
            y_pred_list[task] = np.concatenate([y_pred_list[task], y_pred.cpu().detach().numpy()])

    return y_pred_list


def train_AFP(
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        prefix_filename: str='property',
        feats_dicts_save_name: str='feats',
        save_path: str='saved_models',

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

        ratios: list = None,
        std_list: list = None,
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
    :param ratios: for calculation of multi-task loss
    :param std_list: for calculation of multi-task loss
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
        raise ValueError('loss must be either mse or mae')

    # training process
    best_params = {}
    best_params['train_epoch'] = 0
    best_params['test_epoch'] = 0
    best_params['train_MSE'] = 9e8
    best_params['test_MSE'] = 9e8
    metric_logger = defaultdict(list)

    for epoch in range(epochs):
        train_r2_score_list, train_eval_MAE, train_eval_MSE = eval(
            model=model,
            dataset=train_dataset,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            std_list=std_list,
        )

        test_r2_score_list, test_eval_MAE, test_eval_MSE = eval(
            model=model,
            dataset=test_dataset,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            std_list=std_list,
        )

        metric_logger['epoch'].append(epoch)
        metric_logger['train_mae'].append(train_eval_MAE.mean().item())
        metric_logger['train_MSE'].append(train_eval_MSE.mean().item())
        metric_logger['train_r2'].append(train_r2_score_list.mean().item())
        metric_logger['test_MAE'].append(test_eval_MAE.mean().item())
        metric_logger['test_MSE'].append(test_eval_MSE.mean().item())
        metric_logger['test_r2'].append(test_r2_score_list.mean().item())
        logger_save_name = 'model_' + prefix_filename + '_' + start_time + '.csv'
        metric_logger_pd = pd.DataFrame(metric_logger)
        metric_logger_pd.to_csv(save_path / logger_save_name, index=False)

        if train_eval_MSE.mean() < best_params['train_MSE']:
            best_params['train_epoch'] = epoch
            best_params['train_MSE'] = train_eval_MSE.mean()
        if test_eval_MSE.mean() < best_params['test_MSE']:
            best_params['test_epoch'] = epoch
            best_params['test_MSE'] = test_eval_MSE.mean()
            model_save_name = 'model_' + prefix_filename + '_' + start_time + '_' + str(epoch) + '.pt'
            torch.save(model, save_path / model_save_name)
        if (epoch - best_params["train_epoch"] > patience_train) and (epoch - best_params["test_epoch"] > patience_test):
            break
        print('Epoch', epoch, 'Train MSE', train_eval_MSE.mean(), 'Train r2', train_r2_score_list.mean(),
              'Test MSE', test_eval_MSE.mean(), 'Test r2', test_r2_score_list.mean())

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
            ratios=ratios
        )


def eval_AFP(
        eval_dataset: pd.DataFrame,
        feats_path: str='feats.pickle',
        model_path: str='saved_models',
        return_predictions: bool=False,
        batch_size: int = 200,
        std_list: list = None,
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
        return_predictions=return_predictions,
        std_list=std_list,
    )
    if return_predictions:
        predictions, r2_score_list, eval_MAE, eval_MSE = res
        return predictions, r2_score_list, eval_MAE, eval_MSE
    else:
        r2_score_list, eval_MAE, eval_MSE = res
        return r2_score_list, eval_MAE, eval_MSE


def predict_AFP(
        tasks: list,
        smiles_list: list,
        model_path: str='saved_models',
        batch_size: int = 200,
):

    feats_dict = save_smiles_dicts(smiles_list)
    model = torch.load(model_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    pred_dataset = pd.DataFrame()
    pred_dataset['smiles'] = smiles_list

    predictions = predict(
        model=model,
        dataset=pred_dataset,
        feature_dicts=feats_dict,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
    )

    predictions = pd.DataFrame(predictions)
    predictions = pd.concat([pred_dataset, predictions], axis=1)
    return predictions


if __name__ == '__main__':

    from rdkit import Chem

    tasks = ['task-1', 'task-2']

    smis = ['CC', 'CCOC', 'CCCCCOC', 'CCSOCC',
            'O=C1C2=C(C=CC=C2)C(/C1=C/C(C3)=CC4=C3C=CC5=C4C=CC6=C5C=CC=C6)=C(C#N)/C#N']
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]

    train_dataset = pd.DataFrame()
    train_dataset['smiles'] = smis
    train_dataset['task-1'] = range(len(smis))
    train_dataset['task-2'] = range(len(smis))

    test_dataset = pd.DataFrame()
    test_dataset['smiles'] = smis
    test_dataset['task-1'] = range(len(smis))
    test_dataset['task-2'] = range(len(smis))

    train_AFP(train_dataset=train_dataset, test_dataset=test_dataset, epochs=3, patience_train=0, patience_test=0)
    print(eval_AFP(eval_dataset=test_dataset,
                   model_path='saved_models/model_reg_mt.pt',
                   feats_path='saved_models/feats.pickle',
                   return_predictions=True))

    print()
    print(predict_AFP(tasks=tasks,
                      smiles_list=smis,
                      model_path='saved_models/model_reg_mt.pt',
                      ))

