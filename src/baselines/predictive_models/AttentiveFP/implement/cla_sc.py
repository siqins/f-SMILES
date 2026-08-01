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
from sklearn.metrics import roc_auc_score
from AttentiveFP_codes import (Fingerprint, Fingerprint_viz, save_smiles_dicts,
                               get_smiles_dicts, get_smiles_array, moltosvg_highlight)


def train(model, dataset, optimizer, loss_functions, feature_dicts, tasks, batch_size,
          epoch=0, verbose=False, device='cpu', per_task_output_units_num: int=2):
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
            y_pred = mol_prediction[:, idx * per_task_output_units_num:(idx + 1) * per_task_output_units_num]
            y_val = batch_df[task].values
            # valid_inds = np.where((y_val==0) | (y_val==1))[0]
            valid_inds = np.where(y_val >= 0)[0]
            if len(valid_inds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in valid_inds]).astype(float)
            valid_inds = torch.LongTensor(valid_inds).squeeze().to(device)
            y_pred_adjust = torch.index_select(y_pred, 0, valid_inds)

            loss += loss_functions[idx](y_pred_adjust, torch.LongTensor(y_val_adjust).to(device))

        loss.backward()
        optimizer.step()


def eval(model, dataset, loss_functions, feature_dicts, tasks, batch_size,
         return_predictions=False, device='cpu', per_task_output_units_num: int=2):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for task in tasks:
        y_val_list[task] = []
        y_pred_list[task] = []
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
            y_pred = mol_prediction[:, idx * per_task_output_units_num:(idx + 1) * per_task_output_units_num]
            y_val = batch_df[task].values

            # valid_inds = np.where((y_val == 0) | (y_val == 1))[0]
            valid_inds = np.where(y_val >= 0)[0]
            if len(valid_inds) == 0:
                continue

            y_val_adjust = np.array([y_val[v] for v in valid_inds]).astype(float)
            valid_inds = torch.LongTensor(valid_inds).squeeze().to(device)
            y_pred_adjust = torch.index_select(y_pred, 0, valid_inds)

            a = torch.LongTensor(y_val_adjust).to(device)
            loss = loss_functions[idx](y_pred_adjust, a)
            y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
            losses_list.append(loss.cpu().detach().numpy())

            y_val_list[task].extend(y_val_adjust)
            y_pred_list[task].extend(y_pred_adjust)

    eval_roc = [roc_auc_score(y_val_list[task], y_pred_list[task]) for task in tasks]
    eval_loss = np.mean(losses_list)

    if return_predictions:
        return y_pred_list, eval_roc, eval_loss
    return eval_roc, eval_loss


def predict(model, dataset, feature_dicts, batch_size, device='cpu', per_task_output_units_num: int=2):
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
            y_pred = mol_prediction[:, idx * per_task_output_units_num: (idx + 1) * per_task_output_units_num]
            y_pred = F.softmax(y_pred, dim=-1).data.cpu().numpy()[:, 1]
            y_pred_list[task] = np.concatenate([y_pred_list[task], y_pred])

    return y_pred_list


def train_AFP(
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        per_task_output_units_num: int=2,
        prefix_filename: str='property',
        feats_dicts_save_name: str='feats',
        save_path: str='saved_models',

        lr=0.0001,
        weight_decay=10**-4.3,
        optimizer='adam',
        loss='cross_entropy',

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
    :param per_task_output_units_num: 2 for classification model with 2 classes, or other numbers for more classes
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
    tasks = train_dataset.columns.tolist()
    tasks.remove('smiles')
    output_units_num = len(tasks) * per_task_output_units_num

    print('num_atom_features', num_atom_features)
    print('num_bond_features', num_bond_features)

    # calc the weights for all tasks
    weights = []
    all_dataset = pd.concat([train_dataset, test_dataset])
    all_dataset.reset_index(drop=True, inplace=True)
    for idx, task in enumerate(tasks):
        negative_df = all_dataset[all_dataset[task] == 0]
        positive_df = all_dataset[all_dataset[task] == 1]
        weight = [(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0],
                  (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]]
        weights.append(weight)

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
    if loss == 'cross_entropy':
        loss = [nn.CrossEntropyLoss(torch.Tensor(weight).to(device), reduction='mean') for weight in weights]
    else:
        raise ValueError('loss must be cross_entropy')

    # training process
    best_params = {}
    best_params['roc_epoch'] = 0
    best_params['loss_epoch'] = 0
    best_params['test_roc'] = 0
    best_params['test_loss'] = 9e8
    metric_logger = defaultdict(list)

    for epoch in range(epochs):
        train_roc, train_loss = eval(
            model=model,
            dataset=train_dataset,
            loss_functions=loss,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
        )

        test_roc, test_loss = eval(
            model=model,
            dataset=test_dataset,
            loss_functions=loss,
            feature_dicts=feats_dicts,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
        )

        train_roc_mean = np.mean(train_roc)
        test_roc_mean = np.mean(test_roc)

        metric_logger['epoch'].append(epoch)
        metric_logger['train_roc'].append(train_roc_mean)
        metric_logger['train_loss'].append(train_loss)
        metric_logger['test_roc'].append(train_roc_mean)
        metric_logger['test_loss'].append(test_loss)
        logger_save_name = 'model_' + prefix_filename + '_' + start_time + '.csv'
        metric_logger_pd = pd.DataFrame(metric_logger)
        metric_logger_pd.to_csv(save_path / logger_save_name, index=False)

        if test_roc_mean > best_params['test_roc']:
            best_params['roc_epoch'] = epoch
            best_params['test_roc'] = test_roc_mean
        if test_loss < best_params['test_loss']:
            best_params['test_epoch'] = epoch
            best_params['test_loss'] = test_loss
            model_save_name = 'model_' + prefix_filename + '_' + start_time + '_' + str(epoch) + '.pt'
            torch.save(model, save_path / model_save_name)
        if (epoch - best_params["roc_epoch"] > patience_train) and (epoch - best_params["loss_epoch"] > patience_test):
            break
        print('Epoch', epoch, 'Train roc', train_roc_mean, 'Test roc', test_roc_mean)

        train(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            loss_functions=loss,
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

    # calc the weights for all tasks
    weights = []
    all_dataset = pd.concat([train_dataset, test_dataset])
    all_dataset.reset_index(drop=True, inplace=True)
    for idx, task in enumerate(tasks):
        negative_df = all_dataset[all_dataset[task] == 0]
        positive_df = all_dataset[all_dataset[task] == 1]
        weight = [(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0],
                  (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]]
        weights.append(weight)
    loss = [nn.CrossEntropyLoss(torch.Tensor(weight), reduction='mean') for weight in weights]

    model = torch.load(model_path, weights_only=False)
    res = eval(
        model=model,
        dataset=eval_dataset,
        loss_functions=loss,
        feature_dicts=feats_dict,
        tasks=tasks,
        batch_size=batch_size,
        return_predictions=return_predictions
    )
    if return_predictions:
        predictions, roc, loss = res
        return predictions, roc, loss
    else:
        roc, loss = res
        return roc, loss


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
        batch_size=batch_size,
        device=device,
    )

    predictions = pd.DataFrame(predictions)
    predictions.columns = tasks
    predictions = pd.concat([pred_dataset, predictions], axis=1)
    return predictions



if __name__ == '__main__':

    import random
    from rdkit import Chem

    tasks = ['task']

    smis = ['CC', 'CCOC', 'CCCCCOC', 'CCSOCC',
            'O=C1C2=C(C=CC=C2)C(/C1=C/C(C3)=CC4=C3C=CC5=C4C=CC6=C5C=CC=C6)=C(C#N)/C#N']
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]

    train_dataset = pd.DataFrame()
    train_dataset['smiles'] = smis
    train_dataset['task'] = [random.choice([0,1]) for _ in range(len(smis))]

    test_dataset = pd.DataFrame()
    test_dataset['smiles'] = smis
    test_dataset['task'] = [random.choice([0,1]) for _ in range(len(smis))]

    train_AFP(train_dataset=train_dataset, test_dataset=test_dataset, epochs=3, save_path='saved_models')

    print()

    print(eval_AFP(eval_dataset=test_dataset,
                   model_path='saved_models/model_cla_sc.pt',
                   feats_path='saved_models/feats.pickle',
                   return_predictions=True))

    print()
    print(predict_AFP(tasks=tasks,
                      smiles_list=smis,
                      model_path='saved_models/model_cla_sc.pt',
                      ))