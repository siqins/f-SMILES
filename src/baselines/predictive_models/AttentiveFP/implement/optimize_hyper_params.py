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


def f(radius, T, fingerprint_dim, weight_decay, learning_rate, p_dropout, direction=False):
    loss_function = nn.MSELoss()
    loss_function.cuda()
    model = Fingerprint(int(round(radius)), int(round(T)), num_atom_features, num_bond_features,
                        int(round(fingerprint_dim)), output_units_num, p_dropout)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)

    best_param = {}
    best_param["train_epoch"] = 0
    best_param["test_epoch"] = 0
    best_param["train_MSE"] = 9e8
    best_param["test_MSE"] = 9e8
    for epoch in range(800):
        train(model, train_df, optimizer, loss_function, epoch + 1)
        train_MAE, train_MSE = eval(model, train_df)
        test_MAE, test_MSE = eval(model, test_df)
        if train_MSE < best_param["train_MSE"]:
            best_param["train_epoch"] = epoch
            best_param["train_MSE"] = train_MSE
        if test_MSE < best_param["test_MSE"]:
            best_param["test_epoch"] = epoch
            best_param["test_MSE"] = test_MSE
        if (epoch - best_param["train_epoch"] > 6) and (epoch - best_param["test_epoch"] > 8):
            break
    # print(best_param["test_epoch"], best_param["test_MSE"])
    with open(log_file, 'a') as f:
        f.write(','.join([str(int(round(radius))), str(int(round(T))), str(int(round(fingerprint_dim))), str(p_dropout),
                          str(weight_decay), str(learning_rate)]))
        f.write(',' + str(best_param["test_epoch"]) + ',' + str(best_param["test_MSE"]) + '\n')

    # GPGO maximize performance by default, set performance to its negative value for minimization
    if direction:
        return best_param["test_MSE"]
    else:
        return -best_param["test_MSE"]


from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

cov = matern32()
gp = GaussianProcess(cov)
acq = Acquisition(mode='UCB')
param = {
    'radius': ('int', [2, 6]),
    'T': ('int', [1, 5]),
    'fingerprint_dim': ('int', [30, 300]),
    'weight_decay': ('cont', [2, 6]),
    'learning_rate': ('cont', [2, 5]),
    'p_dropout': ('cont', [0, 0.5])
}
np.random.seed(168)
gpgo = GPGO(gp, acq, f, param)
gpgo.run(max_iter=30, init_evals=2)

# hp_opt, valid_performance_opt = gpgo.getResult()