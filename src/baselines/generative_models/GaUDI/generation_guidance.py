import json
import os
import random
from datetime import datetime
from time import time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from torch import Tensor, randn
import matplotlib.pyplot as plt
from rdkit import Chem

from analyze.analyze import analyze_validity_for_molecules, analyze_rdkit_validity_for_molecules
from edm.equivariant_diffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
)
from models_edm import get_model
from cond_prediction.prediction_args import PredictionArgs
from cond_prediction.train_cond_predictor import get_cond_predictor_model
from sampling_edm import sample_guidance
from data.aromatic_dataloader import create_data_loaders

from utils.args_edm import Args_EDM
from utils.plotting import plot_graph_of_rings, plot_rdkit
from utils.helpers import switch_grad_off, get_edm_args, get_cond_predictor_args

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch


@torch.no_grad()
def predict(model, x, h, node_mask, edge_mask, edm_model) -> Tuple[Tensor, Tensor]:
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    pred = model(xh, node_mask, edge_mask, t)
    return pred


@torch.no_grad()
def get_target_function_values(x, h, target_function, node_mask, edge_mask, edm_model):
    bs, n_nodes, n_dims = x.size()
    # edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    # edge_mask = edge_mask.repeat(bs, 1, 1).view(-1, 1).to(args.device)
    # node_mask = torch.ones(bs, n_nodes, 1).to(args.device)
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
    node_mask = node_mask.view(bs, n_nodes, 1)

    t = torch.zeros(x.size(0), 1, device=x.device).float()
    x, h, _ = edm_model.normalize(
        x,
        {"categorical": h, "integer": torch.zeros(0).to(x.device)},
        node_mask,
    )
    xh = torch.cat([x, h["categorical"]], dim=-1)
    return target_function(xh, node_mask, edge_mask, t)


def eval_stability(x, one_hot, node_mask, edge_mask, dataset="cata"):
    bs, n, _ = x.shape
    atom_type = one_hot.argmax(2).cpu().detach()
    molecule_list = [(x[i][node_mask[i, :, 0].bool()].cpu().detach(), atom_type[i]) for i in range(x.shape[0])]

    stability_dict, molecule_stable_list = analyze_validity_for_molecules(
        molecule_list, dataset=args.dataset
    )
    print(f"Stability for {args.exp_dir}")
    for key, value in stability_dict.items():
        try:
            print(f"   {key}: {value:.2%}")
        except:
            pass

    stability_dict, molecule_stable_list = analyze_rdkit_validity_for_molecules(
        molecule_list, dataset=dataset, calc_novelty=True, molecule_connected_bool=stability_dict['molecule_connected_bool']
    )
    x = x[stability_dict["molecule_valid_bool"]]
    one_hot = one_hot[stability_dict["molecule_valid_bool"]]
    node_mask = node_mask[stability_dict["molecule_valid_bool"]]
    edge_mask = edge_mask.view(bs, n, n)[stability_dict["molecule_valid_bool"]].view(-1, 1)
    return stability_dict, x, one_hot, node_mask, edge_mask


def design(
    args, model, cond_predictor, target_function, nodes_dist, prop_dist, scale, n_nodes
):
    switch_grad_off([model, cond_predictor])
    model.eval()
    cond_predictor.eval()

    print()
    print("Design molecule...")
    nodesxsample = Tensor([n_nodes] * args.batch_size).long()
    # nodesxsample = nodes_dist.sample(args.batch_size)

    # sample molecules - guidance generation
    start_time = time()
    # sample molecules
    x, one_hot, node_mask, edge_mask = sample_guidance(
        args,
        model,
        target_function,
        nodesxsample,
        scale=scale,
    )
    print(f"Generated {x.shape[0]} molecules in {time() - start_time:.2f} seconds")
    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)

    # evaluate stability
    stability_dict, x_stable, atom_type_stable, _, _ = eval_stability(
        x, one_hot, node_mask, edge_mask, dataset=args.dataset
    )
    print(f"{scale=}")
    print(f"{stability_dict['mol_valid']=:.2%} out of {x.shape[0]}")

    # evaluate target function values and prediction
    target_function_values = (
        get_target_function_values(
            x, one_hot, target_function, node_mask, edge_mask, model
        )
        .detach()
        .cpu()
    )
    pred = (
        predict(cond_predictor, x, one_hot, node_mask, edge_mask, model).detach().cpu()
    )
    pred = prop_dist.unnormalize(pred)

    print(f"Mean target function value: {target_function_values.mean().item():.4f}")

    timestamp = datetime.now().strftime("%m%d_%H_%M_%S")
    dir_name = f"best/{timestamp}_{scale}"
    os.makedirs("best", exist_ok=True)
    os.makedirs(dir_name, exist_ok=True)

    # find best molecule - can be unvalid
    # best_idx = target_function_values.min(0).indices.item()
    # best_value = target_function_values[best_idx]
    # atom_type = one_hot.argmax(2).cpu().detach()
    # print(f"best value: {best_value}, pred: {pred[best_idx]}")
    # best_str = ", ".join([f"{t:.3f}" for t in pred[best_idx]])
    # plot_graph_of_rings(
    #     x[best_idx].detach().cpu(),
    #     atom_type[best_idx].detach().cpu(),
    #     filename=f"{dir_name}/all.png",
    #     title=f"{best_str}\n {best_value}",
    #     dataset=args.dataset,
    # )

    # find best valid molecules
    pred = pred[stability_dict["molecule_valid_bool"]]
    target_function_values = target_function_values[
        stability_dict["molecule_valid_bool"]
    ]
    print(
        f"Mean target function value (from valid): {target_function_values.mean().item():.4f}"
    )

    # save the generated molecules
    result = pd.DataFrame()
    result['smi'] = [Chem.MolToSmiles(Chem.MolFromInchi(inchi)) for inchi in stability_dict['valid_inchi']]
    result_data = pd.DataFrame(pred.numpy(), columns=["PCE-D18-IC_DH","PCE-D18-IC2F_OB","PCE-D18-IC2Cl_HE"])
    result = pd.concat([result, result_data], axis=1)
    result.to_csv(f"best/{timestamp}_{scale}/result-mol.csv", index=False)

    best_idxs = target_function_values.argsort()
    for i in range(min(5, target_function_values.shape[0])):
        idx = best_idxs[i]
        value = target_function_values[idx]
        print(f"best value (from stable): {pred[idx]}, " f"score: {value}")

        best_str = ", ".join([f"{t:.3f}" for t in pred[idx]])
        plot_graph_of_rings(
            x_stable[idx].detach().cpu(),
            atom_type_stable[idx].argmax(1).detach().cpu(),
            filename=f"{dir_name}/{i}.pdf",
            title=f"{best_str}\n {value}",
            dataset=args.dataset,
        )
        plot_rdkit(
            x_stable[idx].detach().cpu(),
            atom_type_stable[idx].argmax(1).detach().cpu(),
            filename=f"{dir_name}/mol_{i}.pdf",
            title=f"{best_str}\n {value}",
            dataset=args.dataset,
        )

    # plot target function values histogram
    plt.close()
    plt.hist(target_function_values.numpy().squeeze(), density=True)
    plt.show()


def main(args, cond_predictor_args):
    # Set controllable parameters
    args.batch_size = 1000   # number of molecules to generate
    scale = 0.8             # gradient scale - the guidance strength
    n_nodes = 8            # number of rings in the generated molecules

    # Load models
    train_loader, _, _ = create_data_loaders(cond_predictor_args)
    model, nodes_dist, prop_dist = get_model(args, train_loader)
    cond_predictor = get_cond_predictor_model(cond_predictor_args, train_loader.dataset)

    # define target function - attached two examples for the max gap and OPV target functions.
    # You can create any target function of the predicted properties.
    def target_function_max_gap(_input, _node_mask, _edge_mask, _t):
        pred = cond_predictor(_input, _node_mask, _edge_mask, _t)
        gap = pred[:, 1]
        return -gap

    def target_function_opv(_input, _node_mask, _edge_mask, _t):
        pred = cond_predictor(_input, _node_mask, _edge_mask, _t)
        # pred = prop_dist.unnormalize(pred)
        IC_DH = pred[:, 0]
        IC2F_OB = pred[:, 1]
        IC2Cl_HE = pred[:, 2]
        return - (IC_DH + IC2F_OB + IC2Cl_HE) / 3

    design(
        args,
        model,
        cond_predictor,
        target_function_opv,
        nodes_dist,
        prop_dist,
        scale,
        n_nodes,
    )


if __name__ == "__main__":
    # load arguments of the pre-trained EDM and conditional predictor models
    args = get_edm_args(r'summary/hetro-test')
    cond_predictor_args = get_cond_predictor_args(
        r'prediction_summary/hetro-test'
    )

    # Where the magic is
    for _ in range(50):
        try:
            main(args, cond_predictor_args)
        except:
            pass
