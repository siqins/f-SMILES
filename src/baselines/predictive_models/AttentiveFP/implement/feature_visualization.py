import time
import copy
import pickle
import pathlib
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import torch
from IPython.display import display, SVG
from AttentiveFP_codes import (Fingerprint, Fingerprint_viz, save_smiles_dicts,
                               get_smiles_dicts, get_smiles_array, moltosvg_highlight)


def min_max_norm(dataset):
    if isinstance(dataset, list):
        norm_list = list()
        min_value = min(dataset)
        max_value = max(dataset)

        for value in dataset:
            tmp = (value - min_value) / (max_value - min_value)
            norm_list.append(tmp)
    return norm_list


def visualize(
        model_path: str,
        dataset: pd.DataFrame,
        batch_size: int=200,
        save_path: str=None,
        attention_weights: bool=True,
        feature_relations: bool=True,
        annot: bool=True, # annot for feature relations
):
    # set model
    model = torch.load(model_path, weights_only=False, map_location='cpu')
    model_dict = model.state_dict()
    model_wts = copy.deepcopy(model_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visual_model = Fingerprint_viz(
        radius=model.radius,
        T=model.T,
        input_feature_dim=model.input_feature_dim,
        input_bond_dim=model.input_bond_dim,
        fingerprint_dim=model.fingerprint_dim,
        output_units_num=model.output_units_num,
        p_dropout=model.p_dropout,
    )
    visual_model.load_state_dict(model_wts)
    (model.align[0].weight == visual_model.align[0].weight).all()

    visual_model.to(device)
    visual_model.eval()

    # set dataset
    feats_dicts = save_smiles_dicts(dataset.smiles.tolist())

    test_MAE_list = []
    test_MSE_list = []

    out_feature_sorted = []
    out_weight_sorted = []
    mol_feature_sorted = []

    test_MAE_list = []
    test_MSE_list = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for idx, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feats_dicts)
        (atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz,
         mol_attention_weight_viz, mol_prediction) = visual_model(torch.Tensor(x_atom).to(device),
                                                                  torch.Tensor(x_bonds).to(device),
                                                                  torch.LongTensor(x_atom_index).to(device),
                                                                  torch.LongTensor(x_bond_index).to(device),
                                                                  torch.Tensor(x_mask).to(device)
                                                                  )

        mol_pred = np.array(mol_prediction.data.squeeze().cpu().numpy())
        atom_feature = np.stack([atom_feature_viz[L].cpu().detach().numpy() for L in range(model.radius + 1)])
        atom_weight = np.stack([mol_attention_weight_viz[t].cpu().detach().numpy() for t in range(model.T)])
        mol_feature = np.stack([mol_feature_viz[t].cpu().detach().numpy() for t in range(model.T)])

        mol_feature_sorted.extend([mol_feature[:, i, :] for i in range(mol_feature.shape[1])])

        # attention weights of atoms
        for i, smiles in enumerate(smiles_list):
            atom_num = i
            ind_mask = x_mask[i]
            ind_atom = smiles_to_rdkit_list[smiles]
            ind_feature = atom_feature[:, i]
            ind_weight = atom_weight[:, i]
            out_feature = []
            out_weight = []
            for j, one_or_zero in enumerate(list(ind_mask)):
                if one_or_zero == 1.0:
                    out_feature.append(ind_feature[:, j])
                    out_weight.append(ind_weight[:, j])
            feature_reorder = np.stack([out_feature[m] for m in np.argsort(ind_atom)])
            weight_reorder = np.stack([out_weight[m] for m in np.argsort(ind_atom)])
            # out_feature_sorted.extend([out_feature[m] for m in np.argsort(ind_atom)])
            # out_weight_sorted.extend([out_weight[m] for m in np.argsort(ind_atom)])
            out_feature_sorted.extend(feature_reorder)
            out_weight_sorted.extend(weight_reorder)

            mol = Chem.MolFromSmiles(smiles)

            if attention_weights:
                aromatic_boolean = [int(mol.GetAtomWithIdx(i).GetIsAromatic()) for i in range(mol.GetNumAtoms())]
                print(len(aromatic_boolean))
                if len(aromatic_boolean) > 30 and np.sum(aromatic_boolean) > 3:
                    weight_norm = min_max_norm([out_weight[m][0] for m in np.argsort(ind_atom)])

                    print('aromatic atom count: ' + str(np.sum(aromatic_boolean)) + '\n' 
                          'prediction: ' + str(mol_pred[atom_num]))
                    print(smiles)
                    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
                    cmap = cm.get_cmap('Oranges')
                    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
                    atom_colors = {}
                    weight_norm = np.array(weight_norm).flatten()
                    threshold = weight_norm[np.argsort(weight_norm)[6]]
                    weight_norm = np.where(weight_norm < threshold, 0, weight_norm)

                    for i_ in range(len(ind_atom)):
                        atom_colors[i_] = plt_colors.to_rgba(float(weight_norm[i_]))
                    rdDepictor.Compute2DCoords(mol)

                    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
                    drawer.SetFontSize(1)
                    op = drawer.drawOptions()

                    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
                    drawer.DrawMolecule(mol, highlightAtoms=range(0, len(ind_atom)), highlightBonds=[],
                                        highlightAtomColors=atom_colors)
                    drawer.FinishDrawing()
                    svg = drawer.GetDrawingText()
                    svg2 = svg.replace('svg:', '')
                    # svg3 = SVG(svg2)
                    # display(svg3)

                    if save_path is not None:
                        svg_filename = f"{save_path}/smi_{i}.svg"
                        with open(svg_filename, 'w') as f:
                            f.write(svg2)
                        print(f"SVG文件已保存: {svg_filename}")

            if feature_relations:
                draw_index = list(range(len(ind_atom)))

                drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
                drawer.SetFontSize(0.56)
                op = drawer.drawOptions()
                for index, re_index in enumerate(draw_index):
                    op.atomLabels[index] = mol.GetAtomWithIdx(index).GetSymbol() + str(re_index)

                mol = rdMolDraw2D.PrepareMolForDrawing(mol)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                svg2 = svg.replace('svg:', '')
                # svg3 = SVG(svg2)
                # display(svg3)

                if save_path is not None:
                    svg_filename = f"{save_path}/smi_features_{i}.svg"
                    with open(svg_filename, 'w') as f:
                        f.write(svg2)
                    print(f"SVG文件已保存: {svg_filename}")

                intra_mol_correlation = [np.corrcoef(feature_reorder[:, L]) for L in range(model.radius + 1)]

                for L in range(model.radius + 1):
                    plt.figure(dpi=300)
                    fig, ax = plt.subplots(figsize=(20, 16))
                    mask = np.zeros_like(intra_mol_correlation[L])
                    mask[np.triu_indices_from(mask)] = False
                    sns.heatmap(np.around(intra_mol_correlation[L], 1), cmap="YlGnBu", annot=annot,
                                ax=ax, mask=mask,
                                square=True, annot_kws={"size": 16})
                    svg_filename = f"{save_path}/smi_{i}_features_relations_radius_{L}.png"
                    plt.savefig(svg_filename, dpi=500)
                    plt.close()


if __name__ == '__main__':

    tasks = ['task']

    smis = ['CC', 'CCOC', 'CCCCCOC', 'CCSOCC',
            'O=C1C2=C(C=CC=C2)C(/C1=C/C(C3)=CC4=C3C=CC5=C4C=CC6=C5C=CC=C6)=C(C#N)/C#N']
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]

    train_dataset = pd.DataFrame()
    train_dataset['smiles'] = smis
    train_dataset['task'] = range(len(smis))

    visualize(
        model_path='saved_models/model_reg_st.pt',
        dataset=train_dataset,
        save_path='saved_models',
        attention_weights=False,
        feature_relations=True,
    )