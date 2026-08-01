import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.baselines.predictive_models.baseline_models import parse_data_for_predictive_models, train_predictive_models, inference_predictive_models
import warnings
warnings.filterwarnings('ignore')

def train():

    raw_data = pd.read_csv(r'../datasets/fused_units_generated_calculated.csv')

    prop_names = ['lumo', 'gap', 'energy', 'ionization_potential aip', 'electron_affinity aea']
    props_mapper = {
        'lumo': 'LUMO',
        'gap': 'HLG',
        'energy': 'Erel',
        'ionization_potential aip': 'IP',
        'electron_affinity aea': 'EA',
    }

    model_selects = ['MPNN', 'GAT', 'AttentiveFP', 'FraGAT', 'GraphMVP', 'MolCLR-GIN', 'KANO']
    for model_select in model_selects:
        for prop_name in prop_names:
            for _ in range(3):
                print(f'Training model {model_select} for prop {prop_name}')
                n_smis = len(raw_data)
                n_split = int(0.9 * n_smis)
                props = raw_data[prop_name].values
                train_props = props[:n_split]
                valid_props = props[n_split:]
                mean_, std_ = train_props.mean(), train_props.std()
                train_smis = raw_data['smiles'][:n_split].tolist()
                valid_smis = raw_data['smiles'][n_split:].tolist()
                train_props = (train_props - mean_) / std_
                valid_props = (valid_props - mean_) / std_

                data = parse_data_for_predictive_models(train_smis, train_props, valid_smis, valid_props)

                save_path = rf'../experiments/prediction_baselines/{props_mapper[prop_name]}/{_}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                train_predictive_models(
                    model_select=model_select,
                    parsed_data=data,
                    epochs=50,
                    save_path=save_path,
                    batch_size=64,

                    learning_rate = 6e-4,
                    weight_decay = 0,
                )

def inference():
    model_selects = ['MPNN', 'GAT', 'AttentiveFP', 'FraGAT', 'GraphMVP', 'MolCLR-GIN', 'KANO']

    raw_data = pd.read_csv(r'../datasets/fused_units_generated_calculated.csv')
    props = ['lumo', 'gap', 'energy', 'ionization_potential aip', 'electron_affinity aea']
    props_mapper = {
        'lumo': 'LUMO',
        'gap': 'HLG',
        'energy': 'Erel',
        'ionization_potential aip': 'IP',
        'electron_affinity aea': 'EA',
    }

    for model_select in model_selects:
        for prop in props:
            for idx in [0, 1, 2]:
                model_name = 'model.pth'
                raw_model_path = rf'../experiments/prediction_baselines/{props_mapper[prop]}/{idx}/{model_select}'

                model_path = os.path.join(raw_model_path, f'{model_name}')

                n_smis = len(raw_data)
                n_split = int(0.9 * n_smis)
                props_values = raw_data[prop].values
                train_props = props_values[:n_split]
                valid_props = props_values[n_split:]
                mean_, std_ = train_props.mean(), train_props.std()
                train_smis = raw_data['smiles'][:n_split].tolist()
                valid_smis = raw_data['smiles'][n_split:].tolist()

                print('Prediction.')
                prediction = inference_predictive_models(
                    model_select=model_select,
                    smiles=valid_smis,
                    model_path=model_path
                )
                prediction = np.array(prediction) * std_ + mean_
                df = pd.DataFrame({
                    'prediction': prediction,
                    'labels': valid_props
                })
                df.to_csv(os.path.join(raw_model_path, f'{model_name}.csv'), index=False)
                print(model_select, prop, ' r2_score : ', r2_score(df['labels'].values, df['prediction'].values))


def train_rfr():
    from rdkit import Chem
    from tqdm import tqdm
    from rdkit.Chem import MACCSkeys, AllChem
    from sklearn.ensemble import RandomForestRegressor

    def calc_fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        ec_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        return np.array(list(map(int, list(ec_fp))))

    raw_data = pd.read_csv(r'../data/fused_units_generated_calculated.csv')

    prop_names = ['lumo', 'gap', 'energy', 'ionization_potential aip', 'electron_affinity aea']
    props_mapper = {
        'lumo': 'LUMO',
        'gap': 'HLG',
        'energy': 'Erel',
        'ionization_potential aip': 'IP',
        'electron_affinity aea': 'EA',
    }

    for prop_name in prop_names:
        for _ in range(3):
            print(f'Training model RFR for prop {prop_name}')
            n_smis = len(raw_data)
            n_split = int(0.9 * n_smis)
            props = raw_data[prop_name].values
            train_props = props[:n_split]
            valid_props = props[n_split:]
            mean_, std_ = train_props.mean(), train_props.std()
            train_smis = raw_data['smiles'][:n_split].tolist()
            valid_smis = raw_data['smiles'][n_split:].tolist()
            train_props = (train_props - mean_) / std_

            fps_train = [calc_fp(smi) for smi in tqdm(train_smis)]
            fps_valid = [calc_fp(smi) for smi in tqdm(valid_smis)]
            fps_train = np.array(fps_train)
            fps_valid = np.array(fps_valid)

            model = RandomForestRegressor(random_state=_)
            model.fit(fps_train, train_props)

            prediction = model.predict(fps_valid)
            prediction = np.array(prediction) * std_ + mean_

            print('RFR', prop_name, ' r2_score : ', r2_score(valid_props, prediction))



if __name__ == '__main__':

    train()
    inference()
    train_rfr()