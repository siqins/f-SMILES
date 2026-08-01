import pathlib
import json
import warnings
warnings.filterwarnings("ignore")


def parse_data_for_predictive_models(train_data, train_labels, valid_data, valid_labels,
                                     test_data=None, test_labels=None):
    # data : smiles
    # label: values
    data = {
        'train_data': train_data,
        'train_labels': train_labels,
        'valid_data': valid_data,
        'valid_labels': valid_labels,
        'test_data': test_data,
        'test_labels': test_labels,
    }

    return data

def train_predictive_models(
        model_select: str,
        parsed_data,
        epochs: int=200,
        batch_size: int=64,

        learning_rate: float=0.001,
        weight_decay=10 ** -4.3,
        gamma: float = 0.98,
        loss_select: str='l2',
        save_path: str='experiments',
        verbose: bool = False,
        patience: int = 20,
        num_workers=0,
        save: bool = False,
        **kwargs
):

    model_save_path = pathlib.Path(save_path) / model_select
    save_path = pathlib.Path(save_path) / model_select / 'processed_data'

    if model_select == 'AttentiveFP':

        from AttentiveFP.implement import reg_st

        train_dataset = reg_st.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = reg_st.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = reg_st.parse_data(parsed_data['test_data'], parsed_data['test_labels'])

        loss = 'mse' if loss_select == 'l2' else 'mae'

        reg_st.train_AFP(
            train_dataset=train_dataset,
            test_dataset=valid_dataset,
            save_path=model_save_path,

            lr=learning_rate,
            weight_decay=weight_decay,
            loss=loss,

            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            patience_test=patience,
            **kwargs
        )

    elif model_select == 'MPNN':

        from MPNN.implement import tune

        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        tune.train(
            trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,

            save_path=save_path,
            model_save_path=model_save_path,

            model_select='mpnn',
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            patience=patience,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )

    elif model_select == 'GAT':

        from GAT.implement import tune

        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        tune.train(
            trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,

            save_path=save_path,
            model_save_path=model_save_path,

            model_select='gat',
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            patience=patience,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )

    elif model_select == 'FraGAT':

        from FraGAT.implement import tune

        # make sure the save folder is empty.

        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        tune.train(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,

            RootPath=str(model_save_path),

            MaxEpoch=epochs,
            lr=learning_rate,
            WeightDecay=weight_decay,
            BatchSize=batch_size,

            **kwargs
        )

    elif model_select == 'GraphMVP':

        from GraphMVP.implement import tune

        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        res=tune.train(
            trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,
            save=save,
            save_path=save_path,
            model_save_path=model_save_path,

            model_select='graphmvp',

            lr=learning_rate,
            gamma=gamma,
            decay=weight_decay,
            loss_select=loss_select,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            patience=patience,
            gnn_type='gin', # gin gcn

            **kwargs
        )
        res.to_csv(model_save_path / 'results.csv', index=False)

    elif model_select == 'KANO':

        from KANO.implement import tune
        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        res = tune.train(
            trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,
            save=save,
            save_path=save_path,
            model_save_path=model_save_path,

            model_select='kano',

            init_lr=learning_rate,
            loss_select=loss_select,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            patience=patience,
            weight_decay=weight_decay,
            **kwargs
        )

        res.to_csv(model_save_path / 'results.csv', index=False)

    elif model_select == 'MolCLR-GIN':

        from MolCLR.imlplement import tune
        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        res = tune.train(
            trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,
            save=save,
            save_path=save_path,
            model_save_path=model_save_path,

            model_select='gin',

            init_lr=learning_rate,
            gamma=gamma,
            loss_select=loss_select,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            patience=patience,
            weight_decay=weight_decay,
            **kwargs
        )

        res.to_csv(model_save_path / 'results.csv', index=False)

    elif model_select == 'MolCLR-GCN':

        from MolCLR.imlplement import tune
        train_dataset = tune.parse_data(parsed_data['train_data'], parsed_data['train_labels'])
        valid_dataset = tune.parse_data(parsed_data['valid_data'], parsed_data['valid_labels'])
        if parsed_data['test_data']:
            test_dataset = tune.parse_data(parsed_data['test_data'], parsed_data['test_labels'])
        else:
            test_dataset = None

        res = tune.train(
            trainset=train_dataset,
            validset=valid_dataset,
            testset=test_dataset,
            save=save,
            save_path=save_path,
            model_save_path=model_save_path,

            model_select='gcn',

            init_lr=learning_rate,
            gamma=gamma,

            loss_select=loss_select,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
            patience=patience,
            weight_decay=weight_decay,
            **kwargs
        )

        res.to_csv(model_save_path / 'results.csv', index=False)

    else:
        raise ValueError('Invalid model_select for predictive models.')


def inference_predictive_models(
        model_select: str,
        smiles,
        model_path,
        batch_size: int=64,
        verbose: bool = False,
        num_workers: int=0,
        **kwargs
):

    if model_select == 'AttentiveFP':

        from AttentiveFP.implement import reg_st

        tasks = ['task']
        predictions = reg_st.predict_AFP(
            tasks=tasks,
            smiles_list=smiles,
            model_path=model_path, # model_path='saved_models/model_reg_st.pt'
            batch_size=batch_size,
        )
        predictions = predictions[tasks[0]].tolist()

    elif model_select == 'MPNN':

        from MPNN.implement import tune
        predictions = tune.predict(
            smis=smiles,
            model_path=model_path,
            model_select='mpnn', # 'experiments/mpnn/last.ckpt'
            batch_size=batch_size,
            num_workers=num_workers,
        )

    elif model_select == 'GAT':

        from GAT.implement import tune
        predictions = tune.predict(
            smis=smiles,
            model_path=model_path,
            model_select='gat', # 'experiments/mpnn/last.ckpt'
            batch_size=batch_size,
            num_workers=num_workers,
        )

    elif model_select == 'FraGAT':

        from FraGAT.implement import tune
        from FraGAT.FraGAT_codes.Config import Configs

        config_path = pathlib.Path(model_path).parent.parent / 'config.json'
        with open(config_path, 'r') as f:
            opt = json.load(f)
        opt = Configs(opt)

        predictions = tune.predict(
            opt=opt,
            smiles_list=smiles,
            model_path=model_path,
        )

    elif model_select == 'GraphMVP':

        from GraphMVP.implement import tune
        predictions = tune.predict(
            smis=smiles,
            model_path=model_path,
            model_select='graphmvp',

            batch_size=batch_size,
            verbose=verbose,
        )

    elif model_select == 'KANO':
        from KANO.implement import tune
        predictions = tune.predict(
            smis=smiles,
            model_path=model_path,
            model_select='kano',

            batch_size=batch_size,
            verbose=verbose,
        )

    elif model_select == 'MolCLR-GIN':
        from MolCLR.imlplement import tune
        predictions = tune.predict(
            smis=smiles,
            model_path=model_path,
            model_select='gin',

            batch_size=batch_size,
            verbose=verbose,
        )

    elif model_select == 'MolCLR-GCN':
        from MolCLR.imlplement import tune
        predictions = tune.predict(
            smis=smiles,
            model_path=model_path,
            model_select='gcn',

            batch_size=batch_size,
            verbose=verbose,
        )

    else:
        raise ValueError('Invalid model_select for models.')

    return predictions
