import os, sys
import shutil
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import numpy as np
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.utils.data import Dataset
from KANO_codes.chemprop.data import MoleculeDataset, MoleculeDatapoint
from KANO_codes.chemprop.models import MoleculeModel, add_functional_prompt
from KANO_codes.chemprop.nn_utils import initialize_weights, NoamLR
from KANO_codes.chemprop.features import get_available_features_generators
import warnings
warnings.filterwarnings('ignore')


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

def set_model(model_select, args, encoder_name='CMPNN', task='regression',  multiclass: bool=False, step='functional_prompt', pretrain: bool=False):

    path = os.path.dirname(os.path.dirname(__file__))
    checkpoint_path = os.path.join(path, 'KANO_codes', 'dumped', 'pretrained_graph_encoder', 'original_CMPN_0623_1350_14000th_epoch.pkl')

    if model_select == 'kano':
        model = MoleculeModel(classification=task=='classification', multiclass=multiclass, pretrain=pretrain)
        model.create_encoder(args, encoder_name=encoder_name)
        model.create_ffn(args)
        initialize_weights(model)

        if step == 'functional_prompt':
            add_functional_prompt(model, args)

        model.encoder.load_state_dict(torch.load(checkpoint_path, map_location='cpu'), strict=False)
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
        labels_list = df_dataset['labels']
        graph_list = []
        for idx, smiles in enumerate(tqdm(smiles_list, file=sys.stdout, ncols=100)):
            graph = [smiles, labels_list[idx]]
            graph_list.append(graph)
        if save:
            torch.save(graph_list, save_path)

    return graph_list

def build_optimizer(model, args):
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    increase = ['prompt_generator']
    no_increase_param = [param for name, param in model.named_parameters() if not any(inc in name for inc in increase)]
    increase_param = [param for name, param in model.named_parameters() if any(inc in name for inc in increase)]
    params = [{'params': no_increase_param, 'lr': args.init_lr, 'weight_decay': args.weight_decay}, {'params': increase_param, 'lr': args.init_lr*5, 'weight_decay': args.weight_decay}]
    # params = [{'params': model.parameters(),'lr': args.init_lr, 'weight_decay': 0}]
    return torch.optim.Adam(params)


def build_lr_scheduler(optimizer, args, total_epochs=None):
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """

    total_epochs=total_epochs or [args.epochs] * args.num_lrs
    return NoamLR(
    optimizer=optimizer,
    warmup_epochs=[args.warmup_epochs] *2,
    total_epochs=total_epochs *2,
    steps_per_epoch=args.train_data_size // args.batch_size,
    init_lr=[args.init_lr, args.init_lr*5],
    max_lr=[args.max_lr, args.max_lr*5],
    final_lr=[args.final_lr, args.final_lr*5]
    )


def train_one_epoch(model, prompt, loss_fn, train_data, valid_data, optimizer, epoch, batch_size, device):

    total_train_loss = 0.0
    total_valid_loss = 0.0
    step = 'finetune'

    model.train()
    n_iter_train = len(train_data) // batch_size + 1 if len(train_data) % batch_size != 0 else len(train_data) // batch_size
    n_iter = n_iter_train * batch_size
    for idx in trange(0, n_iter, batch_size, desc='Training'):
        # if idx * batch_size > len(train_data):
        #     break
        mol_batch = MoleculeDataset(train_data[idx : idx + batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        class_weights = torch.ones(targets.shape)
        mask, targets, class_weights = mask.to(device), targets.to(device), class_weights.to(device)
        model.zero_grad()
        preds = model(step, prompt, batch, features_batch)
        loss = loss_fn(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        total_train_loss += loss.item()
        # iter_count += len(mol_batch)

        loss.backward()
        optimizer.step()

    model.eval()
    n_iter_valid = len(valid_data) // batch_size + 1 if len(valid_data) % batch_size != 0 else len(valid_data) // batch_size
    n_iter = n_iter_valid * batch_size
    n_iter = n_iter_valid * batch_size
    for idx in trange(0, n_iter, batch_size, desc='Validation'):
        mol_batch = MoleculeDataset(valid_data[idx: idx + batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        class_weights = torch.ones(targets.shape)
        mask, targets, class_weights = mask.to(device), targets.to(device), class_weights.to(device)
        preds = model(step, prompt, batch, features_batch)
        loss = loss_fn(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        total_valid_loss += loss.item()

    total_train_loss = total_train_loss / n_iter_train
    total_valid_loss = total_valid_loss / n_iter_valid
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_valid_loss:.4f}")
    return total_train_loss, total_valid_loss

def predict_one_epoch(model, prompt, loss_fn, valid_data, epoch, batch_size, device, verbose=True):
    model.eval()
    step = 'finetune'

    total_valid_loss = 0.0
    n_iter_valid = len(valid_data) // batch_size + 1 if len(valid_data) % batch_size != 0 else len(valid_data) // batch_size
    n_iter = n_iter_valid * batch_size
    for idx in trange(0, n_iter, batch_size, desc='Validation'):
        mol_batch = MoleculeDataset(valid_data[idx: idx + batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        class_weights = torch.ones(targets.shape)
        mask, targets, class_weights = mask.to(device), targets.to(device), class_weights.to(device)
        preds = model(step, prompt, batch, features_batch)
        loss = loss_fn(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        total_valid_loss += loss.item()

    print(f"Epoch {epoch}|Test Loss: {loss:.4f}")
    return total_valid_loss / n_iter_valid

class Args:
    def __init__(self, features_generator, hidden_size, init_lr, depth, dropout, activation,
                 undirected, max_lr, final_lr, warmup_epochs, dataset_type, step,
                 ffn_hidden_size=None, ffn_num_layers=2, atom_messages=False,
                 bias: bool=False, features_only:bool=False, output_size: int=1, batch_size=64, weight_decay=1e-4,
                 ):
        self.features_generator = features_generator
        self.hidden_size = hidden_size
        self.init_lr = init_lr
        self.depth = depth
        self.dropout = dropout
        self.activation = activation
        self.undirected = undirected
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.dataset_type = dataset_type

        if ffn_hidden_size is None:
            self.ffn_hidden_size = hidden_size
        else:
            self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers

        self.atom_messages = atom_messages
        self.bias = bias
        self.features_only = features_only

        self.use_input_features = features_generator
        self.step = step

        self.output_size = output_size
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False


def train(
        trainset,
        validset,
        testset = None,
        save=False,
        save_path='experiments/processed_data',
        model_save_path: str = 'experiments/kano',

        model_select: str = 'kano',
        encoder_name: str='CMPNN',  # CMPNN MPNN
        multiclass: bool=False,
        loss_select: str = 'l2',
        task: str = 'regression',
        prompt: bool = False,
        step: str = "functional_prompt", # choices=['pretrain', 'functional_prompt', 'finetune_add', 'finetune_concat'],

        init_lr: float = 0.0001,
        max_lr: float = 0.001,
        final_lr: float = 0.0001,
        warmup_epochs: int = 2,

        hidden_size: int=300,
        ffn_hidden_size: int=300,
        ffn_num_layers: int=3,
        depth: int=3,
        dropout: float=0.0,
        activation: str='ReLU', #choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU']
        undirected: bool = False,
        atom_messages=False, # Use messages on atoms instead of messages on bonds
        bias: bool = False,
        features_only: bool = False,
        output_size: int=1,

        epochs: int = 501,
        batch_size: int = 128,
        num_workers: int = 0,

        verbose: bool = True,
        patience: int = 20,
        weight_decay=1e-4,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    device = set_device()
    set_seed(seed=42)

    args = Args(
        features_generator=None, #get_available_features_generators(),
        hidden_size=hidden_size,
        init_lr=init_lr,
        depth=depth,
        dropout=dropout,
        activation=activation,
        undirected=undirected,
        max_lr=max_lr,
        final_lr=final_lr,
        warmup_epochs=warmup_epochs,
        dataset_type=task, # ['classification', 'regression', 'multiclass']
        ffn_hidden_size=ffn_hidden_size,
        ffn_num_layers=ffn_num_layers,
        atom_messages=atom_messages,
        bias=bias,
        features_only=features_only,
        output_size=output_size,
        batch_size=batch_size,
        step=step,
        weight_decay=weight_decay,
    )

    train_graphs = create_data(trainset, save=save, save_path=save_path+'/train.pth' if isinstance(save_path, str) else save_path / 'train.pth')
    valid_graphs = create_data(validset, save=save, save_path=save_path+'/valid.pth' if isinstance(save_path, str) else save_path / 'valid.pth')
    train_dataset = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
            args=args,
        ) for line in train_graphs
    ])
    valid_dataset = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
            args=args,
        ) for line in valid_graphs
    ])
    if testset is not None:
        test_graphs = create_data(testset, save=save, save_path=save_path+'/test.pth' if isinstance(save_path, str) else save_path / 'test.pth')
        test_dataset = MoleculeDataset([
            MoleculeDatapoint(
                line=line,
                args=args,
            ) for line in test_graphs
        ])

    args.features_size = train_dataset.features_size()
    if train_dataset.data[0].features is not None:
        args.features_dim = len(train_dataset.data[0].features)
    args.train_data_size = len(train_dataset)

    model = set_model(model_select,  args, encoder_name, task,  multiclass, step)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = model.to(device)

    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args, total_epochs=[epochs])
    loss_fn = set_loss_fn(loss_select)

    best_epoch, best_loss = 0, 1e9
    train_loss_list, valid_loss_list, test_loss_list = [], [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss, valid_loss = train_one_epoch(model, prompt, loss_fn, train_dataset, valid_dataset, optimizer, epoch, batch_size, device)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if testset is not None:
            test_loss = predict_one_epoch(model, prompt, loss_fn, test_dataset, epoch, batch_size, device)
            test_loss_list.append(test_loss)

        scheduler.step()
        print(f'Epoch {epoch} / {epochs} : Learning rate :', scheduler.get_lr())

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
        model_path: str = 'experiments/kano',

        model_select: str = 'kano',
        encoder_name: str='CMPNN',  # CMPNN MPNN
        multiclass: bool=False,
        task: str = 'regression',
        prompt: bool = False,
        step: str = "functional_prompt",  # choices=['pretrain', 'functional_prompt', 'finetune_add', 'finetune_concat'],

        init_lr: float = 0.0001,
        max_lr: float = 0.001,
        final_lr: float = 0.0001,
        warmup_epochs: int = 2,

        hidden_size: int=300,
        ffn_hidden_size: int=300,
        ffn_num_layers: int=3,
        depth: int=3,
        dropout: float=0.0,
        activation: str='ReLU', #choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU', 'GELU']
        undirected: bool = False,
        atom_messages=False, # Use messages on atoms instead of messages on bonds
        bias: bool = False,
        features_only: bool = False,
        output_size: int=1,

        epochs: int = 501,
        batch_size: int = 128,
        num_workers: int = 0,

        verbose: bool = True,
        patience: int = 20,
):
    device = set_device()
    set_seed(seed=42)

    args = Args(
        features_generator=None, #get_available_features_generators(),
        hidden_size=hidden_size,
        init_lr=init_lr,
        depth=depth,
        dropout=dropout,
        activation=activation,
        undirected=undirected,
        max_lr=max_lr,
        final_lr=final_lr,
        warmup_epochs=warmup_epochs,
        dataset_type=task, # ['classification', 'regression', 'multiclass']
        ffn_hidden_size=ffn_hidden_size,
        ffn_num_layers=ffn_num_layers,
        atom_messages=atom_messages,
        bias=bias,
        features_only=features_only,
        output_size=output_size,
        batch_size=batch_size,
        step=step
    )
    smis = parse_data(smis, values=list(range(len(smis))))

    graphs = create_data(smis, save=False)
    pred_dataset = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
            args=args,
        ) for line in graphs
    ])

    args.features_size = pred_dataset.features_size()
    if pred_dataset.data[0].features is not None:
        args.features_dim = len(pred_dataset.data[0].features)

    model = set_model(model_select,  args, encoder_name, task,  multiclass)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    model = model.to(device)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    step = 'finetune'

    predictions = []
    with torch.no_grad():
        n_iter_valid = len(pred_dataset) // batch_size + 1
        n_iter = n_iter_valid * batch_size
        for idx in trange(0, n_iter, batch_size, desc='Validation'):
            mol_batch = MoleculeDataset(pred_dataset[idx: idx + batch_size])
            smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
            batch = smiles_batch
            preds = model(step, prompt, batch, features_batch).squeeze()
            predictions.append(preds)

    predictions = [tensor if tensor.dim() > 0 else tensor.unsqueeze(0) for tensor in predictions]
    predictions = torch.cat(predictions).cpu().numpy()
    return predictions
