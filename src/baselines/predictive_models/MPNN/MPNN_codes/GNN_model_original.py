import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F

from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from embedder_original import embedder
from layers import gat_layer, gcn_layer, gin_layer, graphsage_layer, mpnn_layer

from torch_scatter import scatter_mean, scatter_add, scatter_std
import pandas as pd

from sklearn.metrics import r2_score


class GNN_model_Trainer(embedder):
    def __init__(self, args, train_df, valid_df, test_df, repeat, time, fold):
        embedder.__init__(self, args, train_df, valid_df, test_df, repeat, time, fold)

        self.model = GNN_model(
            device=self.device,
            num_layers=self.args.num_layers,
            dropout_ratio=self.args.dropout,
            model_type=self.args.model_type
        )
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=self.args.patience,
            mode='min',
            verbose=True,
            factor=0.5
        )
        self.warm_step = self.args.warm_step
        self.epoch_scheduler = StepLR(
            self.optimizer,
            step_size=self.warm_step,
            verbose=True,
            gamma=0.1
        )
        self.fold = fold
        self.property = args.dataset

        print(self.model)
        self.print_model_parameters(self.model)



    def print_model_parameters(self, model):
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel()} parameters")
                total_params += param.numel()
                print(f"For now parameters: {total_params}")
        print(f"Total parameters: {total_params}")

    def train(self):

        loss_fn = torch.nn.MSELoss()

        epoch_list = []
        train_rmse_list = []
        train_mse_list = []
        train_mae_list = []
        train_r2_list = []
        valid_rmse_list = []
        valid_mse_list = []
        valid_mae_list = []
        valid_r2_list = []
        test_rmse_list = []
        test_mse_list = []
        test_mae_list = []
        test_r2_list = []

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model([
                    samples[0].to(self.device),
                ])

                loss = loss_fn(outputs, samples[1].reshape(-1, 1).to(self.device).float())

                loss.backward()
                self.optimizer.step()
                self.train_loss += loss

            self.model.eval()
            self.evaluate(epoch)

            train_pred, train_true, val_pred, val_true, test_pred, test_true = self.predict(model=self.model)
            train_r2_list.append(r2_score(train_true, train_pred))
            valid_r2_list.append(r2_score(val_true, val_pred))
            test_r2_list.append(r2_score(test_true, test_pred))

            epoch_list.append(epoch)
            train_rmse_list.append(self.train_rmse_loss)
            train_mse_list.append(self.train_loss)
            train_mae_list.append(self.train_mae_loss)
            valid_rmse_list.append(self.val_rmse_loss)
            valid_mse_list.append(self.val_loss)
            valid_mae_list.append(self.val_mae_loss)
            test_rmse_list.append(self.test_rmse_loss)
            test_mse_list.append(self.test_loss)
            test_mae_list.append(self.test_mae_loss)

            if epoch < self.warm_step + 1:
                self.epoch_scheduler.step()
            else:
                self.scheduler.step(self.val_loss)

            # Early stopping
            if len(self.best_val_losses) > int(self.args.es / self.args.eval_freq):
                if self.best_val_losses[-1] == self.best_val_losses[-int(self.args.es / self.args.eval_freq)]:
                    self.is_early_stop = True
                    torch.save(self.model,
                               f'{self.CHECKPOINT_PATH}/final_{self.property}_model_{self.repeat}_{self.fold}.pt')
                    break

            torch.save(self.model,
                       f'{self.CHECKPOINT_PATH}/final_{self.property}_model_{self.repeat}_{self.fold}.pt')

        self.evaluate(epoch)

        best_model = torch.load(f'{self.CHECKPOINT_PATH}/best_{self.property}_model_{self.repeat}_{self.fold}.pt')
        best_model.to(self.device)
        best_model.eval()
        train_pred, train_true, val_pred, val_true, test_pred, test_true = self.predict(model=best_model)
        self.best_train_r2 = r2_score(train_true, train_pred)
        self.best_val_r2 = r2_score(val_true, val_pred)
        self.best_test_r2 = r2_score(test_true, test_pred)

        result = {
            "epoch": epoch_list,
            "train_r2": train_r2_list,
            "valid_r2": valid_r2_list,
            "test_r2": test_r2_list,
            "train_rmse": train_rmse_list,
            "valid_rmse": valid_rmse_list,
            "test_rmse": test_rmse_list,
            "train_mse": train_mse_list,
            "valid_mse": valid_mse_list,
            "test_mse": test_mse_list,
            "train_mae": train_mae_list,
            "valid_mae": valid_mae_list,
            "test_mae": test_mae_list
        }
        train_data_result = {
            'train_true': train_true,
            'train_pred': train_pred,
        }

        valid_data_result = {
            'valid_true': val_true,
            'valid_pred': val_pred,
        }

        test_data_result = {
            'test_true': test_true,
            'test_pred': test_pred
        }

        pd.DataFrame(result).to_csv(
            f"results/{self.args.embedder}/{self.args.model_type}/{self.property}/{self.time}/{self.property}_result_{self.repeat}_{self.fold}.csv",
            index=False)
        pd.DataFrame(train_data_result).to_csv(
            f"results/{self.args.embedder}/{self.args.model_type}/{self.property}/{self.time}/{self.property}_train_true_pred_{self.repeat}_{self.fold}.csv",
            index=False)
        pd.DataFrame(valid_data_result).to_csv(
            f"results/{self.args.embedder}/{self.args.model_type}/{self.property}/{self.time}/{self.property}_valid_true_pred_{self.repeat}_{self.fold}.csv",
            index=False)
        pd.DataFrame(test_data_result).to_csv(
            f"results/{self.args.embedder}/{self.args.model_type}/{self.property}/{self.time}/{self.property}_test_true_pred_{self.repeat}_{self.fold}.csv",
            index=False)

        return (self.best_val_loss, self.best_val_mae_loss, self.best_val_rmse_loss, self.best_val_r2,
                self.best_train_loss, self.best_train_mae_loss, self.best_train_rmse_loss, self.best_train_r2)


class GNN_model(nn.Module):
    def __init__(self,
                 device,
                 node_input_dim=46,
                 edge_input_dim=10,
                 node_hidden_dim=256,
                 edge_hidden_dim=256,
                 num_layers=5,
                 num_step_set2_set=1,
                 num_layer_set2set=1,
                 dropout_ratio=0,
                 model_type='MPNN'
                 ):
        super(GNN_model, self).__init__()

        self.device = device
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.model_type = model_type

        if self.model_type == 'MPNN':
            from layers.mpnn_layer import MPNN
            self.mol_gather = MPNN(self.node_input_dim, self.edge_input_dim,
                                     self.node_hidden_dim, self.edge_input_dim,
                                     self.num_layers)

        elif self.model_type == 'GCN':
            from layers.gcn_layer import GCN
            self.mol_gather = GCN(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim*2, self.num_layers)

        elif self.model_type == 'GAT':
            from layers.gat_layer import GAT
            self.mol_gather = GAT(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim*2, self.num_layers)

        elif self.model_type == 'GIN':
            from layers.gin_layer import GIN
            self.mol_gather = GIN(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim*2, self.num_layers)

        elif self.model_type == 'GraphSAGE':
            from layers.graphsage_layer import GraphSAGE
            self.mol_gather = GraphSAGE(self.node_input_dim, self.node_hidden_dim, self.node_hidden_dim*2, self.num_layers)

        self.set2set = Set2Set(node_hidden_dim*2, processing_steps=4, num_layers=2)

        self.predictor = nn.Sequential(
            nn.Linear(self.node_hidden_dim*4, self.node_hidden_dim),  #######8 --  4
            # nn.LeakyReLU(),  ###
            # nn.SELU(),  ###
            # nn.SiLU(),  ###
            # nn.ReLU(), ###
            # nn.Sigmoid(), ###
            # nn.Linear(256, 128),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.node_hidden_dim, self.node_hidden_dim // 2),
            nn.BatchNorm1d(self.node_hidden_dim // 2),
            # nn.SELU(),  ###
            nn.LeakyReLU(),  ###
            # nn.SiLU(), ###
            # nn.ReLU(), ###
            # nn.Sigmoid(), ###
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.node_hidden_dim // 2, self.node_hidden_dim // 4),
            nn.BatchNorm1d(self.node_hidden_dim // 4),
            # nn.SELU(),  ###
            nn.LeakyReLU(),  ###
            # nn.SiLU(), ###
            # nn.ReLU(), ###
            # nn.Sigmoid(), ###
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.node_hidden_dim // 4, 1)
        )
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        mol = data[0]

        # node embeddings for each graph
        mol_features = self.mol_gather(mol)

        # Add normalization
        self.mol_features = F.normalize(mol_features, dim=1)

        # readout (pooling) after normalization
        # self.final_features = global_mean_pool(self.mol_features, mol.batch)
        self.final_features = self.set2set(self.mol_features, mol.batch)


        # get predictions
        predictions = self.predictor(self.final_features)

        return predictions

    def get_checkpoints(self):

        return self.final_features
