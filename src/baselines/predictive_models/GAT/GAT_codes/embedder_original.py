import numpy as np
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random

random.seed(0)

import os

from argument import config2string
from data import Dataclass
from torch_geometric.loader import DataLoader
from utils import create_batch_mask


class embedder:

    def __init__(self, args, train_df, valid_df, test_df, repeat, time, fold):
        self.args = args
        self.config_str = "experiment{}_fold{}_".format(repeat + 1, fold + 1) + config2string(args)
        print("\n[Config] {}\n".format(self.config_str))

        # get time
        self.time = time
        print(self.time)
        self.results_path = f"results/{args.embedder}/{self.args.model_type}/{self.args.dataset}/{self.time}"
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        self.CHECKPOINT_PATH = f"model_checkpoints/{args.embedder}/{args.model_type}/{self.args.dataset}/{self.time}"
        if not os.path.exists(self.CHECKPOINT_PATH):
            os.makedirs(self.CHECKPOINT_PATH)
        self.check_dir = self.CHECKPOINT_PATH + self.config_str + ".pt"

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        self.train_dataset = Dataclass(train_df)
        self.val_dataset = Dataclass(valid_df)

        if test_df is not None:
            self.test_dataset = Dataclass(test_df)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size)
        else:
            self.test_loader = None

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size)

        self.is_early_stop = False

        self.best_val_loss = 100000000000.0
        self.best_val_losses = []

        self.fold = fold
        self.repeat = repeat

    def predict(self, model):
        train_pred = []
        train_true = []

        val_pred = []
        val_true = []

        test_pred = []
        test_true = []

        with torch.no_grad():
            for bc, samples in enumerate(self.train_loader):
                output = model([
                    samples[0].to(self.device)
                ])
                train_pred.extend(output.cpu().detach().numpy().flatten().tolist())
                train_true.extend(samples[1].reshape(-1, 1).to(self.device).cpu().detach().numpy().flatten().tolist())

            for bc, samples in enumerate(self.val_loader):
                output = model([
                    samples[0].to(self.device)
                ])
                val_pred.extend(output.cpu().detach().numpy().flatten().tolist())
                val_true.extend(samples[1].reshape(-1, 1).to(self.device).cpu().detach().numpy().flatten().tolist())

            if self.test_loader is not None:
                for bc, samples in enumerate(self.test_loader):
                    output = model([
                        samples[0].to(self.device)
                    ])
                    test_pred.extend(output.cpu().detach().numpy().flatten().tolist())
                    test_true.extend(
                        samples[1].reshape(-1, 1).to(self.device).cpu().detach().numpy().flatten().tolist())
            else:
                test_pred = [0, 0, 0]
                test_true = [0, 0, 0]
            # print(val_true)
            # print(val_pred)
        return train_pred, train_true, val_pred, val_true, test_pred, test_true

    def evaluate(self, epoch):
        train_losses = []
        train_mae_losses = []

        valid_losses = []
        valid_mae_losses = []

        test_losses = []
        test_mae_losses = []

        with torch.no_grad():
            for bc, samples in enumerate(self.train_loader):
                output = self.model([
                    samples[0].to(self.device)
                ])

                train_loss = ((output - samples[1].reshape(-1, 1).to(self.device)) ** 2).reshape(-1)
                train_mae_loss = torch.abs((output - samples[1].reshape(-1, 1).to(self.device))).reshape(-1)
                train_losses.append(train_loss.cpu().detach().numpy())
                train_mae_losses.append(train_mae_loss.cpu().detach().numpy())

            for bc, samples in enumerate(self.val_loader):
                output = self.model([
                    samples[0].to(self.device)
                ])
                # print(output)
                # print(samples[5].reshape(-1, 1))
                val_loss = ((output - samples[1].reshape(-1, 1).to(self.device)) ** 2).reshape(-1)
                val_mae_loss = torch.abs((output - samples[1].reshape(-1, 1).to(self.device))).reshape(-1)
                valid_losses.append(val_loss.cpu().detach().numpy())
                valid_mae_losses.append(val_mae_loss.cpu().detach().numpy())

            if self.test_loader is not None:
                for bc, samples in enumerate(self.test_loader):
                    output = self.model([
                        samples[0].to(self.device)
                    ])

                    test_loss = ((output - samples[1].reshape(-1, 1).to(self.device)) ** 2).reshape(-1)
                    test_mae_loss = torch.abs((output - samples[1].reshape(-1, 1).to(self.device))).reshape(-1)
                    test_losses.append(test_loss.cpu().detach().numpy())
                    test_mae_losses.append(test_mae_loss.cpu().detach().numpy())
                self.test_loss = np.mean(np.hstack(test_losses))
                self.test_rmse_loss = np.sqrt(self.test_loss)
                self.test_mae_loss = np.mean(np.hstack(test_mae_losses))
            else:
                self.test_loss = 0
                self.test_rmse_loss = 0
                self.test_mae_loss = 0

        self.train_loss = np.mean(np.hstack(train_losses))
        self.train_rmse_loss = np.sqrt(self.train_loss)
        self.train_mae_loss = np.mean(np.hstack(train_mae_losses))
        self.val_loss = np.mean(np.hstack(valid_losses))
        self.val_rmse_loss = np.sqrt(self.val_loss)
        self.val_mae_loss = np.mean(np.hstack(valid_mae_losses))
        # print("#########")
        # print(self.val_mae_loss)

        if self.val_loss < self.best_val_loss:
            self.best_train_loss = self.train_loss
            self.best_val_loss = self.val_loss
            self.best_test_loss = self.test_loss
            self.best_train_mae_loss = self.train_mae_loss
            self.best_val_mae_loss = self.val_mae_loss
            self.best_test_mae_loss = self.test_mae_loss
            self.best_train_rmse_loss = self.train_rmse_loss
            self.best_val_rmse_loss = self.val_rmse_loss
            self.best_test_rmse_loss = self.test_rmse_loss
            self.best_epoch = epoch
            torch.save(self.model, f'{self.CHECKPOINT_PATH}/best_{self.property}_model_{self.repeat}_{self.fold}.pt')

        self.best_val_losses.append(self.best_val_loss)

        self.eval_config = "[Epoch: {}] train MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f} || valid MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f} || test MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f}".format(
            epoch, self.train_loss, self.train_rmse_loss, self.train_mae_loss, self.val_loss, self.val_rmse_loss,
            self.val_mae_loss, self.test_loss, self.test_rmse_loss,
            self.test_mae_loss)
        self.best_config = "[Best Epoch: {}] Best train MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f} || Best valid MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f} || Best test MSE / RMSE / MAE --> {:.4f} / {:.4f} / {:.4f}".format(
            self.best_epoch, self.best_train_loss, self.best_train_rmse_loss, self.best_train_mae_loss,
            self.best_val_loss, self.best_val_rmse_loss, self.best_val_mae_loss, self.best_test_loss,
            self.best_test_rmse_loss, self.best_test_mae_loss)

        print(self.eval_config)
        print(self.best_config)
