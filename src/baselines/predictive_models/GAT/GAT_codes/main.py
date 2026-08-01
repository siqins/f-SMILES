import torch
import random
import numpy as np
import os
import argument
from utils import write_summary, write_summary_cv
from sklearn.model_selection import StratifiedKFold
import pandas as pd

torch.set_num_threads(8)
from datetime import datetime


def seed_everything(seed):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment():
    args, unknown = argument.parse_args()
    for model_name in ['GCN', 'GAT', 'MPNN', 'GIN', 'GraphSAGE']:

        args.model_type = model_name
        time = datetime.now().strftime('%b%d_%H-%M-%S')


        dataset = torch.load("./data/processed/{}.pt".format(args.dataset))
        len_data = len(dataset)

        # random.shuffle(dataset)
        len_test = int(len_data / args.fold)
        len_val = int(len_test * 0.5)
        # len_val = int(len_data / args.fold)
        # len_test = int(len_val * 0.5)
        # print(dataset[0])
        dataset = np.asarray(dataset, dtype='object')
        # print(dataset[-1])
        y_values = np.array([data[-1] for data in dataset])
        group = pd.qcut(y_values, q=5, labels=[1, 2, 3, 4, 5])

        repeat_list = []
        cv_train_rmses = []
        cv_train_maes = []
        cv_train_mses = []
        cv_train_r2s = []
        cv_val_rmses = []
        cv_val_maes = []
        cv_val_mses = []
        cv_val_r2s = []

        seed_list = []
        val_mses = []
        val_maes = []
        val_rmses = []
        val_r2s = []
        train_mses = []
        train_maes = []
        train_rmses = []
        train_r2s = []

        for repeat in range(args.repeat):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            fold = 0
            # Set Random Seed
            # seed_everything(repeat)
            best_val_mses = []
            best_val_maes = []
            best_val_rmses = []
            best_val_r2s = []
            best_train_mses = []
            best_train_maes = []
            best_train_rmses = []
            best_train_r2s = []

            # for fold in range(args.fold):
            for train_index, val_index in skf.split(dataset, group, groups=group):
                fold += 1
                seed_list.append(str(repeat) + f"_{fold}")
                # test_index = np.asarray([i for i in range(fold * len_test, (fold + 1) * len_test)])
                # train_index = np.asarray([i for i in range(len_data) if np.isin(i, test_index, invert = True)])
                # print(train_index)

                train_set = dataset[train_index].tolist()
                val_set = dataset[val_index].tolist()

                # Get validation set
                # val_set = test_set[:len_val]  ###
                # test_set = test_set[len_val:] ###
                # val_set = test_set[:len_val].tolist()
                # test_set = test_set[len_val:].tolist()

                best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2, config_str, best_config = main(
                    args, train_set, val_set, None, repeat, time, fold)  ###

                best_val_mses.append(best_val_mse)
                best_val_maes.append(best_val_mae)
                best_val_rmses.append(best_val_rmse)
                best_val_r2s.append(best_val_r2)
                best_train_mses.append(best_train_mse)
                best_train_maes.append(best_train_mae)
                best_train_rmses.append(best_train_rmse)
                best_train_r2s.append(best_train_r2)

                val_mses.append(best_val_mse)
                val_maes.append(best_val_mae)
                val_rmses.append(best_val_rmse)
                val_r2s.append(best_val_r2)
                train_mses.append(best_train_mse)
                train_maes.append(best_train_mae)
                train_rmses.append(best_train_rmse)
                train_r2s.append(best_train_r2)

            mean_val_mse = np.mean(np.asarray(best_val_mses))
            mean_val_mse_std = np.std(np.asarray(best_val_mses))

            mean_val_mae = np.mean(np.asarray(best_val_maes))
            mean_val_mae_std = np.std(np.asarray(best_val_maes))

            mean_val_rmse = np.mean(np.asarray(best_val_rmses))
            mean_val_rmse_std = np.std(np.asarray(best_val_rmses))

            mean_val_r2 = np.mean(np.asarray(best_val_r2s))
            mean_val_r2_std = np.std(np.asarray(best_val_r2s))

            mean_train_mse = np.mean(np.asarray(best_train_mses))
            mean_train_mse_std = np.std(np.asarray(best_train_mses))

            mean_train_mae = np.mean(np.asarray(best_train_maes))
            mean_train_mae_std = np.std(np.asarray(best_train_maes))

            mean_train_rmse = np.mean(np.asarray(best_train_rmses))
            mean_train_rmse_std = np.std(np.asarray(best_train_rmses))

            mean_train_r2 = np.mean(np.asarray(best_train_r2s))
            mean_train_r2_std = np.std(np.asarray(best_train_r2s))

            repeat_list.append(repeat)
            cv_train_mses.append(mean_train_mse)
            cv_train_maes.append(mean_train_mae)
            cv_train_rmses.append(mean_train_rmse)
            cv_train_r2s.append(mean_train_r2)
            cv_val_mses.append(mean_val_mse)
            cv_val_maes.append(mean_val_mae)
            cv_val_rmses.append(mean_val_rmse)
            cv_val_r2s.append(mean_val_r2)

        final_cv_result = pd.DataFrame({
            "Repeat Number": repeat_list,
            "train_mean_mse": cv_train_mses,
            "val_mean_mse": cv_val_mses,
            "train_mean_rmse": cv_train_rmses,
            "val_mean_rmse": cv_val_rmses,
            "train_mean_mae": cv_train_maes,
            "val_mean_mae": cv_val_maes,
            "train_mean_r2": cv_train_r2s,
            "val_mean_r2": cv_val_r2s
        })
        final_cv_result.to_csv(
            f"results/{args.embedder}/{args.model_type}/{args.dataset}/{args.dataset}_cv_result_{time}.csv", index=False)

        final_result = pd.DataFrame({
            "seed": seed_list,
            "train_mse": train_mses,
            "val_mse": val_mses,
            "train_rmse": train_rmses,
            "val_rmse": val_rmses,
            "train_mae": train_maes,
            "val_mae": val_maes,
            "train_r2": train_r2s,
            "val_r2": val_r2s
        })
        final_result.to_csv(
            f"results/{args.embedder}/{args.model_type}/{args.dataset}/{args.dataset}_final_result_{time}.csv", index=False)


def main(args, train_df, valid_df, test_df, repeat, time, fold=5):
    if args.embedder == "CIGIN":
        from models.CIGIN import CIGIN_Trainer
        embedder = CIGIN_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    if args.embedder == "CGIB":
        from models.CGIB import CGIB_Trainer
        embedder = CGIB_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    elif args.embedder == "GNN_cross_attention":
        from models.GNN_cross_attention import GNN_cross_attention_Trainer
        embedder = GNN_cross_attention_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    elif args.embedder == "GNN":
        from models.GNN_model import GNN_model_Trainer
        embedder = GNN_model_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    elif args.embedder == "GNN_original":
        from models.GNN_model_original import GNN_model_Trainer
        embedder = GNN_model_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    elif args.embedder == "GNN_original_finetune":
        from models.GNN_model_original_finetune import GNN_model_Trainer
        embedder = GNN_model_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    elif args.embedder == "GNN_rxn":
        from models.GNN_model_rxn import GNN_model_Trainer
        embedder = GNN_model_Trainer(args, train_df, valid_df, test_df, repeat, time, fold)
        best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2 = embedder.train()

    return best_val_mse, best_val_mae, best_val_rmse, best_val_r2, best_train_mse, best_train_mae, best_train_rmse, best_train_r2, embedder.config_str, embedder.best_config


if __name__ == "__main__":
    experiment()
