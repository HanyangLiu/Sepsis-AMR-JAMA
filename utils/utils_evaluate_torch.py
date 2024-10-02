import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import all_models
from experiment import experiment
import torch

torch.set_float32_matmul_precision('high')
import torch.backends.cudnn as cudnn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import seed
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset import sepsisDataModule
from tqdm import tqdm
import shap
import pandas as pd
# import matplotlib.pyplot as plt
from utils.utils_data import select_subgroup, instance_filter, select_hospital
from sklearn import preprocessing, metrics
from utils.utils_evaluation import evaluate_multi_NN, plot_prc, plot_roc, evaluate


def build_lit_model(checkpoints):
    config = checkpoints[0]["hyper_parameters"]
    # torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define models
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'],
                                  )

    dm = sepsisDataModule(**config["data_params"], pin_memory=torch.cuda.is_available())
    dm.setup()

    pytorch_model = all_models[config['model_params']['name']](
        size_static=dm.size_static,
        size_timeseries=dm.size_timeseries,
        device=device,
        **config['model_params']
    )
    LitModel = experiment(pytorch_model, config['exp_params'], config)
    trainer = L.Trainer(logger=tb_logger,
                        callbacks=[
                            RichProgressBar(),
                            LearningRateMonitor(),
                            ModelCheckpoint(save_top_k=1,
                                            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                            monitor="val_auprc",
                                            mode='max',
                                            save_last=True,
                                            ),
                            EarlyStopping(monitor="val_auprc",
                                          mode="max",
                                          patience=1),
                        ],
                        **config['trainer_params'])

    return LitModel, trainer, dm


def get_plot_points(batch_outputs):
    probs = torch.cat([ele['probs'] for ele in batch_outputs], dim=0).cpu().numpy()
    labels = torch.cat([ele['labels'] for ele in batch_outputs], dim=0).cpu().numpy()
    return probs, labels


def test_subgroup(LitModel, trainer, dm, checkpoints, test_data, subgroup='16', binary=False):
    # plot results
    for infection_instance in ['Community-acquired', 'Hospital-acquired']:
        # select infection type
        if infection_instance == 'Community-acquired':
            data = test_data[test_data.infection_id == 0]
        elif infection_instance == 'Hospital-acquired':
            data = test_data[test_data.infection_id > 0]
        else:
            data = test_data
        sub_indices = select_subgroup(data, group=subgroup)

        if binary:
            prefix = infection_instance
            N = 1000
            AUROC, AUPRC = [], []
            TPRs, PRECs = np.empty(shape=(2, len(checkpoints), N)), np.empty(shape=(2, len(checkpoints), N))
            for model_id, cp in enumerate(checkpoints):
                LitModel.load_state_dict(cp["state_dict"])
                dataloader = dm.test_dataloader(idx_test=sub_indices)
                outputs = trainer.predict(LitModel, dataloaders=dataloader)
                probs, labels = get_plot_points(outputs)
                probs[:, 1] = probs[:, 1] + probs[:, 2]
                probs = probs[:, :2]
                labels = (labels > 0).astype(int)
                auroc, auprc, TPR, PREC = evaluate_multi_NN(probs, labels)
                AUROC.append(list(auroc.values()))
                AUPRC.append(list(auprc.values()))
                TPRs[:, model_id, :] = np.array([TPR[i] for i in TPR.keys()])
                PRECs[:, model_id, :] = np.array([PREC[i] for i in PREC.keys()])

            tpr_mean, tpr_std = np.mean(TPRs, axis=1), np.std(TPRs, axis=1)
            prec_mean, prec_std = np.mean(PRECs, axis=1), np.std(PRECs, axis=1)

            print('--------------------------------------------')
            print('Evaluation of test set:')
            print("AU-ROC: {} ({})".format(np.mean(np.array(AUROC), axis=0), np.std(np.array(AUROC), axis=0)),
                  "AU-PRC: {} ({})".format(np.mean(np.array(AUPRC), axis=0), np.std(np.array(AUPRC), axis=0)))
            print('--------------------------------------------')

            x = np.linspace(0, 1, N + 1)[:N]
            label_names = ['SS', 'RS & RR']
            plot_roc(x, tpr_mean, tpr_std,
                     auc=[np.mean(np.array(AUROC), axis=0), np.std(np.array(AUROC), axis=0)],
                     multiclass=True,
                     labels=label_names,
                     prefix=prefix)
            plot_prc(x, prec_mean, prec_std,
                     auc=[np.mean(np.array(AUPRC), axis=0), np.std(np.array(AUPRC), axis=0)],
                     multiclass=True,
                     labels=label_names,
                     prefix=prefix)

            # FP/FN analysis
            idx = np.argmax(tpr_mean[1, :] - x)  # Youden index
            FPR = x[idx]
            FNR = 1 - tpr_mean[1, :][idx]
            print("False positive rate:", FPR)
            print("False negative rate:", FNR)


        else:
            prefix = infection_instance
            N = 1000
            AUROC, AUPRC = [], []
            TPRs, PRECs = np.empty(shape=(3, len(checkpoints), N)), np.empty(shape=(3, len(checkpoints), N))
            for model_id, cp in enumerate(checkpoints):
                LitModel.load_state_dict(cp["state_dict"])
                dataloader = dm.test_dataloader(idx_test=sub_indices)

                outputs = trainer.predict(LitModel, dataloaders=dataloader)
                probs, labels = get_plot_points(outputs)
                auroc, auprc, TPR, PREC = evaluate_multi_NN(probs, labels)
                AUROC.append(list(auroc.values()))
                AUPRC.append(list(auprc.values()))
                TPRs[:, model_id, :] = np.array([TPR[i] for i in TPR.keys()])
                PRECs[:, model_id, :] = np.array([PREC[i] for i in PREC.keys()])

            tpr_mean, tpr_std = np.mean(TPRs, axis=1), np.std(TPRs, axis=1)
            prec_mean, prec_std = np.mean(PRECs, axis=1), np.std(PRECs, axis=1)

            x = np.linspace(0, 1, N + 1)[:N]
            label_names = ['SS', 'RS', 'RR']
            plot_roc(x, tpr_mean, tpr_std,
                     auc=[np.mean(np.array(AUROC), axis=0), np.std(np.array(AUROC), axis=0)],
                     multiclass=True,
                     labels=label_names,
                     prefix=prefix)
            plot_prc(x, prec_mean, prec_std,
                     auc=[np.mean(np.array(AUPRC), axis=0), np.std(np.array(AUPRC), axis=0)],
                     multiclass=True,
                     labels=label_names,
                     prefix=prefix)


def test_all_subgroups(LitModel, trainer, dm, checkpoints, test_data, postfix=None):
    # sub-cohort results
    rows = []

    for model_id, cp in enumerate(checkpoints):
        # load model state and test
        LitModel.load_state_dict(cp["state_dict"])

        for g in range(19 + 1):
            # if g != 0: continue
            print('Testing subgroup {}...'.format(g))

            for infection_instance in ['all', 'initial', 'subsequent']:
                prefix = 'Subgroup_{}-{}_instances'.format(g, infection_instance)

                # select infection type
                if infection_instance == 'initial':
                    data = test_data[test_data.infection_id == 0]
                elif infection_instance == 'subsequent':
                    data = test_data[test_data.infection_id > 0]
                else:
                    data = test_data

                # select sub-cohort
                sub_indices = select_subgroup(data, group=str(g))
                if sub_indices is None or len(sub_indices) == 0:
                    continue

                dataloader = dm.test_dataloader(sub_indices)

                sub_data = data.loc[sub_indices.index]
                sub_data['label'] = np.logical_or(sub_data['RS'], sub_data['RR'])
                sub_data['label'] = sub_data['label'].astype(int)

                res = trainer.test(LitModel, dataloaders=dataloader, verbose=False)

                if len(res):
                    auroc, auprc = res[0]['test_auroc'], res[0]['test_auprc']
                else:
                    auroc, auprc = None, None

                print(prefix)

                rows.append({
                    "subgroup": g,
                    "n_instances": len(sub_data),
                    "fraction_in_set": len(sub_data) / len(test_data),
                    "positive_rate": sum(sub_data['label']) / len(sub_data),
                    "infection_instance": infection_instance,
                    "hospital_id": 'all',
                    "model_id": model_id,
                    "auroc": auroc,
                    "auprc": auprc,
                })

    df_results = pd.DataFrame.from_dict(rows, orient='columns')
    df_results.to_csv("../data_analysis/subgroup_performance_{}.csv".format(postfix), index=False)

    return df_results

def test_all_hospitals(LitModel, trainer, dm, checkpoints, test_data, postfix=None):
    # sub-cohort results
    rows = []

    for model_id, cp in enumerate(checkpoints):
        # load model state and test
        LitModel.load_state_dict(cp["state_dict"])

        for g in range(10):
            # if g != 0: continue
            print('Testing hospital {}...'.format(g))

            for infection_instance in ['all', 'initial', 'subsequent']:
                prefix = 'Subgroup_{}-{}_instances'.format(g, infection_instance)

                # select infection type
                if infection_instance == 'initial':
                    data = test_data[test_data.infection_id == 0]
                elif infection_instance == 'subsequent':
                    data = test_data[test_data.infection_id > 0]
                else:
                    data = test_data

                # select sub-cohort
                sub_indices = select_hospital(data, group=str(g))
                if sub_indices is None or len(sub_indices) == 0:
                    continue

                dataloader = dm.test_dataloader(sub_indices)

                sub_data = data.loc[sub_indices.index]
                sub_data['label'] = np.logical_or(sub_data['RS'], sub_data['RR'])
                sub_data['label'] = sub_data['label'].astype(int)

                res = trainer.test(LitModel, dataloaders=dataloader, verbose=False)

                if len(res):
                    auroc, auprc = res[0]['test_auroc'], res[0]['test_auprc']
                else:
                    auroc, auprc = None, None

                print(prefix)

                rows.append({
                    "subgroup": g,
                    "n_instances": len(sub_data),
                    "fraction_in_set": len(sub_data) / len(test_data),
                    "positive_rate": sum(sub_data['label']) / len(sub_data),
                    "infection_instance": infection_instance,
                    "hospital_id": 'all',
                    "model_id": model_id,
                    "auroc": auroc,
                    "auprc": auprc,
                })

    df_results = pd.DataFrame.from_dict(rows, orient='columns')
    df_results.to_csv("../data_analysis/hospital_performance_{}.csv".format(postfix), index=False)

    return df_results

