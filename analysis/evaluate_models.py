import torch
torch.set_float32_matmul_precision('high')
import os
import yaml
from models import all_models
from experiment import experiment
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset import sepsisDataModule
import pandas as pd
os.chdir("../")


def init_model(config, task):
    # Define data, model, experiment
    tb_logger = TensorBoardLogger(save_dir=os.path.join(config['logging_params']['save_dir'], task),
                                  name=config['model_params']['name'],
                                  )
    dm = sepsisDataModule(**config["data_params"], pin_memory=torch.cuda.is_available())
    pytorch_model = all_models[config['model_params']['name']](
        size_static=dm.size_static,
        size_timeseries=dm.size_timeseries,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **config['model_params']
    )
    LitModel = experiment(pytorch_model, config['exp_params'], config)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        monitor="val_auprc",
        mode='max',
        save_last=True,
    )
    trainer = L.Trainer(logger=tb_logger,
                        callbacks=[
                            RichProgressBar(),
                            LearningRateMonitor(),
                            checkpoint_callback,
                            EarlyStopping(monitor="val_auprc",
                                          mode="max",
                                          patience=2),
                        ],
                        **config['trainer_params'])

    return trainer, LitModel, dm, checkpoint_callback


config_file = "configs/muse.yaml"
task = "AMR"
versions = [
    8,
    9,
    10,
]


with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
trainer, LitModel, dm, _ = init_model(config, task=task)
dm.setup()
model_dir = os.path.join(config['logging_params']['save_dir'], task, config['model_params']['name'])

checkpoints = []
for version in versions:
    cp_path = os.path.join(model_dir, "version_{}".format(version), "checkpoints")
    for file in os.listdir(cp_path):
        if file.startswith("epoch="):
            file_path = os.path.join(cp_path, file)
            break
    checkpoints.append(file_path)


result = pd.DataFrame()
for cp in checkpoints:
    LitModel.load_state_dict(torch.load(cp)["state_dict"])
    result_row = trainer.test(LitModel, dataloaders=dm.test_dataloader())
    result = result._append(result_row)

print(result.describe().loc[["mean", "std"], :])

