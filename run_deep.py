import os
import yaml
import argparse
from models import all_models
from experiment import experiment
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from dataset import sepsisDataModule
import pandas as pd

NUM_RUNS = 5
# os.environ["CUDA_VISIBLE_DEVICES"]=""
# torch.cuda.is_available = lambda : False
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device, torch.cuda.is_available())

# load configs
parser = argparse.ArgumentParser(description='Generic runner')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/deep_model.yaml')
parser.add_argument('--task', '-t',
                    default='AMR')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Set random seed and precision
config["data_params"]["multiclass"] = config["model_params"]["multiclass"]
config['data_params']['task'] = args.task
print(config)

result = pd.DataFrame()
for seed in range(NUM_RUNS):
    config["exp_params"]["manual_seed"] = seed
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    # Define data, model, experiment
    tb_logger = TensorBoardLogger(save_dir=os.path.join(config['logging_params']['save_dir'], args.task),
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

    # Train/eval
    print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(LitModel, datamodule=dm)
    result_row = trainer.test(ckpt_path="best", dataloaders=dm.test_dataloader())
    result = result._append(result_row)

print(result.describe().loc[["mean", "std"], :])
