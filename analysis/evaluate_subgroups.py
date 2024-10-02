import torch
torch.set_float32_matmul_precision('high')
import os

import pandas as pd
from utils.utils_data import select_subgroup, instance_filter
from utils.utils_evaluate_torch import build_lit_model, test_all_subgroups, test_subgroup, test_all_hospitals
os.chdir("../")


save_data_dir = '../data_prepared/'


# # load from checkpoint
# checkpoints = [
#     torch.load("../logs/complete_cases_202404/version_0/checkpoints/epoch=0-step=468.ckpt"),
#     torch.load("../logs/complete_cases_202404/version_1/checkpoints/epoch=1-step=936.ckpt"),
#     torch.load("../logs/complete_cases_202404/version_2/checkpoints/epoch=0-step=468.ckpt"),
#     torch.load("../logs/complete_cases_202404/version_3/checkpoints/epoch=1-step=936.ckpt"),
#     torch.load("../logs/complete_cases_202404/version_4/checkpoints/epoch=0-step=468.ckpt"),
# ]

checkpoints = [
    torch.load("./logs/AMR/AggMM/version_34/checkpoints/epoch=0-step=468.ckpt"),
    torch.load("./logs/AMR/AggMM/version_35/checkpoints/epoch=0-step=468.ckpt"),
    torch.load("./logs/AMR/AggMM/version_36/checkpoints/epoch=1-step=936.ckpt"),
    torch.load("./logs/AMR/AggMM/version_37/checkpoints/epoch=0-step=468.ckpt"),
    torch.load("./logs/AMR/AggMM/version_38/checkpoints/epoch=0-step=468.ckpt"),
]

LitModel, trainer, dm = build_lit_model(checkpoints)


# extract test data
all_data = pd.read_csv('../data_prepared/data_table_fam_comorb.csv').set_index(['patient_id', 'admission_id', 'infection_id'])
test_data = all_data.loc[dm.idx_test.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()


# plot results
test_subgroup(LitModel, trainer, dm, checkpoints, test_data, subgroup='0', binary=False)

# # sub-cohort results
# test_all_subgroups(LitModel, trainer, dm, checkpoints,  test_data, postfix='NN')
# test_all_hospitals(LitModel, trainer, dm, checkpoints,  test_data, postfix='NN')