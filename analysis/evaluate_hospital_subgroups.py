import torch
import os
torch.set_float32_matmul_precision('high')

import pandas as pd
from utils.utils_data import select_subgroup, instance_filter
from utils.utils_evaluate_torch import build_lit_model, test_all_subgroups, test_subgroup
os.chdir("../")


save_data_dir = '../data_prepared/'


# load from checkpoint
checkpoints = [
    torch.load("./logs/AMR/AggMM/version_34/checkpoints/epoch=0-step=468.ckpt"),
    torch.load("./logs/AMR/AggMM/version_35/checkpoints/epoch=0-step=468.ckpt"),
    torch.load("./logs/AMR/AggMM/version_36/checkpoints/epoch=1-step=936.ckpt"),
    torch.load("./logs/AMR/AggMM/version_37/checkpoints/epoch=0-step=468.ckpt"),
    torch.load("./logs/AMR/AggMM/version_38/checkpoints/epoch=0-step=468.ckpt"),
]

LitModel, trainer, dm = build_lit_model(checkpoints)


all_data = pd.read_csv('../data_prepared/data_table_fam_comorb.csv').set_index(['patient_id', 'admission_id', 'infection_id'])


# test on hospital 2574
test_data = all_data.loc[dm.idx_test.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()
# test_subgroup(LitModel, trainer, dm, checkpoints, test_data, subgroup='16')
sub_indices = select_subgroup(test_data, group='14')
sub_test_data = all_data.loc[sub_indices.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()
df_results_0 = test_all_subgroups(LitModel, trainer, dm, checkpoints, sub_test_data, postfix='hospital_2574')
# test_subgroup(LitModel, trainer, dm, checkpoints, sub_test_data, subgroup='11', binary=True)


# test on hospital 3148
test_data = all_data.loc[dm.idx_test.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()
# test_subgroup(LitModel, trainer, dm, checkpoints, test_data, subgroup='17')
sub_indices = select_subgroup(test_data, group='15')
sub_test_data = all_data.loc[sub_indices.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()
df_results_1 = test_all_subgroups(LitModel, trainer, dm, checkpoints,  sub_test_data, postfix='hospital_3148')
# test_subgroup(LitModel, trainer, dm, checkpoints, sub_test_data, subgroup='11', binary=True)


# test on hospital 6729
test_data = all_data.loc[dm.idx_test.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()
# test_subgroup(LitModel, trainer, dm, checkpoints, test_data, subgroup='19')
sub_indices = select_subgroup(test_data, group='17')
sub_test_data = all_data.loc[sub_indices.set_index(['patient_id', 'admission_id', 'infection_id']).index].reset_index()
df_results_2 = test_all_subgroups(LitModel, trainer, dm, checkpoints,  sub_test_data, postfix='hospital_6729')
# test_subgroup(LitModel, trainer, dm, checkpoints, sub_test_data, subgroup='11', binary=True)

