from os.path import join

import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
import lightning as L
from utils.utils_data import load_data, RandomizedGroupKFold
import numpy as np
import pickle
from paths import processed_data_path, comorb_vocab_size


class myDataset(Dataset):
    def __init__(self,
                 indices,
                 all_data,
                 max_codes,
                 static_size,
                 ts_size,
                 max_procedure=50):
        self.static, self.codes, self.timeseries, self.procedure, self.labels, self.label_mask = all_data
        self.indices = indices
        self.static_size = static_size
        self.ts_size = ts_size
        self.comorb_size = (max_codes,)
        self.procedure_size = (max_procedure,)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.indices.iloc[idx]
        patient_id, admission_id, infection_id = row.patient_id, row.admission_id, row.infection_id
        modality_mask = np.zeros((3,))

        # Static
        if (patient_id, admission_id, infection_id) in self.static.index:
            static = self.static.loc[(patient_id, admission_id, infection_id)].astype(float).values
            modality_mask[0] = 1
        else:
            static = np.zeros(self.static_size)

        # Longitudinal
        X, X_interv, X_filled, X_mask = self.timeseries
        if (patient_id, admission_id, infection_id) in X:
            ts = X[(patient_id, admission_id, infection_id)]
            ts_interv = X_interv[(patient_id, admission_id, infection_id)]
            ts_filled = X_filled[(patient_id, admission_id, infection_id)]
            ts_mask = X_mask[(patient_id, admission_id, infection_id)]
            modality_mask[1] = 1
        else:
            ts = np.zeros(self.ts_size)
            ts_interv = np.zeros(self.ts_size)
            ts_filled = np.zeros(self.ts_size)
            ts_mask = np.zeros(self.ts_size)

        # Comorbidity
        if patient_id in self.codes:
            comorb = self.codes[patient_id][: self.comorb_size[0]]
            modality_mask[2] = 1
        else:
            comorb = np.zeros(self.comorb_size)
            comorb[0] = comorb_vocab_size + 1

        # # Procedure
        # procedure = np.zeros(self.procedure_size)
        # if (patient_id, admission_id, infection_id) in self.procedure:
        #     arr = self.procedure[(patient_id, admission_id, infection_id)]
        #     procedure[:len(arr)] = arr[:self.procedure_size[0]]
        # else:
        #     procedure[0] = 6489

        # Labels
        label = self.labels.loc[(patient_id, admission_id, infection_id)].values
        label_mask = self.label_mask.loc[(patient_id, admission_id, infection_id)].values

        return {
            "static": torch.Tensor(static),
            "comorb": torch.LongTensor(comorb),
            "ts": {
                "X": torch.Tensor(ts),
                "delta": torch.Tensor(ts_interv),
                "X_LOCF": torch.Tensor(ts_filled),
                "mask": torch.Tensor(ts_mask),
                "mean": torch.Tensor(np.zeros(self.ts_size[1], ))
            },
            # "procedure": torch.LongTensor(procedure),
            "modality_mask": torch.BoolTensor(modality_mask),
            "label": torch.Tensor(label),
            "label_mask": torch.BoolTensor(label_mask),
        }


class sepsisDataModule(L.LightningDataModule):
    def __init__(self,
                 max_codes=200,
                 batch_size=128,
                 pin_memory=False,
                 num_workers=1,
                 multiclass=False,
                 task="AMR",
                 use_unlabeled=False,
                 ):
        super().__init__()
        self.max_codes = max_codes
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.data_dir = processed_data_path
        self.multiclass = multiclass
        self.task = task
        self.use_unlabeled = use_unlabeled
        self.prepare_data()

    def prepare_labels_binary(self):
        # load labels
        labels = self.labels_raw.set_index(['patient_id', 'admission_id', 'infection_id'])[['SS', 'UN']]
        labels['label'] = ~labels['SS']
        labels.loc[labels['UN'], 'label'] = 0
        labels['label'] = labels['label'].astype(int)
        labels = labels[['label']]
        return labels

    def prepare_labels_multiclass(self):
        labels = self.labels_raw.set_index(['patient_id', 'admission_id', 'infection_id'])[['SS', 'RS', 'RR', 'UN']]
        labels.loc[labels.SS, 'label'] = 0
        labels.loc[labels.RS, 'label'] = 1
        labels.loc[labels.RR, 'label'] = 2
        labels.loc[labels.UN, 'label'] = 0
        labels['label'] = labels['label'].astype(int)
        labels = labels[['label']]
        return labels

    def prepare_labels_GNB(self):
        # load labels
        labels = self.labels_raw.set_index(['patient_id', 'admission_id', 'infection_id'])[['GNB', 'UN']]
        labels = labels.rename(columns={'GNB': 'label'})
        labels.loc[labels['UN'], 'label'] = 0
        labels['label'] = labels['label'].astype(int)
        return labels

    def prepare_static(self):
        data_demo = pd.read_csv(join(self.data_dir, 'deep_static.csv'))
        return data_demo.set_index(['patient_id', 'admission_id', 'infection_id'])

    def prepare_comorb(self):
        with open(join(self.data_dir, 'deep_comorb_codes_{}.pickle'.format(self.max_codes)), 'rb') as f:
            codes = pickle.load(f)
        return codes

    def prepare_procedure(self):
        with open(join(self.data_dir, 'deep_procedure_codes.pickle'), 'rb') as f:
            codes_arr = pickle.load(f)
        return codes_arr

    def prepare_timeseries(self):
        with open(join(self.data_dir, 'deep_timeseries_4H_normalized.pickle'), 'rb') as f:
            ts = pickle.load(f)
        with open(join(self.data_dir, 'deep_timeseries_4H_interv.pickle'), 'rb') as f:
            ts_interv = pickle.load(f)
        with open(join(self.data_dir, 'deep_timeseries_4H_filled.pickle'), 'rb') as f:
            ts_filled = pickle.load(f)
        with open(join(self.data_dir, 'deep_timeseries_4H_mask.pickle'), 'rb') as f:
            ts_mask = pickle.load(f)
        return [ts, ts_interv, ts_filled, ts_mask]

    def prepare_data(self):
        self.labels_raw = pd.read_csv(join(self.data_dir, 'df_label_full.csv'))
        label_mask = ~self.labels_raw.set_index(['patient_id', 'admission_id', 'infection_id'])[["UN"]]
        if not self.use_unlabeled:
            self.labels_raw = self.labels_raw[~self.labels_raw.UN]  # excluding data with unknown labels

        if self.task == "GNB":
            self.multiclass = False
            labels = self.prepare_labels_GNB()
        elif self.task == "AMR":
            labels = self.prepare_labels_multiclass() if self.multiclass else self.prepare_labels_binary()
        else:
            ValueError("Please specify task: GNB or AMR.")

        data_static = self.prepare_static()
        data_comorb = self.prepare_comorb()
        data_timeseries = self.prepare_timeseries()
        data_procedure = self.prepare_procedure()

        # indices of patient admissions
        self.indices_labeled = self.labels_raw[~self.labels_raw['UN']][['patient_id', 'admission_id', 'infection_id']]
        self.all_data = [data_static, data_comorb, data_timeseries, data_procedure, labels, label_mask]

        # data shape
        self.size_static = (len(data_static.columns),)
        self.size_timeseries = np.shape(next(iter(data_timeseries[0].values())))

    def setup(self, stage=None):
        # train/valid/test split
        cv = RandomizedGroupKFold(groups=self.indices_labeled['admission_id'].to_numpy(),
                                  n_splits=5,
                                  random_state=42)
        train_val_ix, test_ix = cv[0]
        self.idx_train_val, self.idx_test = self.indices_labeled.iloc[train_val_ix], self.indices_labeled.iloc[test_ix]

        gss = GroupShuffleSplit(n_splits=2, test_size=1 / 8, random_state=42)
        splits = gss.split(self.idx_train_val, groups=self.idx_train_val['admission_id'])
        train_ix, valid_ix = next(splits)
        self.idx_train, self.idx_valid = self.idx_train_val.iloc[train_ix], self.idx_train_val.iloc[valid_ix]

        if self.use_unlabeled:
            self.idx_train = pd.concat([
                self.idx_train,
                self.labels_raw[self.labels_raw['UN']][['patient_id', 'admission_id', 'infection_id']]
            ], axis=0)

    def train_dataloader(self):
        return DataLoader(
            myDataset(self.idx_train, self.all_data, self.max_codes, self.size_static, self.size_timeseries),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            myDataset(self.idx_valid, self.all_data, self.max_codes, self.size_static, self.size_timeseries),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self, idx_test=None):
        if idx_test is None:
            idx_test = self.idx_test
        return DataLoader(
            myDataset(idx_test, self.all_data, self.max_codes, self.size_static, self.size_timeseries),
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False
        )
