import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import datetime
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import os


project_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
project_name = os.path.basename(project_path)
server_name = os.uname()[1]
remote_root = "/data/hanyang/sepsis/"


prefix = "cohort3"
raw_data_path = os.path.join(remote_root, "cohort_3")
manual_data_path = os.path.join(remote_root, "manual_tables")
remote_project_path = os.path.join(remote_root, project_name)
processed_data_path = os.path.join(raw_data_path, "data_processed")
tmp_data_path = os.path.join(raw_data_path, "data_tmp")


def process_comorb_binary():
    print("Processing comorbidity into binary features...")
    # process comorbidities
    raw_labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    raw_comorb = pd.read_csv(join(raw_data_path, '{}_diagnoses_for_comorbidities.csv'.format(prefix)), low_memory=False)
    raw_comorb['diagnosis_code'] = raw_comorb['ICDX_DIAGNOSIS_CODE'].astype(str)
    raw_comorb['diagnosis_code_fam'] = raw_comorb['ICDX_DIAGNOSIS_CODE'].astype(str).apply(lambda x: x.split('.')[0])
    comorb = raw_comorb[['reference_no', 'diagnosis_code_fam', 'diagnosis_code']]
    # comorb.to_csv(join(save_dir, 'data_comorb_raw.csv'), index=False)

    df_comorb_fam = pd.pivot_table(comorb[['reference_no', 'diagnosis_code_fam']],
                                   index=['reference_no'],
                                   columns='diagnosis_code_fam',
                                   aggfunc=len,
                                   fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb_fam = (
        raw_labels.iloc[:, :2].merge(df_comorb_fam, how='inner', left_on='patient_id', right_on='reference_no')
            .drop(columns=['reference_no']))
    df_comorb_fam.to_csv(join(processed_data_path, 'data_comorb_fam.csv'), index=False)
    sdf_comorb_fam = df_comorb_fam.astype(pd.SparseDtype("int", 0))
    sdf_comorb_fam.to_pickle(join(processed_data_path, 'data_comorb_fam.pickle'))

    df_comorb = pd.pivot_table(comorb[['reference_no', 'diagnosis_code']],
                               index=['reference_no'],
                               columns='diagnosis_code',
                               aggfunc=len,
                               fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb = (raw_labels.iloc[:, :2].merge(df_comorb, how='inner', left_on='patient_id', right_on='reference_no')
                 .drop(columns=['reference_no']))
    df_comorb.to_csv(join(processed_data_path, 'data_comorb.csv'), index=False)
    sdf_comorb = df_comorb.astype(pd.SparseDtype("int", 0))
    sdf_comorb.to_pickle(join(processed_data_path, 'data_comorb.pickle'))


def process_vitals_simple():
    print("Processing vitals...")
    # process vitals
    data = pd.read_csv(join(raw_data_path, '{}_vitals.csv'.format(prefix)))
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    data['MEAS_DATE'] = pd.to_datetime(data['MEAS_DATE'])
    data = (data[['patient_id', 'admission_id', 'MEAS_DATE', 'RESULT_NAME', 'MEASUREMENT']]
            .sort_values(by=['patient_id', 'admission_id', 'RESULT_NAME', 'MEAS_DATE']))
    data[['patient_id', 'admission_id', 'RESULT_NAME']] = data[['patient_id', 'admission_id', 'RESULT_NAME']].astype(
        str)
    data = data.drop(index=data[data['MEASUREMENT'] == ' '].index).reset_index()
    data['MEASUREMENT'] = data['MEASUREMENT'].astype(float)
    if resample_window:
        data = (data.groupby(['patient_id', 'admission_id', 'RESULT_NAME'])
                .resample(rule=resample_window, on='MEAS_DATE')
                .mean(numeric_only=True).drop(columns=['index']).reset_index())
    data = (data.pivot_table(index=['patient_id', 'admission_id', 'MEAS_DATE'],
                             columns='RESULT_NAME', values='MEASUREMENT').add_prefix('vital_').reset_index())
    data.to_csv(join(processed_data_path, 'data_vitals.csv'), index=False)


    # extract last values before each infections
    processed_data = data
    time_column = 'MEAS_DATE'
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['admission_id'], col_dates['collection_date']))
        data_table = processed_data.copy()
        data_table['collection_date'] = data_table['admission_id'].apply(
            lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data_table[time_column] = pd.to_datetime(data_table[time_column])
        data_table['collection_date'] = pd.to_datetime(data_table['collection_date'])
        # last value
        data_table = data_table[data_table[time_column] <= data_table['collection_date']]
        data_last = data_table.groupby(['patient_id', 'admission_id']).last().drop(
            columns=['collection_date', time_column]).reset_index()
        # max/min within 24h before
        data_table = data_table[np.logical_and(data_table[time_column] <= data_table['collection_date'],
                                               data_table[time_column] > data_table[
                                                   'collection_date'] - datetime.timedelta(
                                                   days=1))]
        data_min = data_table.groupby(['patient_id', 'admission_id']).min().drop(
            columns=['collection_date', time_column]).reset_index()
        data_max = data_table.groupby(['patient_id', 'admission_id']).max().drop(
            columns=['collection_date', time_column]).reset_index()
        data_table = data_min.merge(data_max.iloc[:, 1:], on='admission_id', suffixes=['_min', '_max'])
        data_table = data_table.merge(data_last.iloc[:, 1:], on='admission_id')
        data_table['admission_instance'] = data_table['admission_id'].astype(str) + '-' + str(infection_id)
        data_out = pd.concat([data_out, data_table], axis=0)

    data_out = data_out.sort_values(['patient_id', 'admission_id', 'admission_instance'])
    data_cols = processed_data.columns[3:].tolist()
    data_cols = data_cols + [ele + '_min' for ele in data_cols] + [ele + '_max' for ele in data_cols]
    data_out = data_out[['patient_id', 'admission_id', 'admission_instance'] + data_cols]
    data_out.to_csv(join(processed_data_path, 'data_last_min_max_vitals_by_instance.csv'), index=False)


def process_labs_simple():
    print("Processing labs...")
    # process labs
    data = pd.read_csv(join(raw_data_path, '{}_labs.csv'.format(prefix)))
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    data = data[~data['PRINT_NAME'].isna()]
    data['patient_id'] = data['patient_id'].astype(int)

    data['LAST_UPDATED_TMSTP'] = pd.to_datetime(data['LAST_UPDATED_TMSTP'])
    data['result'] = data['NORMALIZED_RESULT_VALUE'].astype('str').str.extract('([0-9][,.]*[0-9]*)').astype(float)
    data = data[~data['result'].isna()]
    data = (data[['patient_id', 'admission_id', 'LAST_UPDATED_TMSTP', 'PRINT_NAME', 'result']]
            .sort_values(by=['patient_id', 'admission_id', 'PRINT_NAME', 'LAST_UPDATED_TMSTP']))
    data[['patient_id', 'admission_id', 'PRINT_NAME']] = data[['patient_id', 'admission_id', 'PRINT_NAME']].astype(str)
    if resample_window:
        data = (data.groupby(['patient_id', 'admission_id', 'PRINT_NAME'])
                .resample(rule=resample_window, on='LAST_UPDATED_TMSTP')
                .mean(numeric_only=True).reset_index())
    data = (data.pivot_table(index=['patient_id', 'admission_id', 'LAST_UPDATED_TMSTP'],
                             columns='PRINT_NAME', values='result').add_prefix('lab_').reset_index())
    data.to_csv(join(processed_data_path, 'data_labs.csv'), index=False)


    # extract last values before each infections
    processed_data = data
    time_column = 'LAST_UPDATED_TMSTP'
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['admission_id'], col_dates['collection_date']))
        data_table = processed_data.copy()
        data_table['collection_date'] = data_table['admission_id'].apply(
            lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data_table[time_column] = pd.to_datetime(data_table[time_column])
        data_table['collection_date'] = pd.to_datetime(data_table['collection_date'])
        # last value
        data_table = data_table[data_table[time_column] <= data_table['collection_date']]
        data_last = data_table.groupby(['patient_id', 'admission_id']).last().drop(
            columns=['collection_date', time_column]).reset_index()
        # max/min within 24h before
        data_table = data_table[np.logical_and(data_table[time_column] <= data_table['collection_date'],
                                               data_table[time_column] > data_table[
                                                   'collection_date'] - datetime.timedelta(
                                                   days=1))]
        data_min = data_table.groupby(['patient_id', 'admission_id']).min().drop(
            columns=['collection_date', time_column]).reset_index()
        data_max = data_table.groupby(['patient_id', 'admission_id']).max().drop(
            columns=['collection_date', time_column]).reset_index()
        data_table = data_min.merge(data_max.iloc[:, 1:], on='admission_id', suffixes=['_min', '_max'])
        data_table = data_table.merge(data_last.iloc[:, 1:], on='admission_id')
        data_table['admission_instance'] = data_table['admission_id'].astype(str) + '-' + str(infection_id)
        data_out = pd.concat([data_out, data_table], axis=0)

    data_out = data_out.sort_values(['patient_id', 'admission_id', 'admission_instance'])
    data_cols = processed_data.columns[3:].tolist()
    data_cols = data_cols + [ele + '_min' for ele in data_cols] + [ele + '_max' for ele in data_cols]
    data_out = data_out[['patient_id', 'admission_id', 'admission_instance'] + data_cols]
    data_out.to_csv(join(processed_data_path, 'data_last_min_max_labs_by_instance.csv'), index=False)


if __name__ == "__main__":
    # paths and global variables
    resample_window = '4H'  # None or '1H'
    time_window = 5  # days

    process_vitals_simple()
    process_labs_simple()
    process_comorb_binary()
