import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import datetime
import os
from joblib import Parallel, delayed
import collections
import pickle
from sklearn import preprocessing
from os.path import join

import multiprocessing
import numpy as np
from utils.utils_graph import GraphEmb
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


def minmax_scale(df):
    for col in df.columns:
        df.loc[:, col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def standard_scale(df):
    for col in df.columns:
        df.loc[:, col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def process_static():
    print("Processing static variables...")
    # load input
    labels_raw = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))
    demo = pd.read_csv(join(processed_data_path, 'data_demographics.csv'))
    vasop = pd.read_csv(join(processed_data_path, 'data_last_vasop_by_instance.csv'))
    abx = pd.read_csv(join(processed_data_path, 'data_last_abx_by_instance.csv'))
    diagnoses = pd.read_csv(join(processed_data_path, 'data_diagnoses.csv'))
    proc = pd.read_csv(join(raw_data_path, '{}_procedures.csv'.format(prefix)))

    # initial table
    data_table = labels_raw[['patient_id', 'admission_id', 'infection_id']]
    data_table = data_table.assign(
        admission_instance=data_table['admission_id'].astype(str) + '-' + data_table['infection_id'].astype(str))
    # add demographics
    data_table = data_table.merge(demo, how='left', on=['patient_id', 'admission_id'])
    # add vasopressors
    data_table = data_table.merge(vasop, how='left', on=['patient_id', 'admission_id', 'admission_instance']).fillna(
        value=0)
    # add abx
    data_table = data_table.merge(abx, how='left', on=['patient_id', 'admission_id', 'admission_instance']).fillna(
        value=0)

    # add intubation
    labels_raw['start_date'] = pd.to_datetime(labels_raw['admit_date']).dt.date
    labels_raw['end_date'] = pd.to_datetime(labels_raw['collection_date']).dt.date
    labels_raw['date_diff'] = (pd.to_datetime(labels_raw['end_date']) - pd.to_datetime(
        labels_raw['start_date'])) / np.timedelta64(1, 'D')

    intu_true = proc[proc.DESCRIPTION.str.contains('Endotracheal', na=False)].admission_id.unique().tolist()
    data_table['mechanical_ventilation'] = 0
    data_table.loc[data_table.admission_id.isin(intu_true), 'mechanical_ventilation'] = 1

    # add history
    # sensitivity history
    labels_raw['resistant'] = np.logical_or(labels_raw['RS'], labels_raw['RR'])
    labels_raw['resistance_history'] = labels_raw.groupby('admission_id')['resistant'].shift(1).cumsum().fillna(0)
    labels_raw['resistance_history'] = labels_raw['resistance_history'] > 0
    data_table['resistance_history'] = labels_raw['resistance_history']
    # re-admission (within 3 month)
    adm_hist = labels_raw.sort_values(['patient_id', 'admit_date']).groupby(
        ['patient_id', 'admission_id']).first().reset_index()
    adm_hist['last_discharge_date'] = adm_hist.groupby('patient_id')['discharge_date'].shift(1)
    adm_hist['readmission_interval'] = (pd.to_datetime(adm_hist['admit_date']) - pd.to_datetime(
        adm_hist['last_discharge_date'])) / np.timedelta64(1, 'D')
    adm_hist['readmission'] = adm_hist['readmission_interval'].transform(
        lambda x: x <= 90 if not np.isnan(x) else False)
    readmission = adm_hist[['admission_id', 'readmission']].set_index('admission_id')
    data_table['readmission'] = data_table['admission_id'].transform(lambda x: readmission.loc[x, 'readmission'])

    # add pneumonia diagnosis
    diagnoses = diagnoses.rename(columns={'instance_id': 'admission_instance'})
    data_table = data_table.merge(diagnoses[['admission_instance', 'pneumonia_community', 'pneumonia_acquired']],
                                  how='left', on='admission_instance')
    data_table[['pneumonia_community', 'pneumonia_acquired']] = data_table[
        ['pneumonia_community', 'pneumonia_acquired']].astype(int)

    # add time intervals
    data_table = data_table.merge(diagnoses[['admission_instance', 'time_since_admission']], how='left',
                                  on='admission_instance')

    # data normalization
    normalizer = preprocessing.MinMaxScaler()
    data_table.iloc[:, 4:] = normalizer.fit_transform(data_table.iloc[:, 4:])
    data_table = data_table.drop(columns=['admission_instance'])

    # save output
    data_table.to_csv(os.path.join(processed_data_path, 'deep_static.csv'), index=False)


def process_comorb(maxlen=200):
    print("Processing comorbidities...")

    def standardize(code):
        if len(str(code)) > 3:
            if '.' not in str(code):
                code = str(code)[:3] + '.' + str(code)[3:]
        return code

    def convert_icd9_to_10(row):
        code = standardize(row.ICDX_DIAGNOSIS_CODE)
        if row.ICDX_VERSION_NO == 'ICD9' or code[0].isdigit():
            if code in icd9to10_dict:
                code = icd9to10_dict[code]
            else:
                code = "NOT FOUND"

        return code

    # load files
    raw_labels = pd.read_csv(os.path.join(processed_data_path, 'df_label_full.csv'))
    raw_comorb = pd.read_csv(os.path.join(raw_data_path, '{}_diagnoses_for_comorbidities.csv'.format(prefix)), low_memory=False)

    # convert ICD9 to ICD10 codes
    icd9to10 = pd.read_csv(os.path.join(manual_data_path, 'icd9to10.csv'))
    icd9to10['icd9cm_standard'] = icd9to10['icd9cm'].apply(lambda x: standardize(x))
    icd9to10['icd10cm_standard'] = icd9to10['icd10cm'].apply(lambda x: standardize(x))
    icd9to10_dict = dict(zip(icd9to10['icd9cm_standard'].values, icd9to10['icd10cm_standard'].values))
    raw_comorb['diagnosis_code'] = raw_comorb.progress_apply(lambda row: convert_icd9_to_10(row), axis=1)
    comorb = raw_comorb.drop(index=raw_comorb[raw_comorb.diagnosis_code == 'NOT FOUND'].index)
    comorb = comorb.sort_values(by=['reference_no', 'REG_NO'])[['reference_no', 'REG_NO', 'diagnosis_code']]

    adm2idx = {}
    for pid in tqdm(comorb.reference_no.unique().tolist()):
        df = comorb[comorb.reference_no == pid]
        adm2idx.update(dict(zip(df.REG_NO.unique(), range(len(df.REG_NO.unique())))))
    comorb.loc[:, 'visit_no'] = comorb.REG_NO.apply(lambda x: adm2idx[x])
    comorb.visit_no = comorb.visit_no.astype(int)

    # get family codes
    comorb['diagnosis_code_fam'] = comorb['diagnosis_code'].apply(lambda x: x.split('.')[0])
    comorb = comorb[['reference_no', 'visit_no', 'diagnosis_code', 'diagnosis_code_fam']]
    comorb = comorb.drop_duplicates().reset_index()
    comorb.to_csv(os.path.join(processed_data_path, 'deep_comorb_raw.csv'), index=False)

    df_comorb_fam = pd.pivot_table(comorb[['reference_no', 'diagnosis_code_fam']],
                                   index=['reference_no'],
                                   columns='diagnosis_code_fam',
                                   aggfunc=len,
                                   fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb_fam = (
        raw_labels.iloc[:, :2].merge(df_comorb_fam, how='inner', left_on='patient_id', right_on='reference_no')
            .drop(columns=['reference_no']))
    df_comorb_fam.to_csv(os.path.join(processed_data_path, 'deep_comorb_fam.csv'), index=False)

    df_comorb = pd.pivot_table(comorb[['reference_no', 'diagnosis_code']],
                               index=['reference_no'],
                               columns='diagnosis_code',
                               aggfunc=len,
                               fill_value=0).astype(bool).astype(int).reset_index()
    df_comorb = (raw_labels.iloc[:, :2].merge(df_comorb, how='inner', left_on='patient_id', right_on='reference_no')
                 .drop(columns=['reference_no']))
    df_comorb.to_csv(os.path.join(processed_data_path, 'deep_comorb_ori.csv'), index=False)

    # covert into arrays
    comorb = pd.read_csv(os.path.join(processed_data_path, 'deep_comorb_raw.csv'))

    col = 'diagnosis_code'
    code_list = comorb.sort_values(col)[col].unique().tolist()
    code2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['code_id'] = comorb[col].apply(lambda x: code2id[x])

    col = 'diagnosis_code_fam'
    code_list = comorb.sort_values(col)[col].unique().tolist()
    fam2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['fam_id'] = comorb[col].apply(lambda x: fam2id[x])

    col = 'diagnosis_code_cat'
    comorb[col] = comorb['diagnosis_code'].apply(lambda x: x[0])
    code_list = comorb.sort_values(col)[col].unique().tolist()
    cat2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['cat_id'] = comorb[col].apply(lambda x: cat2id[x])

    codes_arr = collections.defaultdict()
    fam_arr = collections.defaultdict()
    cat_arr = collections.defaultdict()
    for pid in tqdm(comorb.reference_no.unique().tolist()):
        df = comorb[comorb.reference_no == pid]

        codes = np.array(df['code_id'].unique().tolist())
        tmp = np.zeros((maxlen,))
        tmp[: min(len(codes), maxlen)] = codes[: maxlen]
        codes_arr[pid] = tmp

        fams = np.array(df['fam_id'].unique().tolist())
        tmp = np.zeros((maxlen,))
        tmp[: min(len(fams), maxlen)] = fams[: maxlen]
        fam_arr[pid] = tmp

        cats = np.array(df['cat_id'].unique().tolist())
        tmp = np.zeros((maxlen,))
        tmp[: min(len(cats), maxlen)] = cats[: maxlen]
        cat_arr[pid] = tmp

    with open(os.path.join(processed_data_path, 'deep_comorb_codes_{}.pickle'.format(maxlen)), 'wb') as f:
        pickle.dump(codes_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(processed_data_path, 'deep_comorb_fams_{}.pickle'.format(maxlen)), 'wb') as f:
        pickle.dump(fam_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(processed_data_path, 'deep_comorb_cats_{}.pickle'.format(maxlen)), 'wb') as f:
        pickle.dump(cat_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    # consider visits
    comorb = pd.read_csv(os.path.join(processed_data_path, 'deep_comorb_raw.csv'))

    col = 'diagnosis_code'
    code_list = comorb.sort_values(col)[col].unique().tolist()
    code2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['code_id'] = comorb[col].apply(lambda x: code2id[x])

    codes_arr = collections.defaultdict()
    for pid in tqdm(comorb.reference_no.unique().tolist()):
        df = comorb[comorb.reference_no == pid]
        visits = []
        for visit in df.visit_no.unique():
            visits.append(list(df.loc[df.visit_no == visit, 'code_id'].values))

        tmp = np.zeros((100, 100))
        max_len = len(max(visits, key=len))
        visit_arr = np.array([i + [0] * (max_len - len(i)) for i in visits])
        visit_arr = visit_arr[: 100, : 100]
        tmp[: len(visits), : max_len] = visit_arr
        codes_arr[pid] = tmp

    with open(os.path.join(processed_data_path, 'deep_comorb_visits.pickle'), 'wb') as f:
        pickle.dump(codes_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    col = 'diagnosis_code_fam'
    code_list = comorb.sort_values(col)[col].unique().tolist()
    fam2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['fam_id'] = comorb[col].apply(lambda x: fam2id[x])

    codes_arr = collections.defaultdict()
    for pid in tqdm(comorb.reference_no.unique().tolist()):
        df = comorb[comorb.reference_no == pid]
        visits = []
        for visit in df.visit_no.unique():
            visits.append(list(df.loc[df.visit_no == visit, 'fam_id'].values))

        tmp = np.zeros((100, 100))
        max_len = len(max(visits, key=len))
        visit_arr = np.array([i + [0] * (max_len - len(i)) for i in visits])
        visit_arr = visit_arr[: 100, : 100]
        tmp[: len(visits), : max_len] = visit_arr
        codes_arr[pid] = tmp

    with open(os.path.join(processed_data_path, 'deep_comorb_visits_fam.pickle'), 'wb') as f:
        pickle.dump(codes_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    col = 'diagnosis_code_cat'
    comorb[col] = comorb['diagnosis_code'].apply(lambda x: x[0])
    code_list = comorb.sort_values(col)[col].unique().tolist()
    cat2id = dict(zip(code_list, range(1, len(code_list) + 1)))
    comorb['cat_id'] = comorb[col].apply(lambda x: cat2id[x])

    codes_arr = collections.defaultdict()
    for pid in tqdm(comorb.reference_no.unique().tolist()):
        df = comorb[comorb.reference_no == pid]
        visits = []
        for visit in df.visit_no.unique():
            visits.append(list(df.loc[df.visit_no == visit, 'cat_id'].values))

        tmp = np.zeros((100, 100))
        max_len = len(max(visits, key=len))
        visit_arr = np.array([i + [0] * (max_len - len(i)) for i in visits])
        visit_arr = visit_arr[: 100, : 100]
        tmp[: len(visits), : max_len] = visit_arr
        codes_arr[pid] = tmp

    with open(os.path.join(processed_data_path, 'deep_comorb_visits_cat.pickle'), 'wb') as f:
        pickle.dump(codes_arr, f, protocol=pickle.HIGHEST_PROTOCOL)


def graph_embedding(embed_size):
    print("Training graph embedding...")

    # Graph embedding
    comorb = pd.read_csv(os.path.join(processed_data_path, 'deep_comorb_raw.csv'))
    graph_emb = GraphEmb(embed_size=embed_size)
    graph_emb.train_embedding()

    code_list = comorb.sort_values('diagnosis_code').diagnosis_code.unique()
    comorb_codes = graph_emb.codes

    embeddings = []
    zero_vec = np.zeros((embed_size,))
    count = 0
    for code in tqdm(code_list):
        if code in comorb_codes:
            embeddings.append(graph_emb.to_vec([code])[0])
        else:
            code = code[:5]  # only use first 5 positions
            try:
                embeddings.append(graph_emb.to_vec([code])[0])
            except:
                count += 1
                embeddings.append(zero_vec)
    df_embeddings = pd.DataFrame(data=code_list, columns=['code'])
    df_embeddings = pd.concat([df_embeddings, pd.DataFrame(embeddings)], axis=1)
    df_embeddings.to_csv(os.path.join(processed_data_path, 'icd10_embeddings_{}.csv'.format(embed_size)), index=False)


def organize_w_infection(data_table, time_column, collection_dates, time_window):
    organized_data = []
    for _, row in tqdm(collection_dates.iterrows()):
        date = pd.to_datetime(row.collection_date)
        data = data_table[data_table.admission_id == row.admission_id]
        data = data[np.logical_and(pd.to_datetime(data[time_column]) < date,
                                   pd.to_datetime(data[time_column]) >= date - datetime.timedelta(time_window))]
        data['infection_id'] = row.infection_id
        organized_data.append(data)

    organized_data = pd.concat(organized_data, axis=0)
    organized_data = organized_data[
        ['patient_id', 'admission_id', 'infection_id', time_column] + data_table.columns.tolist()[3:]]
    return organized_data


def process_vitals():
    print("Processing vitals...")
    def process_parallel(adm_ids, data_all, resample='1H'):
        data = data_all.loc[data_all.admission_id.isin(adm_ids)]
        data_pivot = (data.pivot_table(index=['patient_id', 'admission_id', 'MEAS_DATE'],
                                       columns='RESULT_NAME', values='MEASUREMENT').add_prefix('vital_').reset_index())
        if resample:
            data_pivot = (data_pivot.groupby(['patient_id', 'admission_id'])
                          .resample(resample, on='MEAS_DATE')
                          .mean(numeric_only=False).drop(columns=['patient_id', 'admission_id']).reset_index())

        return data_pivot

    # process vitals
    print("Loading raw data file...")
    vitals = pd.read_csv(os.path.join(raw_data_path, '{}_vitals.csv'.format(prefix)))
    print("Loaded.")
    vitals = vitals[['patient_id', 'admission_id', 'MEAS_DATE', 'RESULT_NAME', 'MEASUREMENT']]
    vitals[['patient_id', 'admission_id']] = vitals[['patient_id', 'admission_id']].astype(str)
    vitals = vitals.drop(index=vitals[vitals['MEASUREMENT'] == ' '].index).reset_index()
    vitals['MEASUREMENT'] = vitals['MEASUREMENT'].astype(float)
    vitals['MEAS_DATE'] = pd.to_datetime(vitals['MEAS_DATE'])

    # feature selection
    selection = pd.read_csv(os.path.join(manual_data_path, 'vital_variables_mcvg.csv'))
    to_keep = selection[selection['KEEP'] == 1].NAME.tolist()
    vitals = vitals[vitals.RESULT_NAME.isin(to_keep)]

    # multiprocessing
    num_cores = multiprocessing.cpu_count() - 6
    adm_ids = vitals.admission_id.unique()
    block_size = 1000
    adm_ids_list = [adm_ids[i: i + block_size] for i in range(0, len(adm_ids), block_size)]
    data_pivot_list = Parallel(n_jobs=num_cores)(
        delayed(process_parallel)(adm_ids, vitals, resample_window) for adm_ids in tqdm(adm_ids_list))
    data_pivot = pd.concat(data_pivot_list, axis=0).sort_values(by=['patient_id', 'admission_id', 'MEAS_DATE'])
    data_pivot.to_csv(os.path.join(processed_data_path, 'deep_vitals_22var_{}.csv'.format(resample_window)), index=False)

    # organize w.r.t. infection time
    data_pivot = pd.read_csv(os.path.join(processed_data_path, 'deep_vitals_22var_{}.csv'.format(resample_window)))
    collection_dates = pd.read_csv(os.path.join(processed_data_path, 'df_label_full.csv'))[
        ['patient_id', 'admission_id', 'infection_id', 'collection_date']]
    organized_data = organize_w_infection(data_pivot, 'MEAS_DATE', collection_dates, time_window)
    organized_data = organized_data.rename(columns={'MEAS_DATE': 'time'})
    organized_data.to_csv(os.path.join(processed_data_path, 'deep_vitals_22var_{}_organized.csv'.format(resample_window)),
                          index=False)


def process_labs():
    print("Processing labs...")
    def process_parallel(adm_ids, data_all, resample='1H'):
        data = data_all.loc[data_all.admission_id.isin(adm_ids)]
        data_pivot = (data.pivot_table(index=['patient_id', 'admission_id', 'LAST_UPDATED_TMSTP'],
                                       columns='PRINT_NAME', values='result').add_prefix('lab_').reset_index())
        if resample:
            data_pivot = (data_pivot.groupby(['patient_id', 'admission_id'])
                          .resample(resample, on='LAST_UPDATED_TMSTP')
                          .mean(numeric_only=False).drop(columns=['patient_id', 'admission_id']).reset_index())
        return data_pivot

    # process labs
    print("Loading raw data file...")
    labs = pd.read_csv(os.path.join(raw_data_path, '{}_labs.csv'.format(prefix)))
    print("Loaded.")
    labs = labs[~labs['PRINT_NAME'].isna()]
    labs['patient_id'] = labs['patient_id'].astype(int)

    labs['result'] = labs['NORMALIZED_RESULT_VALUE'].astype('str').str.extract('([0-9][,.]*[0-9]*)').astype(float)
    labs = labs[~labs['result'].isna()]
    labs = labs[['patient_id', 'admission_id', 'LAST_UPDATED_TMSTP', 'PRINT_NAME', 'result']]
    labs[['patient_id', 'admission_id', 'PRINT_NAME']] = labs[['patient_id', 'admission_id', 'PRINT_NAME']].astype(str)
    labs['LAST_UPDATED_TMSTP'] = pd.to_datetime(labs['LAST_UPDATED_TMSTP'])

    # feature selection
    selection = pd.read_csv(os.path.join(manual_data_path, 'lab_variables_mcvg.csv'))
    to_keep = selection[selection['KEEP'] == 1].NAME.tolist()
    labs = labs[labs.PRINT_NAME.isin(to_keep)]

    # multiprocessing
    num_cores = multiprocessing.cpu_count() - 6
    adm_ids = labs.admission_id.unique()
    block_size = 1000
    adm_ids_list = [adm_ids[i: i + block_size] for i in range(0, len(adm_ids), block_size)]
    data_pivot_list = Parallel(n_jobs=num_cores)(
        delayed(process_parallel)(adm_ids, labs, resample_window) for adm_ids in tqdm(adm_ids_list))
    data_pivot = pd.concat(data_pivot_list, axis=0).sort_values(by=['patient_id', 'admission_id', 'LAST_UPDATED_TMSTP'])
    data_pivot.to_csv(os.path.join(processed_data_path, 'deep_labs_44var_{}.csv'.format(resample_window)), index=False)

    # organize w.r.t. infection time
    data_pivot = pd.read_csv(os.path.join(processed_data_path, 'deep_labs_44var_{}.csv'.format(resample_window)))
    collection_dates = pd.read_csv(os.path.join(processed_data_path, 'df_label_full.csv'))[
        ['patient_id', 'admission_id', 'infection_id', 'collection_date']]
    organized_data = organize_w_infection(data_pivot, 'LAST_UPDATED_TMSTP', collection_dates, time_window)
    organized_data = organized_data.rename(columns={'LAST_UPDATED_TMSTP': 'time'})
    organized_data.to_csv(os.path.join(processed_data_path, 'deep_labs_44var_{}_organized.csv'.format(resample_window)),
                          index=False)


def combine_timeseries():
    print("Combining vitals and labs...")
    df_labs = pd.read_csv(join(processed_data_path, "deep_labs_44var_{}_organized.csv".format(resample_window)))
    df_vitals = pd.read_csv(join(processed_data_path, "deep_vitals_22var_{}_organized.csv".format(resample_window)))

    ratio_labs = df_labs.count() / len(df_labs)
    selected_labs = ratio_labs[ratio_labs > 0.01].index.tolist()

    ratio_vitals = df_vitals.count() / len(df_vitals)
    selected_vitals = ratio_vitals[ratio_vitals > 0.01].index.tolist()

    df_ts = pd.merge(df_vitals[selected_vitals], df_labs[selected_labs], how='outer',
                     on=['patient_id', 'admission_id', 'infection_id', 'time'])
    df_ts.to_csv(join(processed_data_path, 'deep_timeseries_organized.csv'), index=False)

    # normalize
    df_ts.iloc[:, 4:] = standard_scale(df_ts.iloc[:, 4:])
    df_ts.fillna(value=0).to_csv(
        os.path.join(processed_data_path, 'deep_timeseries_{}_normalized.csv'.format(resample_window)), index=False)

    # generate mask
    mask = df_ts.copy()
    mask.iloc[:, 4:] = mask.iloc[:, 4:].notnull().astype(int)
    mask.to_csv(os.path.join(processed_data_path, 'deep_timeseries_{}_mask.csv'.format(resample_window)), index=False)

    # generate intervals
    interv = df_ts.set_index(["patient_id", "admission_id", "infection_id"]).copy()
    for col in df_ts.columns[4:]:
        interv[col] = (
            interv[col].isnull().astype(int)
                .groupby([interv.index, interv[col].notnull().astype(int).cumsum()])
                .cumsum()
                .groupby(interv.index).shift(periods=1, fill_value=0).astype(int)
                .add(1)
        )
    interv = interv.reset_index()
    interv.iloc[:, 4:] = interv.iloc[:, 4:] / interv.iloc[:, 4:].max().max()
    interv.to_csv(os.path.join(processed_data_path, 'deep_timeseries_{}_interv.csv'.format(resample_window)), index=False)

    # impute missing data
    filled = df_ts.groupby(['patient_id', 'admission_id', 'infection_id']).apply(
        lambda x: x.ffill().bfill().fillna(0)).reset_index(drop=True)
    filled.to_csv(join(processed_data_path, 'deep_timeseries_{}_filled.csv'.format(resample_window)), index=False)

    # convert to arrays
    def save_array(df, name=None):
        ts_arr = collections.defaultdict()
        for group_id, df_group in tqdm(df.groupby(['patient_id', 'admission_id', 'infection_id'])):
            ts = np.zeros((24 // int(resample_window[0]) * 5, len(df_ts.columns) - 4))
            ts[-len(df_group):, :] = df_group.iloc[:, 4:].values
            ts_arr[group_id] = ts

        with open(os.path.join(processed_data_path, 'deep_timeseries_{}_{}.pickle'.format(resample_window, name)), 'wb') as f:
            pickle.dump(ts_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_array(df_ts.fillna(value=0), "normalized")
    save_array(filled, "filled")
    save_array(interv, "interv")
    save_array(mask, "mask")


def process_procedure():
    data_raw = pd.read_csv(join(raw_data_path, "{}_procedures.csv".format(prefix))).sort_values(
        by=["admission_id", "PROCEDURE_DATE"]).reset_index()
    labels = pd.read_csv(join(processed_data_path, 'df_label_full.csv'))

    time_column = 'PROCEDURE_DATE'
    data_out = pd.DataFrame()
    for infection_id in tqdm(labels['infection_id'].unique()):
        col_dates = labels[labels['infection_id'] == infection_id]
        id2date = dict(zip(col_dates['admission_id'], col_dates['collection_date']))
        data = data_raw.copy()
        data['collection_date'] = data['admission_id'].apply(lambda x: id2date[int(x)] if int(x) in id2date else np.nan)
        data[time_column] = pd.to_datetime(data[time_column])
        data['collection_date'] = pd.to_datetime(data['collection_date'])
        data = data[data[time_column] <= data['collection_date'] - datetime.timedelta(days=1)]
        data['infection_id'] = infection_id
        data_out = pd.concat([data_out, data], axis=0)

    data_out = data_out.sort_values(['patient_id', 'admission_id', 'infection_id', time_column]).set_index(
        ["patient_id", "admission_id", "infection_id"])

    code2idx = dict(zip(data_out["procedure_code"].unique(), range(1, data_out["procedure_code"].nunique() + 1)))
    data_out["code_idx"] = data_out["procedure_code"].apply(lambda x: code2idx[x])
    data_out = data_out.drop(columns=["index"])
    procedure_codes = dict()
    for ind in tqdm(data_out.index.unique()):
        procedure_codes[ind] = data_out.loc[ind]["code_idx"].unique()

    with open(os.path.join(processed_data_path, 'deep_procedure_codes.pickle'), 'wb') as f:
        pickle.dump(procedure_codes, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # paths and global variables
    resample_window = '4H'  # None or '1H'
    time_window = 5  # days

    process_static()
    process_comorb(maxlen=300)
    graph_embedding(embed_size=128)
    process_vitals()
    process_labs()
    combine_timeseries()
    process_procedure()
