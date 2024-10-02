import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from os.path import join

# path
raw_data_dir = '../cohort_3/'
save_data_dir = '../data_prepared/'
analysis_dir = '../manual_tables/'


def load_data(select_feats=False, feats='last', full_comorb=False):
    # initial table
    labels_raw = pd.read_csv(join(save_data_dir, 'df_label_full.csv'))
    data_table = labels_raw[['patient_id', 'admission_id', 'infection_id', 'SS', 'RS', 'RR', 'UN', 'GNB']]
    data_table = data_table.assign(
        admission_instance=data_table['admission_id'].astype(str) + '-' + data_table['infection_id'].astype(str))

    # load demographics
    demo = pd.read_csv(join(save_data_dir, 'data_demographics.csv'))
    data_table = data_table.merge(demo, how='left', on=['patient_id', 'admission_id'])

    # add vasopressors
    vasop = pd.read_csv(join(save_data_dir, 'data_last_vasop_by_instance.csv'))
    data_table = data_table.merge(vasop, how='left', on=['patient_id', 'admission_id', 'admission_instance'])
    data_table = data_table.fillna(value=0)

    # add abx
    abx = pd.read_csv(join(save_data_dir, 'data_last_abx_by_instance.csv'))
    data_table = data_table.merge(abx, how='left', on=['patient_id', 'admission_id', 'admission_instance'])
    data_table = data_table.fillna(value=0)

    # add intubation
    labels_raw['start_date'] = pd.to_datetime(labels_raw['admit_date']).dt.date
    labels_raw['end_date'] = pd.to_datetime(labels_raw['collection_date']).dt.date
    labels_raw['date_diff'] = (pd.to_datetime(labels_raw['end_date']) - pd.to_datetime(
        labels_raw['start_date'])) / np.timedelta64(1, 'D')
    proc = pd.read_csv('../cohort_3/cohort3_procedures.csv')
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
    pneumonia = pd.read_csv(join(save_data_dir, 'data_diagnoses.csv'))
    pneumonia = pneumonia.rename(columns={'instance_id': 'admission_instance'})
    data_table = data_table.merge(pneumonia[['admission_instance', 'pneumonia_community', 'pneumonia_acquired']],
                                  how='left', on='admission_instance')
    data_table[['pneumonia_community', 'pneumonia_acquired']] = data_table[
        ['pneumonia_community', 'pneumonia_acquired']].astype(int)

    # add time intervals
    interval = pd.read_csv(join(save_data_dir, 'data_diagnoses.csv'))
    interval = interval.rename(columns={'instance_id': 'admission_instance'})
    data_table = data_table.merge(interval[['admission_instance', 'time_since_admission']], how='left',
                                  on='admission_instance')

    # add vitals
    vitals = pd.read_csv(join(save_data_dir, 'data_last_min_max_vitals_by_instance.csv'))
    # feature selection
    selection = pd.read_csv(join(analysis_dir, 'vital_variables_mcvg.csv'))
    to_keep = selection[selection['KEEP'] == 1].NAME.tolist() if select_feats else selection.NAME.tolist()
    to_keep = ['vital_' + x for x in to_keep]
    if feats == 'minmax':
        to_keep = [x + '_min' for x in to_keep] + [x + '_max' for x in to_keep]
    elif feats == 'all':
        to_keep = to_keep + [x + '_min' for x in to_keep] + [x + '_max' for x in to_keep]
    vitals = vitals[['admission_instance'] + to_keep]
    data_table = data_table.merge(vitals, how='left', on=['admission_instance'])
    vital_cols = vitals.columns[1:]
    data_table[vital_cols] = data_table[vital_cols].fillna(value=0)

    # add labs
    labs = pd.read_csv(join(save_data_dir, 'data_last_min_max_labs_by_instance.csv'))
    dup_cols = list(set(vitals.columns[1:].to_list()) & set(labs.columns[1:].to_list()))  # duplicated columns
    labs = labs.drop(columns=dup_cols)  # duplicate with vitals
    # feature selection
    selection = pd.read_csv(join(analysis_dir, 'lab_variables_mcvg.csv'))
    to_keep = selection[selection['KEEP'] == 1].NAME.tolist() if select_feats else selection.NAME.tolist()
    to_keep = ['lab_' + x for x in to_keep]
    if feats == 'minmax':
        to_keep = [x + '_min' for x in to_keep] + [x + '_max' for x in to_keep]
    elif feats == 'all':
        to_keep = to_keep + [x + '_min' for x in to_keep] + [x + '_max' for x in to_keep]
    labs = labs.iloc[:, labs.columns.isin(['admission_instance'] + to_keep)]
    data_table = data_table.merge(labs, how='left', on=['admission_instance'])
    data_table = data_table.fillna(value=0)

    # data normalization
    normalizer = preprocessing.MinMaxScaler()
    tmp = normalizer.fit_transform(data_table.iloc[:, 10:])
    data_table.iloc[:, 10:] = tmp

    # load comorbidities
    if full_comorb:
        comorb = pd.read_csv(join(save_data_dir, 'data_comorb.csv'))
        data_table = data_table.merge(comorb, how='left', on=['patient_id', 'admission_id'])
        data_table = data_table.fillna(value=0)
    else:
        comorb = pd.read_csv(join(save_data_dir, 'data_comorb_fam.csv'))
        data_table = data_table.merge(comorb, how='left', on=['patient_id', 'admission_id'])
        data_table = data_table.fillna(value=0)

    data_table = data_table.set_index('admission_instance')

    return data_table


def RandomizedGroupKFold(groups, n_splits, random_state=None):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.
    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(random_state).shuffle(unique)
    result = []
    for split in np.array_split(unique, n_splits):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result


def instance_filter(instances, mode='all'):
    if mode == 'initial':
        return [ele for ele in instances if ele.split('-')[1] == '0']
    elif mode == 'subsequent':
        return [ele for ele in instances if ele.split('-')[1] != '0']
    elif mode == 'all':
        return instances
    else:
        raise NotImplementedError("Please specify data mode!")


def select_subgroup(df_data, group='1'):
    if group == '0':
        return df_data.index  # white
    elif group == '1':
        return df_data[df_data.race_1 == 1].index  # white
    elif group == '2':
        return df_data[df_data.race_2 == 1].index  # black
    elif group == '3':
        return df_data[df_data.age_yrs >= 65 / 121.0].index
    elif group == '4':
        return df_data[df_data.age_yrs < 65 / 121.0].index
    elif group == '5':
        J15 = df_data.filter(like='J15', axis=1).sum(axis=1)
        return df_data[J15 > 0].index
    elif group == '6':
        A41 = df_data.filter(like='A41', axis=1).sum(axis=1)
        return df_data[A41 > 0].index
    elif group == '7':
        cols = [col for col in df_data.columns if 'A41' in col or 'B96' in col or 'J15' in col or 'Z16' in col]
        if_selected = ~df_data[cols].sum(axis=1).astype(bool)
        return df_data[if_selected].index
    elif group == '8':
        return df_data[
            np.logical_and(df_data.age_yrs < 45 / 121.0, df_data.filter(like='N10', axis=1).sum(axis=1) > 0)].index
    elif group == '9':
        # any comorbidities in C81-C96
        if_selected = (df_data.filter(regex='C8').sum(axis=1) + df_data.filter(regex='C9').sum(axis=1)).astype(bool)
        return df_data[if_selected].index
    elif group == '10':
        return df_data[df_data.filter(like='Z94', axis=1).sum(axis=1) > 0].index
    elif group == '11':
        comorb = pd.read_csv('../cohort_3/cohort3_diagnoses_for_comorbidities.csv', low_memory=False)
        pids = comorb[comorb.ICDX_DIAGNOSIS_CODE == 'K70.30'].reference_no.unique().tolist()
        instances = pd.read_csv('../data_analysis/instance_to_patient_id.csv')
        inst = instances[instances.patient_id.isin(pids)].admission_id.astype(str).unique().tolist()
        df = df_data[['age_yrs']]
        df['admission_instance'] = df.index
        df['admission_id'] = df.admission_instance.apply(lambda x: x.split('-')[0])
        return df_data[df.admission_id.isin(inst)].index
    elif group == '12':
        vasop = pd.read_csv(join(save_data_dir, 'data_last_vasop_by_instance.csv')).drop(
            columns=['patient_id', 'admission_id'])
        vasop_names = vasop.columns[1:]
        if_selected = df_data[vasop_names].sum(axis=1).astype(bool)
        return df_data[if_selected].index
